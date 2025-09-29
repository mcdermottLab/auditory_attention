import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as ta_F
import yaml
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from src.spatial_attn_lightning import BinauralAttentionModule

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StageFeatureExtractor(nn.Module):
    """
    Wraps a model and extracts features at a chosen internal stage using a forward hook.
    """

    def __init__(self, backbone: nn.Module, target_stage: str):
        super().__init__()
        self.backbone = backbone
        self.target_stage = target_stage
        self._features = None

        # Register hook at target stage
        for name, module in self.backbone.named_modules():
            if target_stage in name:
                module.register_forward_hook(self._hook)
                break
        else:
            raise ValueError(f"Stage {target_stage} not found in backbone")

    def _hook(self, module, input, output):
        self._features = output

    def forward(self, **inputs):
        _ = self.backbone(**inputs)  # pass everything to backbone
        return self._features


class MultiClassifierModule(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        audio_transforms: nn.Module,
        target_stage: str,
        layer_idx: int,
        task_dict: Dict[str, int],
        lr_dict: Dict[str, float],
        save_dir: Optional[str] = "linear_readout_weights",
        save_outputs: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.task_dict = task_dict
        self.task_names = list(task_dict.keys())
        self.audio_transforms = audio_transforms
        self.automatic_optimization = False
        self.layer_idx = layer_idx
        self.lr_dict = lr_dict

        # Feature extractor
        self.feature_extractor = StageFeatureExtractor(backbone, target_stage)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # Infer feature dim
        with torch.no_grad():
            dummy = torch.randn(1, 2, 110_250, device="cuda")
            feats = self.feature_extractor(mixture=dummy)
            # delete dummy
            del dummy
            feat_dim = feats.view(1, -1).shape[1]

        # Classifier heads
        self.classifier_heads = nn.ModuleDict(
            {
                task: nn.Linear(feat_dim, num_classes)
                for task, num_classes in task_dict.items()
            }
        )

        self.f0_bin_counts = {}
        self.count_f0_bins = False  # Flag to control f0 bin counting

        # Tracking best validation accuracy per task and running sums
        self.best_val_acc: Dict[str, float] = {
            task: 0.0 for task in self.task_names
        }
        self.val_loss_sum: Dict[str, float] = {task: 0.0 for task in self.task_names}
        self.val_total: Dict[str, int] = {task: 0 for task in self.task_names}
        
        # Early stopping tracking based on validation accuracy
        self.val_acc_no_improve_count: Dict[str, int] = {task: 0 for task in self.task_names}
        self.training_stopped: Dict[str, bool] = {task: False for task in self.task_names}

        # Where to save best-performing classifier head weights
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_outputs = save_outputs

    def forward(self, return_f0_labels: bool = True, **inputs):
        f0_labels = None
        if "num_f0_bins" in self.task_names and return_f0_labels:
            f0_labels = self.get_f0_labels(inputs["mixture"])
        feats = self.feature_extractor(**inputs)
        return feats.view(feats.size(0), -1), f0_labels

    def training_step(self, batch, batch_idx):
        cue_features, cue_mask_ixs, loc_task_ixs, scene_features, labels = batch
        # Map task_name to correct label index
        targets = {}
        for task_name in self.task_names:
            if task_name == "num_word_classes":
                targets[task_name] = labels[:, 0].long().view(-1)
            elif task_name == "num_azim_classes":
                targets[task_name] = labels[:, 1].long().view(-1)
            elif task_name == "num_f0_bins":
                targets[task_name] = None
            else:
                raise ValueError(f"Unknown task_name: {task_name}")
        feats, f0_labels = self(mixture=scene_features, return_f0_labels=True)
        if "num_f0_bins" in self.task_names:
            targets["num_f0_bins"] = f0_labels

        # Only count f0 bins during training and when counting is enabled
        if self.count_f0_bins and self.training:
            for f0_label in f0_labels:
                label_val = f0_label.item()
                self.f0_bin_counts[label_val] = (
                    self.f0_bin_counts.get(label_val, 0) + 1
                )

        for i, task_name in enumerate(self.task_names):
            # Skip training if this task has stopped early
            if self.training_stopped[task_name]:
                continue
                
            if len(self.task_names) > 1:
                optimizer = self.optimizers()[i]
            else:
                optimizer = self.optimizers()
            optimizer.zero_grad()
            head = self.classifier_heads[task_name]
            logits = head(feats)
            loss = F.cross_entropy(logits, targets[task_name])

            self.manual_backward(loss)
            optimizer.step()

            self.log(f"train_loss_{task_name}", loss, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        cue_features, cue_mask_ixs, loc_task_ixs, scene_features, labels = batch
        targets = {}
        for task_name in self.task_names:
            if task_name == "num_word_classes":
                targets[task_name] = labels[:, 0].long().view(-1)
            elif task_name == "num_azim_classes":
                targets[task_name] = labels[:, 1].long().view(-1)
            elif task_name == "num_f0_bins":
                targets[task_name] = None
            else:
                raise ValueError(f"Unknown task_name: {task_name}")
        # Do not update f0_bin_counts in validation
        feats, f0_labels = self(mixture=scene_features, return_f0_labels=True)
        if "num_f0_bins" in self.task_names:
            targets["num_f0_bins"] = f0_labels
            # get indices of f0 bins that are >=23 and <=8
            f0_bin_ixs = (f0_labels >= 23) & (f0_labels <= 8)

        for task_name in self.task_names:
            # Skip validation if this task has stopped early
            if self.training_stopped[task_name]:
                continue
                
            logits = self.classifier_heads[task_name](feats)
            if task_name == "num_f0_bins":
                # remove f0 bins that are >=23 and <=8
                logits = logits[~f0_bin_ixs]
                targets[task_name] = targets[task_name][~f0_bin_ixs]

            loss = F.cross_entropy(logits, targets[task_name])

            # accumulate loss sum and counts per task for epoch-level averaging
            with torch.no_grad():
                total = targets[task_name].numel()
                self.val_loss_sum[task_name] += float(loss.item()) * int(total)
                self.val_total[task_name] += int(total)

                # Log accuracy per task
                if not hasattr(self, "val_correct"):
                    self.val_correct = {}
                if task_name not in self.val_correct:
                    self.val_correct[task_name] = 0

                # Only compute accuracy if targets are not None
                if targets[task_name] is not None:
                    preds = torch.argmax(logits, dim=1)
                    correct = (preds == targets[task_name]).sum().item()
                    self.val_correct[task_name] += correct

    def on_train_epoch_end(self):
        # Save f0 bin counts after the first epoch and disable counting
        if self.current_epoch == 0 and self.count_f0_bins:
            self._save_f0_bin_counts()
            self.count_f0_bins = False  # Disable counting after first epoch

    def _save_f0_bin_counts(self):
        """Save f0 bin counts to a file after the first training epoch."""
        safe_stage = str(self.layer_idx).replace("/", "_")
        file_name = f"layer_{safe_stage}_f0_bin_counts_epoch_0.json"
        save_path = self.save_dir / file_name

        with open(save_path, "w") as f:
            json.dump(self.f0_bin_counts, f, indent=2)

        print(f"Saved f0 bin counts to {save_path}")

    def on_validation_epoch_end(self):
        # compute per-task average validation loss and accuracy for the epoch, save best head if improved (higher accuracy is better)
        for task_name in self.task_names:
            # Skip if this task has stopped early
            if self.training_stopped[task_name]:
                continue
                
            total = self.val_total.get(task_name, 0)
            if total == 0:
                continue
            avg_loss = self.val_loss_sum[task_name] / float(total)
            self.log(
                f"val_avg_loss_{task_name}", avg_loss, prog_bar=False, sync_dist=True
            )

            # Compute validation accuracy if possible
            # We'll need to accumulate correct predictions and total for each task during validation_step
            correct = getattr(self, "val_correct", {}).get(task_name, None)
            if correct is not None:
                val_acc = correct / float(total)
                self.log(
                    f"val_acc_{task_name}", val_acc, prog_bar=False, sync_dist=True
                )

                # Check for improvement and early stopping based on validation accuracy
                if val_acc > self.best_val_acc.get(task_name, 0.0):
                    self.best_val_acc[task_name] = val_acc
                    self.val_acc_no_improve_count[task_name] = 0  # Reset counter
                    if self.save_outputs:
                        self._save_best_head_weights(task_name, val_acc)
                else:
                    self.val_acc_no_improve_count[task_name] += 1
                    
                # Check if we should stop training this task
                if self.val_acc_no_improve_count[task_name] >= 5:
                    self.training_stopped[task_name] = True
                    print(f"Early stopping for task {task_name} - no accuracy improvement for 5 consecutive validation checks")

        # Check if all tasks have stopped - if so, stop the entire training job
        if all(self.training_stopped.values()):
            print("All tasks have stopped early - stopping training job")
            raise KeyboardInterrupt("All tasks stopped early")

        # reset counters for next epoch
        for task_name in self.task_names:
            if not self.training_stopped[task_name]:
                self.val_loss_sum[task_name] = 0.0
                self.val_total[task_name] = 0
                if hasattr(self, "val_correct"):
                    self.val_correct[task_name] = 0

    def _save_best_head_weights(self, task_name: str, val_acc: float):
        head = self.classifier_heads[task_name]
        # file path: save under save_dir/layer_{ix}/{task_name}_best.pth
        try:
            layer_ix = int(self.layer_idx)
        except Exception:
            # fallback to target_stage text if layer_idx is unavailable
            layer_ix = str(self.layer_idx).replace("/", "_")
        save_subdir = self.save_dir / f"layer_{layer_ix}"
        save_subdir.mkdir(parents=True, exist_ok=True)
        file_name = f"{task_name}_best.pth"
        save_path = save_subdir / file_name
        save_data = {
            "state_dict": head.state_dict(),
            "val_acc": self.best_val_acc[task_name],
            "epoch": self.current_epoch,
        }
        torch.save(save_data, save_path)

    def configure_optimizers(self):
        return [
            torch.optim.Adam(
                self.classifier_heads[task].parameters(), lr=self.lr_dict[task]
            )
            for task in self.task_names
        ]

    def predict_step(self, batch, batch_idx):
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        feats, _ = self(mixture=x, return_f0_labels=False)
        return {
            task: torch.softmax(self.classifier_heads[task](feats), dim=1)
            for task in self.task_names
        }

    def get_f0_labels(self, batch):
        f0s = self.estimate_batch_f0(batch)
        f0_bins = torch.floor(12 * torch.log2(f0s / 50)).to(torch.int64)
        return f0_bins

    def estimate_batch_f0(self, batch):
        pitches = ta_F.detect_pitch_frequency(
            batch, sample_rate=44_100, freq_low=50, freq_high=300
        )
        pitches = pitches.mean(dim=(-2, -1))
        return pitches


def list_stage_names(model: nn.Module):
    """
    Print all available module names inside a model so you can pick a target stage.
    """
    print("Available stages in model:\n")
    for name, module in model.named_modules():
        print(f"{name:30s} ({module.__class__.__name__})")


def list_act_names(model: nn.Module):
    """
    Print all available module names inside a model so you can pick a target stage.
    """
    # print("Available stages in model:\n")
    name_list = []
    for name, module in model.named_modules():
        if "ReLU" in module.__class__.__name__:
            # print(f"{name:30s} ({module.__class__.__name__})")
            # get only the section with conv_block_X
            if "conv" in name:
                name = re.search(r"conv_block_[0-9]+", name).group(0)
            elif "relufc" in name:
                name = "relufc"
            name_list.append(name)
    return name_list


def format_stage_name(name: str):
    """
    Convert a stage name to a valid Python attribute name.
    """
    if "conv" in name:
        name = f"_orig_mod.model_dict.{name}.2"  # 2 is the ReLU layer after conv
    elif "fc" in name:
        name = "_orig_mod.relufc"
    return name


class ModelWithFrontEnd(nn.Module):
    def __init__(self, front_end, model):
        super().__init__()
        self.front_end = front_end
        self.model = model

    def forward(
        self,
        cue: torch.Tensor = None,
        mixture: torch.Tensor = None,
        cue_mask_ixs: torch.tensor = None,
    ):
        cue, mixture = self.front_end(cue, mixture)
        return self.model(cue, mixture, cue_mask_ixs)


def main():
    parser = argparse.ArgumentParser(
        description="Train linear readout classifiers on backbone features"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        required=True,
        help="Index of the backbone layer to extract features from",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["num_word_classes", "num_azim_classes", "num_f0_bins"],
        help="Tasks to train",
    )
    parser.add_argument(
        "--save_outputs", action="store_true", help="Save outputs of the model"
    )
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    ckpt_path = Path(
        "attn_cue_models/word_task_v10_main_feature_gain_config/checkpoints/epoch=1-step=24679-v1.ckpt"
    )

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    config["num_workers"] = min(10, 32 // 2)
    config["corpus"]["return_azim_loc_only"] = True
    config["corpus"]["feature_gain"] = True
    config["corpus"]["clean_percentage"] = 1.0

    config["model"]["num_classes"]["num_locs"] = 72
    # Step 1: Create model architecture
    module = BinauralAttentionModule(config=config).cuda()

    # Step 2: Load the checkpoint manually
    checkpoint = torch.load(ckpt_path, map_location="cuda")

    # Step 3: Load the weights only (ignore compilation and trainer states)
    module.load_state_dict(checkpoint["state_dict"], strict=False)

    cochgram = module.coch_gram
    cnn = module.model
    audio_transforms = module.audio_transforms
    backbone = ModelWithFrontEnd(cochgram, cnn)

    # 2. Choose a layer index for feature extraction
    act_names = list_act_names(backbone)
    if args.layer_idx >= len(act_names):
        raise ValueError(
            f"Layer index {args.layer_idx} out of range. Available layers: {len(act_names)}"
        )
    layer_to_get = act_names[args.layer_idx]

    # 3. Remove classifier layer to prevent interference
    backbone.model.classification = nn.Identity()

    task_dict={
            "num_word_classes": 200,
            "num_azim_classes": 72,
            "num_f0_bins": 32,
        }
    lr_dict={
        "num_word_classes": config["hparas"]["lr_word"],
        "num_azim_classes": config["hparas"]["lr_azim"],
        "num_f0_bins": config["hparas"]["lr_f0"],
    }

    # only keep tasks in task_dict if they are in args.tasks
    task_dict = {k: v for k, v in task_dict.items() if k in args.tasks}
    lr_dict = {k: v for k, v in lr_dict.items() if k in args.tasks}

    # 4. Create full Lightning model
    model = MultiClassifierModule(
        backbone=backbone,
        target_stage=layer_to_get,
        layer_idx=args.layer_idx,
        audio_transforms=audio_transforms,
        task_dict=task_dict,
        lr_dict=lr_dict,
        save_outputs=args.save_outputs,
    )

    # WandB logger setup
    if len(args.tasks) == 1:
        if args.tasks[0] == "num_word_classes":
            run_name = f"layer_{args.layer_idx}_word_lr_{config['hparas']['lr_word']}"
        elif args.tasks[0] == "num_azim_classes":
            run_name = f"layer_{args.layer_idx}_azim_lr_{config['hparas']['lr_azim']}"
        elif args.tasks[0] == "num_f0_bins":
            run_name = f"layer_{args.layer_idx}_f0_lr_{config['hparas']['lr_f0']}"
    else:
        run_name = f"layer_{args.layer_idx}_multitask_lr_word_{config['hparas']['lr_word']}_lr_azim_{config['hparas']['lr_azim']}_lr_f0_{config['hparas']['lr_f0']}"
    wandb_logger = WandbLogger(
        project="auditory_attn_linear_readout",
        name=run_name,
        group=f"layer_{args.layer_idx}",
        log_model=False,
    )

    # 5. Train
    trainer = pl.Trainer(
        # detect_anomaly=True,
        accelerator="gpu",
        devices=args.num_gpus,
        val_check_interval=config["hparas"]["valid_step"],
        profiler=None,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger,
    )
    trainer.fit(
        model,
        train_dataloaders=module.train_dataloader(),
        val_dataloaders=module.val_dataloader(),
    )


if __name__ == "__main__":
    main()

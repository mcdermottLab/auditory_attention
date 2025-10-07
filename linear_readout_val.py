import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as ta_F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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
        self.best_val_acc: Dict[str, float] = {task: 0.0 for task in self.task_names}
        self.val_loss_sum: Dict[str, float] = {task: 0.0 for task in self.task_names}
        self.val_total: Dict[str, int] = {task: 0 for task in self.task_names}

        # Where to save best-performing classifier head weights
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_outputs = save_outputs

        self.val_labels = None
        self.val_preds = None

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
                self.f0_bin_counts[label_val] = self.f0_bin_counts.get(label_val, 0) + 1

        for i, task_name in enumerate(self.task_names):
            # Skip training if this task has stopped early

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
            logits = self.classifier_heads[task_name](feats)
            if task_name == "num_f0_bins":
                # remove f0 bins that are >=23 and <=8
                logits = logits[~f0_bin_ixs]
                targets[task_name] = targets[task_name][~f0_bin_ixs]
            if self.val_labels is None:
                self.val_labels = targets[task_name]
            else:
                self.val_labels = torch.cat([self.val_labels, targets[task_name]])
            if self.val_preds is None:
                self.val_preds = torch.argmax(logits, dim=1)
            else:
                self.val_preds = torch.cat([self.val_preds, torch.argmax(logits, dim=1)])

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
        for task_name in self.task_names:
            total = self.val_total.get(task_name, 0)
            if total == 0:
                continue
            avg_loss = self.val_loss_sum[task_name] / float(total)
            self.log(f"val_loss_{task_name}", avg_loss, prog_bar=False, sync_dist=True)

            correct = getattr(self, "val_correct", {}).get(task_name, None)
            if correct is not None:
                val_acc = correct / float(total)
                self.log(f"val_acc_{task_name}", val_acc, prog_bar=True, sync_dist=True)
            out_dict = {
                "val_loss": avg_loss,
                "val_acc": val_acc,
                "val_labels": self.val_labels.cpu().tolist(),
                "val_preds": self.val_preds.cpu().tolist(),
            }
            with open(self.save_dir / f"layer_{self.layer_idx}_{task_name}_val_results.json", "w") as f:
                json.dump(out_dict, f, indent=2)
        self.val_labels = None
        self.val_preds = None

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
        description="Get linear readout validation metrics for a given layer"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        required=True,
        help="Index of the backbone layer to extract features from",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="num_word_classes",
        # options=["num_word_classes", "num_azim_classes", "num_f0_bins"],
        help="Task to evaluate",
    )
    parser.add_argument(
        "--save_outputs", action="store_true", help="Save outputs of the evaluation"
    )
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()

    config_path = Path(args.config_path)

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    config["num_workers"] = min(10, 32 // 2)
    config["corpus"]["return_azim_loc_only"] = True
    config["corpus"]["feature_gain"] = True
    config["corpus"]["clean_percentage"] = 1.0

    config["model"]["num_classes"]["num_locs"] = 72

    module = BinauralAttentionModule(config=config).cuda()

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

    task_dict = {
        "num_word_classes": 200,
        "num_azim_classes": 72,
        "num_f0_bins": 32,
    }
    lr_dict = {
        "num_word_classes": config["hparas"]["lr_word"],
        "num_azim_classes": config["hparas"]["lr_azim"],
        "num_f0_bins": config["hparas"]["lr_f0"],
    }

    # only keep tasks in task_dict if they are in args.tasks
    task_dict = {k: v for k, v in task_dict.items() if k in args.task}
    lr_dict = {k: v for k, v in lr_dict.items() if k in args.task}

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

    load_subdir = model.save_dir / f"layer_{args.layer_idx}"
    candidates = list(load_subdir.glob(f"{args.task}*.pth")) + list(load_subdir.glob(f"{args.task}*.ckpt"))
    if not candidates:
        print(f"No saved weights found for layer {args.layer_idx} at {load_subdir}; skipping load.")
    else:
        file_path = max(candidates, key=os.path.getctime)
        print(f"Loading best weights from {file_path}")
        ckpt = torch.load(file_path, map_location=model.device)
        model.load_state_dict(ckpt["state_dict"], strict=True)

    # WandB logger setup
    if args.task == "num_word_classes":
            run_name = f"layer_{args.layer_idx}_word_lr_{config['hparas']['lr_word']}"
    elif args.task == "num_azim_classes":
        run_name = f"layer_{args.layer_idx}_azim_lr_{config['hparas']['lr_azim']}"
    elif args.task == "num_f0_bins":
        run_name = f"layer_{args.layer_idx}_f0_lr_{config['hparas']['lr_f0']}"
    wandb_logger = WandbLogger(
        project="auditory_attn_lin_eval",
        name=run_name,
        group=f"layer_{args.layer_idx}",
        log_model=False,
    )

    # 5. Evaluate
    trainer = pl.Trainer(
        # detect_anomaly=True,
        accelerator="gpu",
        devices=args.num_gpus,
        val_check_interval=config["hparas"]["valid_step"],
        profiler=None,
        # strategy=DDPStrategy(), # find_unused_parameters=True),
        logger=wandb_logger,
        num_sanity_val_steps=0,
    )
    trainer.validate(
        model,
        dataloaders=module.val_dataloader(),
    )


if __name__ == "__main__":
    main()

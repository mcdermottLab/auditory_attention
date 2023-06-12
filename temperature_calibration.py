import torch
import yaml
import os
from src import attn_tracking_lightning
from argparse import ArgumentParser

class TemperatureScalingCalibrationModule(torch.nn.Module):

    def __init__(self, module, ckpt_path, config):
        super().__init__()
        self.model_path = ckpt_path
        self.model = module.load_from_checkpoint(checkpoint_path=self.model_path, config=config)
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, cue, mixture=None):
        outputs = self.forward_logits(cue, mixture)
        scores = torch.nn.functional.softmax(outputs, dim=-1)[:, 1]
        return scores

    def forward_logits(self, cue, mixture=None):
        logits = self.model(cue, mixture)
        return logits / self.temperature

    def fit(self, n_epochs=100, lr=1e-3):
        self.freeze_base_model()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        loader = self.model.val_dataloader()

        global_loss = float('inf')
        n_tolerance_steps = 100
        for epoch in range(n_epochs):
            for ix, batch in enumerate(loader):
                mixture, cue, label = batch
                self.zero_grad()
                predict = self.forward_logits(cue.cuda(), mixture.cuda())
                loss = criterion(predict, label.cuda())
                loss.backward()
                optimizer.step()

                if abs(loss.item() - global_loss) > 1e-3 and loss.item() < global_loss:
                    global_loss = loss.item()
                    n_tolerance_steps = 100
                elif abs(loss.item() - global_loss) < 1e-3:
                    n_tolerance_steps -= 1

                if n_tolerance_steps == 0:
                    return self
                if ix % 100 == 0:
                    print(f'Epoch {epoch} step: {ix} loss: {loss.item()}')
        return self
                

    def freeze_base_model(self):
        self.model.eval()
        for para in self.model.parameters():
            para.requires_grad = False

        return self

    def save(self, save_path):
        torch.save(self.temperature, save_path)

def cli_main():
    parser = ArgumentParser()
    module = attn_tracking_lightning.AttentionalTrackingModule
    parser.add_argument('--ckpt_path', type=str, default="/om2/user/imgriff/projects/Auditory-Attention/attn_cue_models/attn_cue_jsin_multi_distractor_w_audioset_bs_64_lr_1e-4/checkpoints/epoch=0-step=70000.ckpt")
    parser.add_argument('--config', type=str, default="config/attentional_cue/attn_cue_lr_1e-4_bs_64_constrained_slope_multi_distractor.yaml")
    parser.add_argument('--n_jobs', type=int, default=8)

    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    config = args.config
    config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
    config['n_jobs'] = args.n_jobs
    config['data']['loader']['batch_size'] = 128

    calibration_module = TemperatureScalingCalibrationModule(module, ckpt_path, config).cuda()
    calibration_module.fit(n_epochs=5, lr=1e-4)
    savepath = '/om2/user/rphess/Auditory-Attention/confidenceScores/pilot/parameters/temperatureparemeters2.pt'
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    calibration_module.save(savepath)
    # print('# of parameters:', sum(p.numel() for p in calibration_module.parameters() if p.requires_grad))

if __name__ == "__main__":
    cli_main()

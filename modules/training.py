from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import logging
import sys
import argparse
from modules.model import SpatioTemporalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/model.log"),  # Log to this file
        logging.StreamHandler(),  # Log to console
    ],
)
logger = logging.getLogger(__name__)

# TODO: refactor this class to be in preprocessing or new module just for data
class ClimateDataset(Dataset):
    def __init__(self, data):
        # non_target_cols = ["lon", "lat", "aod", "date", "date_index", "T"]
        # target_cols = [col for col in data.columns if col not in non_target_cols] 
        target_cols = ["T_scaled"]
        feature_cols = ["lon", "lat", "aod"]
        features = data[feature_cols].copy()
        targets = data[target_cols].copy()
        indices = data["date_index"].copy()

        # self.scaler = StandardScaler()
        # scaled_features = self.scaler.fit_transform(features)

        self.features = torch.from_numpy(features.values).float()
        self.targets = torch.from_numpy(targets.values).float()
        self.indices = torch.from_numpy(indices.values).long()

        # standard scale features
        # ? should I keep features saved if I use only scaled features
        self.scaled_features = (self.features - self.features.mean(dim=0) ) / self.features.std(dim=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.scaled_features[idx]
        targets = self.targets[idx]
        indices = self.indices[idx]

        return features, targets, indices

    # -NOTE: maybe add inverse transofrm method for sclaer
    # TODO: add scaler functions for features

class OneCycleScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, last_step=-1):
        super().__init__(optimizer, last_step)

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.half_step = total_steps // 2

    def get_lr(self):
        if self.last_step < self.half_step:
            return [
                (base_lr + (self.max_lr - base_lr) * (self.last_step / self.half_step))
                for base_lr in self.base_lrs
            ]
        elif self.last_step < 2 * self.half_step:
            return [
                (
                    self.max_lr
                    - (self.max_lr - base_lr)
                    * ((self.last_step - self.half_step) / self.half_step)
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [
                (
                    base_lr
                    - (
                        base_lr
                        * (self.half_step - 2 * self.half_step)
                        / (self.total_steps - 2 * self.half_step)
                    )
                )
                for base_lr in self.base_lrs
            ]


class TrainingApp:
    # TODO: need to add test/dataset in the init
    def __init__(self, model, optimizer, training_data, sys_argv=None) -> None:
        self.cli_args = self.parse_cli_args(sys_argv)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model)
        self.optimizer = optimizer

    @staticmethod
    def parse_cli_args(sys_argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-workers", default=8, type=int)
        parser.add_argument("--epochs", default=1, type=int)
        parser.add_argument("--batch-size", default=32, type=int)

        return parser.parse_args(sys_argv)

    def init_dataloaders(self, training_data):
        #  *--NOTE: will focus on train for now
        train_dataset = ClimateDataset(training_data)

        batch_size = self.cli_args.batch_size
        if self.device.type == "cuda":
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # ? does shuffle matter?
            shuffle=True,
            num_workers=self.cli_args.num_workers.batch_sampler,
            pin_memory=self.device.type == "cuda",
        )

        # TODO: add validation section

        return train_dl

    def train_one_epoch(self, epoch_index, train_dl):
        self.model.train()

        for batch_idx, (features, targets, time_indices) in enumerate(train_dl):
            self.optimizer.zero_grad()
            preds = self.model(features, time_indices)

    # def compute_closs(self, preds, targets):
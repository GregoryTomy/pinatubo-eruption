from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import torch
import torch.nn as nn
import logging
import numpy as np
import sys
import os
import time
import datetime as dt
import argparse
from modules.model import SpatioTemporalModel
from modules.utils import setup_logger

logger = setup_logger("training", "logs/training.log", logging.INFO)


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
        self.scaled_features = (
            self.features - self.features.mean(dim=0)
        ) / self.features.std(dim=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.scaled_features[idx]
        targets = self.targets[idx]
        indices = self.indices[idx]

        return (features, targets, indices)

    # -NOTE: maybe add inverse transofrm method for sclaer


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
        self.model = self.load_model(model)
        self.optimizer = optimizer
        self.training_data = training_data
        self.loss_func = nn.MSELoss(reduction="none")

    @staticmethod
    def parse_cli_args(sys_argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-workers", default=8, type=int)
        parser.add_argument("--epochs", default=1, type=int)
        parser.add_argument("--batch-size", default=32, type=int)
        parser.add_argument("--name", default="train", type=str)
        parser.add_argument("--val-perct", default=None, type=float)
        # ! provide argument for setting seed

        return parser.parse_args(sys_argv)

    def load_model(self, model):
        if self.device.type == "cuda":
            logger.info(f"Using CUDA with {torch.cuda.device_count()} devices")

            # distribute over mult/batchple GPUs if available
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def init_dataloaders(self, seed=1):
        dataset = ClimateDataset(self.training_data)

        if self.cli_args.val_perct is not None:
            np.random.seed(seed)
            idx = np.arange(len(dataset))
            val_idx = np.random.choice(
                idx,
                size=int(len(dataset) * self.cli_args.val_perct),
                replace=False,
            )
            train_idx = np.setdiff1d(idx, val_idx, assume_unique=True)
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
        else:
            train_dataset = dataset
            val_dataset = None

        batch_size = self.cli_args.batch_size
        if self.device.type == "cuda":
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # ? does shuffle matter?
            shuffle=True,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        val_dl = None
        if val_dataset is not None:
            val_dl = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.device.type == "cuda",
            )

        return train_dl, val_dl

    def train_one_epoch(self, epoch_index, train_dl):
        self.model.train()

        train_metrics = torch.zeros(len(train_dl.dataset), device=self.device)
        start_time = time.time()

        for batch_idx, batch_tuple in enumerate(train_dl):
            self.optimizer.zero_grad()

            loss = self.compute_batch_loss(batch_idx, batch_tuple, train_metrics)
            loss.backward()
            self.optimizer.step()

            # ? move to train function?
            if batch_idx % 100 == 0:
                log_progress(epoch_index, batch_idx, len(train_dl), start_time)

        # average_loss = train_metrics.sum() / len(train_dl.dataset)
        # logger.info(
        #     f"Average training loss for epoch {epoch_index}: {average_loss.item()}"
        # )

        return train_metrics.to("cpu")

    def validate_one_epoch(self, epoch_index, val_dl):
        self.model.eval()

        val_metrics = torch.zeros(len(val_dl.dataset), device=self.device)
        start_time = time.time()

        # no gradients needed for validation
        with torch.no_grad():
            for batch_idx, batch_tuple in enumerate(val_dl):
                loss = self.compute_batch_loss(batch_idx, batch_tuple, val_metrics)

                # if batch_idx % 10 == 0:
                #     log_progress(epoch_index, batch_idx, len(val_dl), start_time)

        # average_loss = val_metrics.sum() / len(val_dl.dataset)
        # logger.info(
        #     f"Average validation loss for epoch {epoch_index}: {average_loss.item()}"
        # )

        return val_metrics.to("cpu")

    def compute_batch_loss(self, batch_idx, batch_tuple, metrics):
        features, targets, time_indices = batch_tuple

        features = features.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        time_indices = time_indices.to(self.device, non_blocking=True)

        preds = self.model(features, time_indices)
        loss = self.loss_func(preds, targets)
        batch_size = features.size(0)
        start_idx = batch_idx * batch_size
        # handle the case where the last batch might be smaller than batch size
        end_idx = start_idx + targets.size(0)
        # detach as metrics don't need to hold gradients
        metrics[start_idx:end_idx] = loss.detach().squeeze()

        return loss.mean()

    def train(self):
        logger.info(f"Starting training with {type(self).__name__, self.cli_args}")

        train_dl, val_dl = self.init_dataloaders()

        for epoch in range(1, self.cli_args.epochs + 1):
            train_metrics = self.train_one_epoch(epoch, train_dl)

            if val_dl is not None:
                val_metrics = self.validate_one_epoch(epoch, val_dl)
            else:
                val_metrics = None
            
            log_metrics(epoch, train_metrics, val_metrics)

        # check if directory exits
        model_directory = "saved_models"
        os.makedirs(model_directory, exist_ok=True)

        # save the model after training is complete
        model_path = os.path.join(model_directory, f"model_{self.cli_args.name}.pth")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")


################################################
############## Helper functions ################
#################################################


def log_progress(epoch_idx, batch_idx, num_batches, start_time):
    elapsed_time = time.time() - start_time
    batches_left = num_batches - batch_idx
    estimated_total_time = elapsed_time / (batch_idx + 1) * num_batches
    estimated_end_time = start_time + estimated_total_time
    estimated_time_left = estimated_end_time - time.time()

    end_time_str = dt.datetime.fromtimestamp(estimated_end_time).strftime(
        "'%Y-%m-%d %H:%M:%S'"
    )
    estimated_time_left_str = str(dt.timedelta(seconds=estimated_time_left)).split(".")[
        0
    ]

    logger.info(
        f"Epoch {epoch_idx}, Batch {batch_idx}/{num_batches}: "
        f"{batches_left} batches left, "
        f"Estimated completion at {end_time_str}, "
        f"Time left: {estimated_time_left_str}"
    )

def log_metrics(epoch_idx, train_metrics, val_metrics=None):
    train_loss = train_metrics.mean().item()
    val_loss = val_metrics.mean().item() if val_metrics is not None else None

    message = f"Epoch {epoch_idx}, Training Loss: {train_loss:.4f}"
    if val_loss is not None:
        message += f", Validation Loss: {val_loss:.4f}"

    logger.info(message)


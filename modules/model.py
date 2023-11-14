import pandas as pd
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from modules.utils import setup_logger

logger = setup_logger("model", "logs/model.log", logging.WARNING)


class Recompose(nn.Module):
    def __init__(self, climate_data_dict):
        # climate data is the saved dataclass stored in output
        super().__init__()
        self.climate_data_dict = climate_data_dict

        for phase in ["train", "val", "test"]:
            self.register_buffer(f"temporal_bases_{phase}", climate_data_dict[phase].temporal_bases)
            self.register_buffer(f"scaling_factors_{phase}", climate_data_dict[phase].scaling_factors)
            # self.register_buffer(f"time_mean_{phase}", climate_data_dict[phase].time_mean)
            # self.register_buffer(f"centering_{phase}", climate_data_dict[phase].centering)

    def forward(self, targets, batch_indices, phase):
        # phase will take the values "train", "val", or "test"
        temporal_bases = getattr(self, f"temporal_bases_{phase}")
        scaling_factors = getattr(self, f"scaling_factors_{phase}")

        # here batch indices are used to get the corresponding temporal bases
        # shape of temporal bases should be (days, locations)
        select_temp_bases = temporal_bases[batch_indices, :]


        # perform SVD recomp 
        # ! ignoring the scaling and climatology for now
        recomped = torch.mm(
            targets, torch.mm(torch.diag(scaling_factors), select_temp_bases.t())
        )

        # NOTE: the return here returns the first column. All columns should be the same
        return recomped[:, 1].view(-1, 1)


class SpatioTemporalModel(nn.Module):
    def __init__(self, input_channels, hidden_units, output_channels, climate_data_dict):
        super().__init__()
        self.network = nn.Sequential(
            # block 1
            nn.BatchNorm1d(input_channels),
            nn.Linear(input_channels, hidden_units),
            nn.ReLU(),
            # block 2
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            # block 3
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            # block 4
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            # block 5
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, output_channels),
        )
        self.recompose = Recompose(climate_data_dict)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # initalize the weights of the linear layers with Kaiming normalization
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                # --NOTE: Test regular bias initialization
                if m.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, X, indices, phase):
        pred_coeff = self.network(X)
        # logging.info(f"Auxiliary output calculated. Shape: {pred_coeff.shape}")

        recomped = self.recompose(pred_coeff, indices, phase)
        # logging.info(f"Final output recomposed. Shape: {recomped.shape}")

        return recomped

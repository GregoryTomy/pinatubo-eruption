import pandas as pd
import pickle
import torch
from modules.model import SpatioTemporalModel
from modules.training import TrainingApp

NUM_COEFFICIENTS = 30

train_data = pd.read_feather("Data/train_30.feather")
val_data = pd.read_feather("Data/val_30.feather")
test_data = pd.read_feather("Data/test_30.feather")

with open("output/climate_data_30.pkl", "rb") as f:
    climate_data_dict = pickle.load(f)


def main():
    model = SpatioTemporalModel(
        input_channels=3,
        hidden_units=30,
        output_channels=NUM_COEFFICIENTS,
        climate_data_dict=climate_data_dict,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_app = TrainingApp(
        model, optimizer, training_data=train_data, validation_data=val_data, test_data=test_data
    )
    training_app.train()


if __name__ == "__main__":
    main()

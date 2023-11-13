from dataclasses import dataclass
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ClimateData:
    # spatial_coeff: torch.Tensor = None
    temporal_bases: torch.Tensor = None
    scaling_factors: torch.Tensor = None
    time_mean: torch.Tensor = None
    centering: float = 0.0
    num_coeffs: int = None

    # def detrend_and_center(self, target):
    #     target_tensor = torch.tensor(target.values, dtype=torch.float32)
    #     n_samples, n_features = target_tensor.shape
    #     self.time_mean = target_tensor.mean(dim=0)  # climatology
    #     y = target_tensor - self.time_mean
    #     self.centering = torch.sqrt(torch.tensor(n_samples - 1, dtype=torch.float32))
    #     y = y / self.centering
    #     return y

    # ! the above works but since our scaling during recomp isnt working
    # ! I am using the detrended and centered temperature as the target.
    def detrend_and_center(self, target_df):
        self.time_mean = target_df.mean()
        y = target_df - self.time_mean
        self.time_mean = torch.tensor(self.time_mean, dtype=torch.float32)
        n_samples = len(target_df)
        self.centering = (n_samples - 1) ** 0.5
        y = y / self.centering

        return y

    def apply_svd(self, centered_tensor):
        # ! temporary casting to tensor as we switched to dataframe above
        centered_tensor = torch.tensor(centered_tensor.values, dtype=torch.float32)
        _ , self.scaling_factors, self.temporal_bases = torch.svd(
            centered_tensor
        )

        if self.num_coeffs is not None:
            # self.spatial_coeff = self.spatial_coeff[:, : self.num_coeffs]
            self.scaling_factors = self.scaling_factors[: self.num_coeffs]
            self.temporal_bases = self.temporal_bases[:, : self.num_coeffs]
            return self

        return self

    def plot_explained_variance(self):
        explained_var = self.scaling_factors**2 / (self.scaling_factors**2).sum()
        cumulative_var = np.cumsum(explained_var)
        n_components_95 = np.argmax(cumulative_var >= 0.95) + 1

        plt.figure(figsize=(14, 6))
        plt.bar(
            range(len(explained_var)),
            explained_var,
            alpha=0.5,
            label="individual explained variance",
        )
        plt.step(
            range(len(explained_var)),
            cumulative_var,
            where="mid",
            label="cumulative explained variance",
        )
        plt.axvline(
            x=n_components_95,
            color="r",
            linestyle="--",
            label=f"95% variance (n={n_components_95})",
        )
        plt.axhline(y=0.95, color="r", linestyle="--")
        plt.ylabel("Explained Variance Ratio")
        plt.xlabel("Principal Components")
        plt.legend(loc="best")
        plt.savefig("images/explained_variance.png")

    # def plot_bases_coefficients(self, lons, lats):
    #     plt.figure(figsize=(12, 8))
    #     for i in range(3):
    #         plt.subplot(3, 1, i + 1)
    #         plt.plot(self.temporal_bases[:, i].numpy())  # Convert to numpy for plotting
    #         plt.title(f"Temporal Base {i+1}")
    #         plt.xlabel("Time Index")
    #         plt.ylabel("Value")
    #     plt.tight_layout()
    #     plt.savefig("images/temporal_bases.png")

    #     plt.figure(figsize=(15, 10))
    #     for i in range(3):
    #         plt.subplot(2, 3, i + 4)
    #         plt.scatter(
    #             lons, lats, c=self.spatial_coeff[:, i].numpy(), cmap="coolwarm"
    #         )  # Convert to numpy for plotting
    #         plt.colorbar()
    #         plt.title(f"EOF {i + 1} (spatial coefficients)")
    #         plt.xlabel("Longitude")
    #         plt.ylabel("Latitude")
    #     plt.tight_layout()
    #     plt.savefig("images/spatial_coefficients.png")


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={"TOTEXTTAU": "aod"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    start_date = df["date"].min()
    df["date_index"] = (df["date"] - start_date).dt.days
    return df


def split_train_test(df, train_size, seed=9):
    np.random.seed(seed)
    train_idx = np.random.choice(
        df.index, size=int(len(df.index) * train_size), replace=False
    )
    test_idx = df.index.difference(train_idx)
    return df.loc[train_idx], df.loc[test_idx]


def prepare_nn_data(train_df, df):
    model_train_df = train_df.reset_index().melt(
        id_vars=["lon", "lat"], var_name="date", value_name="T_scaled"
    )

    nn_df = model_train_df.merge(
        df, on=["lon", "lat", "date"], how="left"
    )
    return nn_df


# def prepare_nn_data(train_df, spatial_coeff, df):
#     model_train_df = train_df.reset_index().melt(
#         id_vars=["lon", "lat"], var_name="date", value_name="T"
#     )

#     locations = np.array(train_df.index.tolist())
#     nn_df = pd.DataFrame(
#         np.concatenate([locations, spatial_coeff], axis=1),
#         columns=["lon", "lat"]
#         + [f"coeff_{i + 1}" for i in range(spatial_coeff.shape[1])],
#     )

#     nn_df = model_train_df.merge(nn_df, on=["lon", "lat"], how="left")
#     nn_df = nn_df.merge(
#         df[["lon", "lat", "date", "date_index", "aod"]],
#         on=["lon", "lat", "date"],
#         how="left",
#     )
#     return nn_df


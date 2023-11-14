from modules.preprocessing import *
import pickle


def main():
    TEST_TRAIN_SPLIT = 0.8
    NUM_COEFFICIENTS = 30

    file_path = "Data/mergedDaily.csv"
    df = load_and_preprocess_data(file_path)

    temp_data = df.pivot(index=["lon", "lat"], columns="date", values="T")

    # Note: after the pivot, we are focused on the targets alone
    y_train, y_val, y_test = split_train_test(temp_data, TEST_TRAIN_SPLIT)

    # initalize dataclass to hold data features
    # if taking full coefficients, provide no arguments to ClimateData
    train_data = ClimateData(num_coeffs=NUM_COEFFICIENTS)
    val_data = ClimateData(num_coeffs=NUM_COEFFICIENTS)
    test_data = ClimateData(num_coeffs=NUM_COEFFICIENTS)

    # detrend and center
    y_train_dc = train_data.detrend_and_center(y_train)
    y_val_dc = val_data.detrend_and_center(y_val)
    y_test_dc = test_data.detrend_and_center(y_test)

    assert len(y_train_dc) + len(y_test_dc) + len(y_val_dc) == len(temp_data)

    # apply svd
    train_data = train_data.apply_svd(y_train_dc)
    val_data = val_data.apply_svd(y_val_dc)
    test_data = test_data.apply_svd(y_test_dc)


    # create dictionary to hold dataclasses
    climate_data_dict = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    # save dataclass
    with open(f"output/climate_data_{NUM_COEFFICIENTS}.pkl", "wb") as f:
        pickle.dump(climate_data_dict, f)

    # print(data)
    # lons, lats = zip(*y_train.index)
    train_data.plot_explained_variance(name="train")
    val_data.plot_explained_variance(name="val")
    test_data.plot_explained_variance(name="test")
    # data.plot_bases_coefficients(lons, lats)

    # prepare dataframe for neural network
    #! note we are passing the detrended and centered temperatures. Will remove in final build
    test_nn = prepare_nn_data(y_test_dc, df)
    val_nn = prepare_nn_data(y_val_dc, df)
    train_nn= prepare_nn_data(y_train_dc, df)
    # print(nn_df.head())
    test_nn.to_feather(f"Data/test_{NUM_COEFFICIENTS}.feather")
    val_nn.to_feather(f"Data/val_{NUM_COEFFICIENTS}.feather")
    train_nn.to_feather(f"Data/train_{NUM_COEFFICIENTS}.feather")



if __name__ == "__main__":
    main()

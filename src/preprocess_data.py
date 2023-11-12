from modules.preprocessing import *
import pickle


def main():
    TEST_TRAIN_SPLIT = 0.8
    NUM_COEFFICIENTS = 30

    file_path = "Data/mergedDaily.csv"
    df = load_and_preprocess_data(file_path)

    temp_data = df.pivot(index=["lon", "lat"], columns="date", values="T")

    # Note: after the pivot, we are focused on the targets alone
    y_train, y_test = split_train_test(temp_data, TEST_TRAIN_SPLIT)

    # initalize dataclass to hold data features
    # if taking full coefficients, provide no arguments to ClimateData
    data = ClimateData(num_coeffs=NUM_COEFFICIENTS)
    

    # detrend and center
    y_train_dc = data.detrend_and_center(y_train)
    y_test_dc = data.detrend_and_center(y_test)

    # apply svd
    train_data = data.apply_svd(y_train_dc)

    # save dataclass
    with open(f"output/climate_data_{NUM_COEFFICIENTS}.pkl", "wb") as f:
        pickle.dump(data, f)

    # print(data)
    lons, lats = zip(*y_train.index)
    data.plot_explained_variance()
    # data.plot_bases_coefficients(lons, lats)

    # prepare dataframe for neural network
    #! note we are passing the detrended and centered temperatures. Will remove in final build
    nn_df = prepare_nn_data(y_train_dc, df)
    print(nn_df.head())
    nn_df.to_feather(f"Data/train_{NUM_COEFFICIENTS}.feather")


if __name__ == "__main__":
    main()

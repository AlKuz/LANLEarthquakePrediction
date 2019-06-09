from src.models import Model
from keras.models import load_model

import os
import pandas as pd
import numpy as np
import json


def load_data(path, names):
    for name in names:
        data = pd.read_csv(os.path.join(path, name)).values
        yield name[:-4], data


if __name__ == "__main__":
    TEST_DATA_PATH = "/home/alexander/Projects/Kaggle/LANLEarthquakePrediction/data/raw/test"
    RESULT_FILE_PATH = "/home/alexander/Projects/Kaggle/LANLEarthquakePrediction/reports"
    MODELS_PATH = "/home/alexander/Projects/Kaggle/LANLEarthquakePrediction/models"

    MODEL_NAME = "conv_rnn_model_v4"

    model = load_model(os.path.join(MODELS_PATH, MODEL_NAME + ".hdf5"))
    with open(os.path.join(MODELS_PATH, MODEL_NAME + ".json"), 'r') as file:
        model_settings = json.load(file)

    names = os.listdir(TEST_DATA_PATH)
    length = len(names)
    test_data = load_data(TEST_DATA_PATH, names)

    RESULT_FILE = RESULT_FILE_PATH + "/{}.csv".format(MODEL_NAME)
    with open(RESULT_FILE, 'w') as file:
        file.write("seg_id,time_to_failure\n")

    i = 0
    for name, data in test_data:
        data_to_predict = np.expand_dims(data, axis=0)
        result = model.predict(data_to_predict)[0, 0]
        with open(RESULT_FILE, 'a') as file:
            file.write("{},{}\n".format(name, result))
        if i % 10 == 0:
            print("{} / {}".format(i, length))
        i += 1

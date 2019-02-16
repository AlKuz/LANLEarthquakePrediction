"""
Making dataset
"""

import pandas as pd
import numpy as np
import pickle
import random
import string


def generate_data(data_path, path_to_save, timesteps_to_generate, seed=13):
    random.seed(a=seed)
    for chunk in pd.read_csv(data_path, chunksize=timesteps_to_generate):
        data: np.ndarray = chunk.values
        timeseries = (data[:, 0]).astype('int16')
        time = (data[-1, 1]).astype('float32')
        data_dict = {
            'timeseries': timeseries,
            'time': time
        }
        filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16)) + '.pkl'
        with open(path_to_save + filename, 'wb') as f:
            pickle.dump(data_dict, f)


if __name__ == "__main__":
    TRAIN_DATA_PATH = "data/raw/train.csv"
    RESULT_DATA_FOLDER = "data/processed/"

    timesteps = 150000
    generate_data(TRAIN_DATA_PATH, RESULT_DATA_FOLDER, timesteps_to_generate=timesteps)

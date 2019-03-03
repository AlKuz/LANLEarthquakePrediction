"""
Making dataset
"""

import pandas as pd
import numpy as np
import pickle
import random
import string


def generate_data(data_path, path_to_save, timesteps_to_generate, step, seed=13, train_val_test=(0.7, 0.15, 0.15)):
    """
    Generate shuffled data
    :param data_path: path to the original csv data
    :param path_to_save: path to interim folder
    :param timesteps_to_generate: size of timeseries
    :param step: step to shift timeseries through the original timeseries data
    :param seed: seed to generate tameseries name
    :param train_val_test: parts to train, validation and test datasets
    :return: None
    """
    assert sum(train_val_test) == 1

    TIMESTEPS_IN_ALL = 629145480
    NUMBER_BLOCKS = (TIMESTEPS_IN_ALL - timesteps_to_generate + step) // step
    random.seed(a=seed)

    train_data_dict = dict()
    val_data_dict = dict()
    test_data_dict = dict()

    counter = 0
    for chunk in pd.read_csv(data_path, chunksize=timesteps_to_generate * 10):

        try:
            data: np.ndarray = np.concatenate([data, chunk.values])
        except:
            data: np.ndarray = chunk.values

        while data.shape[0] >= timesteps_to_generate:

            timeseries = (data[:timesteps_to_generate, 0]).astype('int16')
            time = (data[timesteps_to_generate-1, 1]).astype('float32')
            data = data[step:, ...]

            random_number = random.uniform(0, 1)
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
            while name in train_data_dict or name in val_data_dict or name in test_data_dict:
                name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
            if 0 < random_number <= train_val_test[0]:
                train_data_dict[name] = (time, timeseries)
            elif train_val_test[0] < random_number <= sum(train_val_test[:2]):
                val_data_dict[name] = (time, timeseries)
            else:
                test_data_dict[name] = (time, timeseries)

            counter += 1
            if counter % 100 == 0:
                print("{} / {} blocks were processed".format(counter, NUMBER_BLOCKS))

    with open(path_to_save + 'train.pkl', 'wb') as f:
        pickle.dump(train_data_dict, f)
    print("Train data was saved")

    with open(path_to_save + 'validation.pkl', 'wb') as f:
        pickle.dump(val_data_dict, f)
    print("Validation data was saved")

    with open(path_to_save + 'test.pkl', 'wb') as f:
        pickle.dump(test_data_dict, f)
    print("Test data was saved")


if __name__ == "__main__":
    TRAIN_DATA_PATH = "../../data/raw/train.csv"
    PROCESSED_DATA_FOLDER = "../../data/processed/"
    INTERIM_DATA_FOLDER = "../../data/processed/"

    timesteps = 150000
    step = 30000
    generate_data(TRAIN_DATA_PATH, PROCESSED_DATA_FOLDER, timesteps, step)

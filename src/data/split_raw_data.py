import pandas as pd
import numpy as np
from src.data.data_manager import DataManager


def split_data(raw_data_path, path_to_save, split_parts=100):
    """
    Args:
        raw_data_path: path to the original csv data
        path_to_save: path to .hdf5 data
        split_parts: number parts to split
    """

    saver = DataManager(path_to_save)

    TIMESTEPS_IN_ALL = 629145480
    counter = 0
    for chunk in pd.read_csv(raw_data_path, chunksize=TIMESTEPS_IN_ALL // split_parts):
        data: np.ndarray = chunk.values
        signal = (data[:, 0]).astype('int16')
        time_left = (data[:, 1]).astype('float32')

        name_group = "0"*(4-len(str(counter))) + str(counter)
        saver.push(name_group + "/signal", signal, 'int16')
        saver.push(name_group + "/time_left", time_left, 'float32')

        counter += 1
        print("{} / {} blocks were processed".format(counter, split_parts))
        if counter == 100:
            break


if __name__ == "__main__":
    ORIGINAL_DATA_PATH = "../../data/raw/train.csv"
    SPLITTED_DATA_PATH = "../../data/interim/train.hdf5"

    split_data(ORIGINAL_DATA_PATH, SPLITTED_DATA_PATH)

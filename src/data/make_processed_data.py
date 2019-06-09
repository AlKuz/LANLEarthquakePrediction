from src.data.data_manager import DataManager

import hashlib
import numpy as np
from sklearn.model_selection import train_test_split


def make_data(loader: DataManager, saver: DataManager, part_name, timesteps, step):
    signal = loader.get(part_name + '/signal')
    time_left = loader.get(part_name + '/time_left')

    start = 0
    while True:
        stop = start + timesteps
        try:
            signal_part: np.ndarray = signal[start:stop]
            time_value = time_left[stop]
        except IndexError:
            print("Process of block {} was finished".format(part_name))
            break
        signal_mean = np.mean(signal_part)
        fourier_part = np.abs(np.fft.rfft(signal_part - signal_mean))

        name = hashlib.md5(signal_part.tostring()).hexdigest()
        saver.push(name + '/signal', signal_part, 'int16')
        saver.push(name + '/fourier', fourier_part, 'float32')
        saver.push(name + '/time', time_value, 'float32')

        start += step


if __name__ == "__main__":
    ORIGINAL_DATA_PATH = "../../data/interim/train.hdf5"
    TRAIN_DATA_PATH = "../../data/processed/train.hdf5"
    VALID_DATA_PATH = "../../data/processed/valid.hdf5"

    TIMESTEPS = 150000
    STEP = 30000

    load_manager = DataManager(ORIGINAL_DATA_PATH)
    train_manager = DataManager(TRAIN_DATA_PATH)
    valid_manager = DataManager(VALID_DATA_PATH)

    train_parts, valid_parts = train_test_split(load_manager.names_from(), test_size=0.1, random_state=13)

    for n, name in enumerate(train_parts):
        make_data(load_manager, train_manager, name, TIMESTEPS, STEP)
        print("{} / {} for train data set was processed".format(n, len(train_parts)))

    for n, name in enumerate(valid_parts):
        make_data(load_manager, valid_manager, name, TIMESTEPS, STEP)
        print("{} / {} for valid data set was processed".format(n, len(valid_parts)))

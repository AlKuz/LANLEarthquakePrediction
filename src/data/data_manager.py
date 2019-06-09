import h5py
import os
import numpy as np


class DataManager(object):

    def __init__(self, filepath):
        self._filepath = filepath
        if not os.path.exists(filepath):
            open(filepath, 'w').close()

    def push(self, name: str, data: np.ndarray, dtype='float32'):
        data_struct = name.split('/')
        with h5py.File(self._filepath, 'a') as group:
            for group_name in data_struct[:-1]:
                try:
                    group = group.create_group(group_name)
                except ValueError:
                    group = group[group_name]
            try:
                group.create_dataset(data_struct[-1], data=data, dtype=dtype)
            except RuntimeError:
                print("Unable to create link ({} already exists)".format(name))

    def get(self, name: str):
        with h5py.File(self._filepath, 'r') as file:
            return file[name][:]

    def names_from(self, group=None):
        with h5py.File(self._filepath, 'r') as file:
            try:
                names = list(file[group].keys())[:]
            except TypeError:
                names = list(file.keys())[:]
            return names


if __name__ == "__main__":
    PATH = "/home/alexander/Projects/LANLEarthquakePrediction/data/test_data_format.hdf5"

    arr1 = np.random.uniform(-100, 100, (10,))
    data_name1 = 'folder/arr1'
    arr2 = np.random.uniform(-100, 100, (10,))
    data_name2 = 'folder/arr2'

    manager = DataManager(PATH)
    manager.push(data_name1, arr1)
    manager.push(data_name2, arr2)

    print(manager.names_from('folder'))

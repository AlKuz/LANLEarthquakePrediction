"""
Data generator module
"""
import pickle
import numpy as np


class DataGenerator(object):
    """Class for data controlling and transformation"""
    # TODO (kuznetsovav) Изменить работу на .hdf5 файлы данных

    def __init__(self, data_path):
        self._data_path = data_path
        self._data = self._load()
        self._key_list = list(self._data.keys())

    def _load(self):
        print("Loading data from {}".format(self._data_path))
        with open(self._data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def run(self, batch_size: int, timesteps: int):
        while True:
            yield self.get_batch(batch_size, timesteps)

    def get_batch(self, batch_size: int, timesteps: int = -1) -> (np.ndarray, np.ndarray):
        time_data = []
        signal_data = []
        data_names = np.random.choice(self._key_list, size=batch_size, replace=False)
        for name in data_names:
            time, signal = self._data[name]
            time_data.append(time)
            signal_data.append(signal)
        time_data = np.array(time_data)
        signal_data = np.array(signal_data)
        signal_data = np.expand_dims(signal_data, axis=-1)
        if timesteps != -1:
            signal_data = self._extract_features(signal_data, timesteps)
        return signal_data, time_data

    @classmethod
    def _extract_features(cls, data: np.ndarray, timesteps: int) -> np.ndarray:
        batch_size = data.shape[0]
        data = np.reshape(data, newshape=(batch_size, timesteps, -1))
        data_features = cls._calculate_statistics(data)
        fourier_features = cls._calculate_fourier_features(data)
        return np.concatenate([data_features, fourier_features], axis=-1)

    @classmethod
    def _calculate_statistics(cls, data: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.min(data, axis=-1, keepdims=True),
            np.max(data, axis=-1, keepdims=True),
            np.mean(data, axis=-1, keepdims=True),
            np.var(data, axis=-1, keepdims=True),
            np.std(data, axis=-1, keepdims=True),
            np.transpose(np.quantile(data, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], axis=-1), (1, 2, 0))
        ], axis=-1)

    @classmethod
    def _calculate_fourier_features(cls, data: np.ndarray) -> np.ndarray:
        mean_data_val = data.mean(axis=-1, keepdims=True)
        fourier = np.abs(np.fft.rfft(data - mean_data_val))
        fourier = fourier[:, :, :-1]
        batch_size, timesteps, _ = fourier.shape
        fourier = np.reshape(fourier, newshape=(batch_size, timesteps, 10, -1))
        fourier = np.max(fourier, axis=-1)
        return fourier


if __name__ == "__main__":
    VALID_DATA = "/home/alexander/Projects/LANLEarthquakePrediction/data/processed/validation.pkl"
    generator = DataGenerator(VALID_DATA)
    time, signal = generator.get_batch(128, 50)
    print(signal.shape)

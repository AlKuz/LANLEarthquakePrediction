"""
Abstract model class
"""
from abc import ABC, abstractmethod
import numpy as np
import json


class Model(ABC):
    """
    Abstract class for all models
    """

    def __init__(self, params: dict, folder, name, seed=13):
        self._params = params
        self._model = None
        self._create_model()
        print("Model was created")
        self._name = name
        self._model_folder = folder

        np.random.seed(seed)
        with open(self._model_folder + self._name + '.info', 'w') as file:
            self._model.summary(print_fn=lambda x: file.write(x + '\n'))
            file.write('\n\n\n')
            file.write('epoch,\ttrain_loss,\ttest_loss\n')

        with open(self._model_folder + self._name + '.json', 'w') as file:
            json.dump(params, file)

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        raise Exception("Realize method")

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        raise Exception("Realize method")

    @abstractmethod
    def train(self, train_data: dict, valid_data: dict):
        pass

    @abstractmethod
    def _create_model(self) -> None:
        raise Exception("Realize method")

    def _data_generator(self, data: dict, batch_size: int, num_parts: int) -> np.ndarray:
        key_list = list(data.keys())
        time_data = np.zeros(shape=(batch_size, 1))
        timeseries_data = np.zeros(shape=(batch_size, 150000))
        while True:
            data_names = np.random.choice(key_list, size=batch_size, replace=False)
            for n, name in enumerate(data_names):
                time, timeseries = data[name]
                time_data[n, 0] = time
                timeseries_data[n, :] = timeseries
            yield self.extract_features(timeseries_data, num_parts), time_data

    @classmethod
    def extract_features(cls, data: np.ndarray, num_parts) -> np.ndarray:
        batches = data.shape[0]
        data = np.reshape(data, newshape=(batches, num_parts, -1))
        fourier = np.abs(np.fft.rfft(data))
        data_features = cls.calc_statistics(data)
        fourier_features = cls.calc_statistics(fourier)
        return np.concatenate([data_features, fourier_features], axis=-1)

    @classmethod
    def calc_statistics(cls, data: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.min(data, axis=-1, keepdims=True),
            np.max(data, axis=-1, keepdims=True),
            np.mean(data, axis=-1, keepdims=True),
            np.var(data, axis=-1, keepdims=True),
            np.std(data, axis=-1, keepdims=True),
            np.transpose(np.quantile(data, [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98], axis=-1), (1, 2, 0)),
        ], axis=-1)

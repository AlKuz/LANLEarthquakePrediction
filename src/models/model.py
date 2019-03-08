"""
Abstract model class
"""
from abc import ABC, abstractmethod
import numpy as np
import json
from librosa.feature import mfcc


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
        while True:
            time_data = []
            timeseries_data = []
            data_names = np.random.choice(key_list, size=batch_size, replace=False)
            for n, name in enumerate(data_names):
                time, timeseries = data[name]
                time_data.append(time)
                timeseries_data.append(timeseries)
            time_data = np.array(time_data)
            timeseries_data = np.array(timeseries_data)
            timeseries_data = self.extract_features(timeseries_data, num_parts)
            yield timeseries_data, time_data

    @classmethod
    def extract_features(cls, data: np.ndarray, num_parts=None) -> np.ndarray:
        data_features = []
        for i in range(data.shape[0]):
            data_features.append(mfcc(data[i].astype('float32')))
        data_features = np.array(data_features)
        data_features = np.transpose(data_features, axes=(0, 2, 1))
        #batches, timesteps = data.shape
        #data = np.reshape(data, newshape=(batches, num_parts, -1))
        #data_features = cls.calc_statistics(data)
        #fourier = np.abs(np.fft.rfft(data))
        #fourier_features = cls.calc_statistics(fourier)
        return data_features #np.concatenate([data_features, fourier_features], axis=-1)

    @classmethod
    def calc_statistics(cls, data: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.min(data, axis=-1, keepdims=True),
            np.max(data, axis=-1, keepdims=True),
            np.mean(data, axis=-1, keepdims=True),
            np.var(data, axis=-1, keepdims=True),
            np.std(data, axis=-1, keepdims=True),
            np.transpose(np.quantile(data, [0.05, 0.25, 0.5, 0.75, 0.95], axis=-1), (1, 2, 0)),
        ], axis=-1)

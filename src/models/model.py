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

    @classmethod
    def _data_generator(cls, data: dict, batch_size: int, feature_func=None) -> np.ndarray:
        key_list = list(data.keys())
        while True:
            time_list = []
            timeseries_list = []
            data_names = np.random.choice(key_list, size=batch_size, replace=False)
            for name in data_names:
                time, timeseries = data[name]
                time_list.append(time)
                timeseries_list.append(timeseries)
            time_list = np.array(time_list)
            if feature_func is None:
                timeseries_list = np.array(timeseries_list)
            else:
                timeseries_list = feature_func(np.array(timeseries_list))
            yield timeseries_list, time_list

    @classmethod
    def extract_features(cls, data: np.ndarray, num_parts=10, as_filters=False, add_fourier=False) -> np.ndarray:
        statistic_list = cls._process_data(data, num_parts, as_filters)
        if add_fourier:
            fourier = np.abs(np.fft.rfft(data))
            statistic_list = statistic_list + cls._process_data(fourier, num_parts, as_filters)
        statistic_list = np.concatenate(statistic_list, axis=1)
        return statistic_list

    @classmethod
    def calc_statistics(cls, data: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.min(data, axis=1, keepdims=True),
            np.max(data, axis=1, keepdims=True),
            np.mean(data, axis=1, keepdims=True),
            np.var(data, axis=1, keepdims=True),
            np.std(data, axis=1, keepdims=True),
            np.transpose(np.quantile(data, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], axis=1)),
        ], axis=1)

    @classmethod
    def _process_data(cls, data: np.ndarray, num_parts: int, as_filters: bool) -> list:
        statistics_list = []
        data_step = data.shape[1] // num_parts
        for i in range(num_parts):
            data_part = data[:, i * data_step:(i + 1) * data_step]
            features = cls.calc_statistics(data_part)
            if as_filters:
                features = np.expand_dims(features, axis=1)
            statistics_list.append(features)
        return statistics_list

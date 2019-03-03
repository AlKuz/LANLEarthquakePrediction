"""
Abstract model class
"""
from abc import ABC, abstractmethod
import numpy as np
import json
from scipy import interpolate


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
    def _extract_features(cls, data: np.ndarray, num_parts=10, as_filters=False) -> np.ndarray:
        calc_statistics = lambda d: np.concatenate([
            np.min(d, axis=1, keepdims=True),
            np.max(d, axis=1, keepdims=True),
            np.mean(d, axis=1, keepdims=True),
            np.var(d, axis=1, keepdims=True),
            np.std(d, axis=1, keepdims=True),
            np.transpose(np.quantile(d, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], axis=1)),
        ], axis=1)

        def calc_fourie(data, n=10):
            amp = np.abs(np.fft.rfft(data))
            freq = np.fft.rfftfreq(data.shape[1], 1 / 1000)
            new_freq = np.linspace(0, max(freq), n)
            new_amp = np.zeros(shape=(amp.shape[0], n))
            for i in range(amp.shape[0]):
                func = interpolate.interp1d(freq, amp[i, :])
                new_amp[i, :] = func(new_freq)
            return new_amp

        statistic_params_list = []

        step = data.shape[1] // num_parts
        for i in range(num_parts):
            data_part = data[:, i * step:(i + 1) * step]
            features = calc_statistics(data_part)
            if as_filters:
                #features_fourie = calc_fourie(data_part)
                features = np.expand_dims(features, axis=1)
                #features_fourie = np.expand_dims(features_fourie, axis=1)
                #features = np.concatenate([features_data, features_fourie], axis=-1)
            statistic_params_list.append(features)

        #if not as_filters:
        #    fourie_filters = calc_fourie(data, n=100)
        #    statistic_params_list.append(fourie_filters)

        return np.concatenate(statistic_params_list, axis=1)

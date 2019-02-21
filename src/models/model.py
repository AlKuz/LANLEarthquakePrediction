"""
Abstract model class
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from numpy import ndarray, random, array, mean, abs


class Model(ABC):
    """
    Abstract class for all models
    """

    def __init__(self, params: dict, seed=13):
        self._params = params
        self._model = None
        self._create_model(params)

        random.seed(seed)

    @abstractmethod
    def predict(self, input_data: ndarray) -> ndarray:
        raise Exception("Realize method")

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        raise Exception("Realize method")

    def train(self, train_data: dict, valid_data: dict, model_path: str, batch_size=1, epochs=1000, early_stop=10):

        train_keys_list = list(train_data.keys())
        valid_keys_list = list(valid_data.keys())

        best_mae = None
        best_epoch = 1

        for e in range(1, epochs+1):

            train_data_list = random.choice(train_keys_list, size=batch_size, replace=False)
            valid_data_list = random.choice(valid_keys_list, size=batch_size, replace=False)

            train_batch_x, train_batch_y = self._create_batch(train_data_list, train_data)
            valid_batch_x, valid_batch_y = self._create_batch(valid_data_list, valid_data)

            train_mae = self._fit_batch(train_batch_x, train_batch_y)

            model_result = self.predict(valid_batch_x)
            valid_mae = self._calculate_mae(model_result, valid_batch_y)

            try:
                if valid_mae < best_mae:
                    self.save_model(model_path)
                    best_mae = valid_mae
                    best_epoch = e
            except:
                best_mae = valid_mae

            print("Epoch {} / {}: train_mae = {:.4f}, valid_mae = {:.4f}".format(e, epochs, train_mae, valid_mae))

            if e - best_epoch >= early_stop:
                break

    @abstractmethod
    def _create_model(self, params) -> None:
        raise Exception("Realize method")

    @abstractmethod
    def _fit_batch(self, train_x: ndarray, train_y: ndarray) -> float:
        raise Exception("Realize method")

    @classmethod
    def _create_batch(cls, data_list: list, data_dict: dict) -> (ndarray, ndarray):
        time_list = []
        timeseries_list = []
        for name in data_list:
            time, timeseries = data_dict[name]
            time_list.append(time)
            timeseries_list.append(timeseries)
        return array(timeseries_list), array(time_list)

    @classmethod
    def _calculate_mae(cls, output: ndarray, target: ndarray) -> float:
        return float(mean(abs(target - output)))


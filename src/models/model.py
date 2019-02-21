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
        self._best_model = None

        self._train_mae_log = []
        self._valid_mae_log = []

        random.seed(seed)

    @abstractmethod
    def predict(self, input_data: ndarray) -> ndarray:
        raise Exception("Realize method")

    @abstractmethod
    def save_model(self, models_folder: str, model_name: str) -> None:
        raise Exception("Realize method")

    def train(self, train_data: dict, valid_data: dict, batch_size=1, epochs=1000) -> None:

        train_keys_list = list(train_data.keys())
        valid_keys_list = list(valid_data.keys())

        best_mae = None

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
                    self._remember_best_model()
                    best_mae = valid_mae
            except:
                best_mae = valid_mae

            self._train_mae_log.append(train_mae)
            self._valid_mae_log.append(valid_mae)

            print("Epoch {} / {}: train_mae = {:.4f}, valid_mae = {:.4f}".format(e, epochs, train_mae, valid_mae))

    @abstractmethod
    def _create_model(self, params) -> None:
        raise Exception("Realize method")

    @abstractmethod
    def _fit_batch(self, train_x: ndarray, train_y: ndarray) -> float:
        raise Exception("Realize method")

    def _remember_best_model(self):
        self._best_model = deepcopy(self._model)

    def _restore_best_model(self):
        self._model = deepcopy(self._best_model)
        self._best_model = None

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


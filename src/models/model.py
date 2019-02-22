"""
Abstract model class
"""
from abc import ABC, abstractmethod
from numpy import ndarray, random, array, abs, concatenate, min, max, mean, var, std, quantile


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

    def train(self, train_data: dict, valid_data: dict, model_path: str, batch_size=1, epochs=1000,
              train_repetitions=100, valid_repetitions=10, early_stop=10):

        train_keys_list = list(train_data.keys())
        valid_keys_list = list(valid_data.keys())

        best_mae = None
        best_epoch = 1

        for e in range(1, epochs+1):

            train_mae = None
            for _ in range(train_repetitions):
                train_data_list = random.choice(train_keys_list, size=batch_size, replace=False)
                train_batch_x, train_batch_y = self._create_batch(train_data_list, train_data)
                train_mae = self._fit_batch(train_batch_x, train_batch_y)

            sum_valid_mae = 0
            for _ in range(valid_repetitions):
                valid_data_list = random.choice(valid_keys_list, size=batch_size, replace=False)
                valid_batch_x, valid_batch_y = self._create_batch(valid_data_list, valid_data)
                model_result = self.predict(valid_batch_x)
                valid_mae = self._calculate_mae(model_result, valid_batch_y)
                sum_valid_mae += valid_mae
            valid_mae = sum_valid_mae / valid_repetitions

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

    @classmethod
    def _extract_features(cls, data: ndarray, num_parts=10) -> ndarray:
        calc_statistics = lambda d: concatenate([
            min(d, axis=1, keepdims=True),
            max(d, axis=1, keepdims=True),
            mean(d, axis=1, keepdims=True),
            var(d, axis=1, keepdims=True),
            std(d, axis=1, keepdims=True),
            quantile(d, 0.25, axis=1, keepdims=True),
            quantile(d, 0.5, axis=1, keepdims=True),
            quantile(d, 0.75, axis=1, keepdims=True)
        ], axis=1)

        statistic_params_list = []
        statistic_params_list.append(calc_statistics(data))

        step = data.shape[1] // num_parts
        for i in range(num_parts):
            statistic_params_list.append(
                calc_statistics(data[:, i*step:(i+1)*step])
            )
        return concatenate(statistic_params_list, axis=1)


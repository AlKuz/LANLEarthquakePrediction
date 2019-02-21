"""
Convolution model
"""
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model as KModel
from keras.optimizers import Adam, SGD
from numpy import ndarray, expand_dims

from src.models.model import Model


class ConvolutionModel(Model):

    def predict(self, input_data: ndarray) -> ndarray:
        if len(input_data.shape) == 2:
            input_data = expand_dims(input_data, axis=-1)
        return self._model.predict(input_data)

    def save_model(self, model_path: str):
        self._model.save(model_path)

    def _create_model(self, params: dict) -> None:
        for p in ['input_size', 'filters', 'kernels', 'optimizer', 'optimizer_params']:
            assert p in params, "There is no {} in the dictionary parameters".format(p)

        input_size = params['input_size']
        assert isinstance(input_size, int)
        filters = params['filters']
        assert isinstance(filters, tuple)
        kernels = params['kernels']
        assert isinstance(kernels, tuple)
        assert len(filters) == len(kernels)
        optimizer = params['optimizer']
        optimizer_params = params['optimizer_params']
        if optimizer == 'Adam':
            optimizer = Adam
        elif optimizer == 'SGD':
            optimizer = SGD
        else:
            raise Exception("Unknown optimizer")

        input_tensor = Input(shape=(input_size, 1))
        model = input_tensor
        for filter, kernel in zip(filters, kernels):
            model = Conv1D(filter, kernel, padding='same', activation='relu')(model)
            model = MaxPooling1D()(model)

        model = Flatten()(model)
        model = Dense(1, activation='relu')(model)
        self._model = KModel(input_tensor, model)

        self._model.compile(optimizer=optimizer(**optimizer_params), loss='mae')

    def _fit_batch(self, train_x: ndarray, train_y: ndarray) -> float:
        if len(train_x.shape) == 2:
            train_x = expand_dims(train_x, axis=-1)
        return self._model.train_on_batch(train_x, train_y)

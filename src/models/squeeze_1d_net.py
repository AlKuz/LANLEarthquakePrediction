"""
Convolution model
"""
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate
from keras.models import Model as KModel
from keras.optimizers import Adam, SGD
from numpy import ndarray, expand_dims

from src.models.model import Model


class Squeeze1DNetModel(Model):

    def predict(self, input_data: ndarray) -> ndarray:
        features = self._extract_features(input_data, num_parts=self._timesteps, as_filters=True)
        return self._model.predict(features)

    def save_model(self, model_path: str):
        self._model.save(model_path)

    def _create_model(self, params: dict) -> None:

        self._timesteps = params['timesteps']
        filters = params['filters']
        kernels = params['kernels']
        optimizer = params['optimizer']
        optimizer_params = params['optimizer_params']

        if optimizer == 'Adam':
            optimizer = Adam
        elif optimizer == 'SGD':
            optimizer = SGD
        else:
            raise Exception("Unknown optimizer")

        input_tensor = Input(shape=(self._timesteps, 8))
        model = input_tensor
        for filter, kernel in zip(filters, kernels):
            if isinstance(filter, int):
                model = self._squeeze_1d_layer(model, filter, kernel)
            else:
                for filter_elem in filter:
                    model = self._squeeze_1d_layer(model, filter_elem, kernel)
            model = MaxPooling1D()(model)

        model = Flatten()(model)
        model = Dense(1, activation='relu')(model)
        self._model = KModel(input_tensor, model)

        self._model.compile(optimizer=optimizer(**optimizer_params), loss='mae')

    def _squeeze_1d_layer(self, input_tensor, filter_size, kernel_size):
        layer = Conv1D(filter_size // 4, 1, padding='same', activation='relu')(input_tensor)
        layer = Concatenate()([
            Conv1D(filter_size, 1, padding='same', activation='relu')(layer),
            Conv1D(filter_size, kernel_size, padding='same', activation='relu')(layer)
        ])
        return layer

    def _fit_batch(self, train_x: ndarray, train_y: ndarray) -> float:
        features = self._extract_features(train_x, num_parts=self._timesteps, as_filters=True)
        return self._model.train_on_batch(features, train_y)

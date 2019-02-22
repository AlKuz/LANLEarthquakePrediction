"""
Dense model
"""
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model as KModel
from keras.optimizers import Adam, SGD
from numpy import ndarray

from src.models.model import Model


class DenseModel(Model):

    def predict(self, input_data: ndarray) -> ndarray:
        features = self._extract_features(input_data, num_parts=self._num_parts)
        return self._model.predict(features)

    def save_model(self, model_path: str):
        self._model.save(model_path)

    def _create_model(self, params: dict) -> None:

        self._num_parts = params['num_feature_parts']
        layers = params['layers']
        optimizer = params['optimizer']
        optimizer_params = params['optimizer_params']

        if optimizer == 'Adam':
            optimizer = Adam
        elif optimizer == 'SGD':
            optimizer = SGD
        else:
            raise Exception("Unknown optimizer")

        input_tensor = Input(shape=((self._num_parts+1)*8,))
        model = input_tensor
        for depth in layers:
            model = Dense(depth, activation='tanh', kernel_initializer='glorot_normal')(model)
        model = Dense(1, activation='relu')(model)
        self._model = KModel(input_tensor, model)

        self._model.compile(optimizer=optimizer(**optimizer_params), loss='mae')

    def _fit_batch(self, train_x: ndarray, train_y: ndarray) -> float:
        features = self._extract_features(train_x, num_parts=self._num_parts)
        return self._model.train_on_batch(features, train_y)


"""
Convolution model
"""
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model as KModel
from numpy import ndarray

from src.models.model import Model


class ConvolutionModel(Model):

    def predict(self, input_data: ndarray) -> ndarray:
        return self._model.predict(input_data)

    def save_model(self, models_folder: str, model_name: str):
        self._model.save(models_folder + model_name)

    def _create_model(self, params: dict) -> None:
        for p in ['input_size', 'filters', 'kernels']:
            assert p in params, "There is no {} in the dictionary parameters".format(p)

        input_size = params['input_size']
        filters = params['filters']
        kernels = params['kernels']

        input_tensor = Input(shape=(input_size,))
        model = input_tensor
        for filter, kernel in zip(filters, kernels):
            model = Conv1D(filter, kernel, padding='same', activation='relu')(model)
            model = MaxPooling1D()(model)

        model = Flatten()(model)
        model = Dense(1, activation='relu')(model)
        self._model = KModel(input_tensor, model)

    def _fit_batch(self, train_x: ndarray, train_y: ndarray) -> float:
        return self._model.train_on_batch(train_x, train_y)

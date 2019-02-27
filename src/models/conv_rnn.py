"""
Convolution model with rnn output
"""
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, GRU, LSTM, Add, Concatenate
from keras.models import Model as KModel
from keras.optimizers import Adam, SGD
from numpy import ndarray, expand_dims

from src.models.model import Model


class ConvolutionRNNModel(Model):

    def __init__(self, params: dict, folder, name, seed: int = 13):
        super().__init__(params, folder, name, seed)

    def predict(self, input_data: ndarray) -> ndarray:
        if len(input_data.shape) == 2:
            input_data = expand_dims(input_data, axis=-1)
        return self._model.predict(input_data)

    def save_model(self, model_path: str):
        self._model.save(model_path)

    def _create_model(self, params: dict) -> None:
        rnn_types = {
            'LSTM': LSTM,
            'GRU': GRU
        }

        input_size = params['input_size']
        filters = params['filters']
        kernels = params['kernels']
        rnn_layer_types = params['rnn_types']
        rnn_layers = params['rnn_layers']
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

        rnn_l = rnn_types[rnn_layer_types[0]](rnn_layers[0], return_sequences=True)(model)
        rnn_r = rnn_types[rnn_layer_types[0]](rnn_layers[0], return_sequences=True, go_backwards=True)(model)
        model = Add()([rnn_l, rnn_r])

        rnn_l = rnn_types[rnn_layer_types[1]](rnn_layers[1], return_sequences=True)(model)
        rnn_r = rnn_types[rnn_layer_types[1]](rnn_layers[1], return_sequences=True, go_backwards=True)(model)
        model = Concatenate()([rnn_l, rnn_r])

        model = Flatten()(model)
        model = Dense(1, activation='relu')(model)

        self._model = KModel(input_tensor, model)

        self._model.compile(optimizer=optimizer(**optimizer_params), loss='mae')

    def _fit_batch(self, train_x: ndarray, train_y: ndarray) -> float:
        if len(train_x.shape) == 2:
            train_x = expand_dims(train_x, axis=-1)
        return self._model.train_on_batch(train_x, train_y)

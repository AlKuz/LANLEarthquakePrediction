"""
Convolution model with rnn output
"""
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, GRU, LSTM, Add, Concatenate
from keras.models import Model as KModel
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.regularizers import l1, l2, l1_l2
from numpy import ndarray, expand_dims

from src.models.model import Model
from src.data import DataGenerator


class ConvolutionRNNModel(Model):

    def __init__(self, params: dict, folder, name, seed: int = 13):
        super().__init__(params, folder, name, seed)

    def predict(self, input_data: ndarray) -> ndarray:
        #features = self.extract_features(input_data, num_parts=self._timesteps, as_filters=True)
        return self._model.predict(input_data)

    def save_model(self, model_path: str):
        self._model.save(model_path)

    def _create_model(self) -> None:

        rnn_types = {
            'LSTM': LSTM,
            'GRU': GRU
        }

        regularizers_dict = {
            'l1': l1,
            'l2': l2,
        }

        optimizer_dict = {
            'Adam': Adam,
            'SGD': SGD
        }

        timesteps = self._params['timesteps']
        filters = self._params['filters']
        kernels = self._params['kernels']
        conv_regularizers_params = self._params['conv_regularizers_params']
        conv_regularizers = regularizers_dict[self._params['conv_regularizers']]
        rnn_layers = self._params['rnn_layers']
        rnn_layer_types = list(map(lambda x: rnn_types[x], self._params['rnn_types']))
        optimizer = optimizer_dict[self._params['optimizer']]
        optimizer_params = self._params['optimizer_params']

        if timesteps == -1:
            input_tensor = Input(shape=(150000, 1))
        else:
            input_tensor = Input(shape=(timesteps, 22))
        model = input_tensor

        for filter, kernel in zip(filters, kernels):
            if conv_regularizers_params != 0:
                model = Conv1D(filter, kernel, strides=kernel, padding='valid', activation='relu',
                               kernel_regularizer=conv_regularizers(conv_regularizers_params))(model)
            else:
                model = Conv1D(filter, kernel, strides=kernel, padding='valid', activation='relu')(model)

        for layer, layer_type in zip(rnn_layers, rnn_layer_types):
            rnn_l = layer_type(layer, return_sequences=True)(model)
            rnn_r = layer_type(layer, return_sequences=True, go_backwards=True)(model)
            model = Concatenate(axis=-1)([rnn_l, rnn_r])

        model = Flatten()(model)
        model = Dense(1, activation='relu')(model)

        self._model = KModel(input_tensor, model)

        self._model.compile(optimizer=optimizer(**optimizer_params), loss='mae')

    def train(self, train: DataGenerator, valid: DataGenerator):

        timesteps = self._params['timesteps']
        batch_size = self._params['batch_size']
        epochs = self._params['epochs']
        train_repetitions = self._params['train_repetitions']
        valid_repetitions = self._params['valid_repetitions']
        early_stop = self._params['early_stop']
        reduce_factor = self._params['reduce_factor']
        epochs_to_reduce = self._params['epochs_to_reduce']

        callback_list = [
            ModelCheckpoint(
                filepath=self._model_folder + self._name + '.hdf5',
                monitor='val_loss',
                save_best_only=True
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=early_stop
            ),
            CSVLogger(
                filename=self._model_folder + self._name + '.info',
                append=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_factor,
                patience=epochs_to_reduce
            )
        ]

        self._model.fit_generator(
            train.run(batch_size, timesteps),
            steps_per_epoch=train_repetitions,
            epochs=epochs,
            callbacks=callback_list,
            validation_data=valid.run(batch_size, timesteps),
            validation_steps=valid_repetitions
        )

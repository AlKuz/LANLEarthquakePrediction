"""
RNN model
"""
from keras.layers import Input, Dense, Flatten, Concatenate, LSTM, GRU, Dropout
from keras.models import Model as KModel
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from numpy import ndarray

from src.models.model import Model


class RNNModel(Model):

    def predict(self, input_data: ndarray) -> ndarray:
        features = self.extract_features(input_data, num_parts=self._timesteps)
        return self._model.predict(features)

    def save_model(self, model_path: str):
        self._model.save(model_path)

    def _create_model(self) -> None:

        rnn_types = {
            'LSTM': LSTM,
            'GRU': GRU
        }

        optimizer_dict = {
            'Adam': Adam,
            'SGD': SGD
        }

        self._timesteps = self._params['timesteps']
        dropout = self._params['dropout']
        add_fourier = self._params['add_fourier']
        layers = self._params['layers']
        layer_types = list(map(lambda x: rnn_types[x], self._params['layer_types']))
        optimizer = optimizer_dict[self._params['optimizer']]
        optimizer_params = self._params['optimizer_params']

        assert len(layers) == len(layer_types)

        if add_fourier:
            self._timesteps *= 2

        input_tensor = Input(shape=(self._timesteps, 30))
        model = Dense(layers[0], activation='tanh')(input_tensor)
        for layer, layer_type in zip(layers, layer_types):
            rnn_l = layer_type(layer, return_sequences=True)(model)
            rnn_r = layer_type(layer, return_sequences=True, go_backwards=True)(model)
            model = Concatenate()([rnn_l, rnn_r])
        model = Flatten()(model)
        model = Dense(1, activation='relu')(model)
        if dropout != 0:
            model = Dropout(dropout)(model)

        self._model = KModel(input_tensor, model)

        self._model.compile(optimizer=optimizer(**optimizer_params), loss='mae')

    def train(self, train_data: dict, valid_data: dict):

        add_fourier = self._params['add_fourier']
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

        train_generator = self._data_generator(train_data, batch_size, self._timesteps)
        valid_generator = self._data_generator(valid_data, batch_size, self._timesteps)

        self._model.fit_generator(
            train_generator,
            steps_per_epoch=train_repetitions,
            epochs=epochs,
            callbacks=callback_list,
            validation_data=valid_generator,
            validation_steps=valid_repetitions
        )

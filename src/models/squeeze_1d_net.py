"""
Convolution model
"""
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Model as KModel
from keras.optimizers import Adam, SGD
from numpy import ndarray, expand_dims

from src.models.model import Model


class Squeeze1DNetModel(Model):

    def predict(self, input_data: ndarray) -> ndarray:
        features = self.extract_features(input_data, num_parts=self._timesteps, as_filters=True)
        return self._model.predict(features)

    def save_model(self, model_path: str):
        self._model.save(model_path)

    def _create_model(self) -> None:

        optimizer_dict = {
            'Adam': Adam,
            'SGD': SGD
        }

        self._timesteps = self._params['timesteps']
        filters = self._params['filters']
        kernels = self._params['kernels']
        optimizer = optimizer_dict[self._params['optimizer']]
        optimizer_params = self._params['optimizer_params']

        input_tensor = Input(shape=(self._timesteps, 15))
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

    def train(self, train_data: dict, valid_data: dict):

        batch_size = self._params['batch_size']
        epochs = self._params['epochs']
        train_repetitions = self._params['train_repetitions']
        valid_repetitions = self._params['valid_repetitions']
        early_stop = self._params['early_stop']

        feature_extractor = lambda x: self.extract_features(x, self._params['timesteps'], as_filters=True)

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
            )
        ]

        self._model.fit_generator(
            self._data_generator(train_data, batch_size, feature_extractor),
            steps_per_epoch=train_repetitions,
            epochs=epochs,
            callbacks=callback_list,
            validation_data=self._data_generator(valid_data, batch_size, feature_extractor),
            validation_steps=valid_repetitions
        )

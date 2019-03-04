"""
Convolution model
"""
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model as KModel
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from numpy import ndarray, expand_dims

from src.models.model import Model


class ConvolutionModel(Model):

    def predict(self, input_data: ndarray) -> ndarray:
        features = self.extract_features(input_data, num_parts=self._params['timesteps'], as_filters=True)
        return self._model.predict(features)

    def save_model(self, model_path: str):
        self._model.save(model_path)

    def _create_model(self) -> None:

        optimizer_dict = {
            'Adam': Adam,
            'SGD': SGD
        }

        timesteps = self._params['timesteps']
        self._features_as_filter = self._params['features_as_filter']
        filters = self._params['filters']
        kernels = self._params['kernels']
        optimizer = optimizer_dict[self._params['optimizer']]
        optimizer_params = self._params['optimizer_params']

        if self._features_as_filter:
            shape = (timesteps, 15)
        else:
            shape = (timesteps*15, 1)
        input_tensor = Input(shape=shape)
        model = input_tensor
        for filter, kernel in zip(filters, kernels):
            if isinstance(filter, int):
                model = Conv1D(filter, kernel, padding='same', activation='relu')(model)
            else:
                for fil in filter:
                    model = Conv1D(fil, kernel, padding='same', activation='relu')(model)
            model = MaxPooling1D()(model)

        model = Flatten()(model)
        model = Dense(1, activation='relu')(model)
        self._model = KModel(input_tensor, model)

        self._model.compile(optimizer=optimizer(**optimizer_params), loss='mae')

    def train(self, train_data: dict, valid_data: dict):

        batch_size = self._params['batch_size']
        epochs = self._params['epochs']
        train_repetitions = self._params['train_repetitions']
        valid_repetitions = self._params['valid_repetitions']
        early_stop = self._params['early_stop']
        reduce_factor = self._params['reduce_factor']
        epochs_to_reduce = self._params['epochs_to_reduce']

        feature_extractor = lambda x:\
            expand_dims(self.extract_features(x, self._params['timesteps'], as_filters=self._features_as_filter),
                        axis=-1)

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
            self._data_generator(train_data, batch_size, feature_extractor),
            steps_per_epoch=train_repetitions,
            epochs=epochs,
            callbacks=callback_list,
            validation_data=self._data_generator(valid_data, batch_size, feature_extractor),
            validation_steps=valid_repetitions
        )

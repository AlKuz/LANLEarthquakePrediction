import sys
sys.path.insert(0, '/home/alexander/Projects/LANLEarthquakePrediction')
from src.models import Model,\
    ConvolutionModel,\
    DenseModel, RNNModel,\
    ConvolutionRNNModel,\
    Squeeze1DNetModel
from src.data import DataGenerator
import pickle


class ModelConfigurator(object):
    """Configure models via class methods"""

    @classmethod
    def convolution_model(cls, folder) -> (Model, str):
        params = {
            'timesteps': 150,
            'filters': (16, 32, 64, 128, 256, 512),
            'kernels': (1, 1, 3, 1, 1, 3),
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.001},
            'batch_size': 128,
            'epochs': 1000,
            'train_repetitions': 100,
            'valid_repetitions': 10,
            'early_stop': 50,
            'reduce_factor': 0.8,
            'epochs_to_reduce': 5,
            'features_as_filter': False
        }
        name = 'convolution_model_v2'
        return ConvolutionModel(params, folder, name), name, params

    @classmethod
    def dense_model(cls, folder) -> (Model, str):
        params = {
            'num_feature_parts': 50,
            'layers': (64, 128),
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.001},
            'batch_size': 128,
            'epochs': 1000,
            'train_repetitions': 100,
            'valid_repetitions': 10,
            'early_stop': 50,
            'reduce_factor': 0.8,
            'epochs_to_reduce': 5,
            'features_as_filter': True
        }
        name = 'dense_model_v2'
        return DenseModel(params, folder, name), name, params

    @classmethod
    def rnn_model(cls, folder) -> (Model, str):
        params = {
            'timesteps': 50,
            'layers': (256, 512),
            'layer_types': ('LSTM', 'LSTM'),
            'dropout': 0,
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.002},
            'batch_size': 128,
            'epochs': 1000,
            'train_repetitions': 200,
            'valid_repetitions': 10,
            'early_stop': 16,
            'reduce_factor': 0.8,
            'epochs_to_reduce': 4
        }
        name = 'rnn_model_v4'
        return RNNModel(params, folder, name), name, params

    @classmethod
    def squeeze_1d_model(cls, folder) -> (Model, str):
        params = {
            'timesteps': 150,
            'filters': (32, 64, 128, 256),
            'kernels': (5, 5, 3, 3),
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.001},
            'batch_size': 128,
            'epochs': 1000,
            'train_repetitions': 100,
            'valid_repetitions': 10,
            'early_stop': 50
        }
        name = 'squeeze_1d_model'
        return Squeeze1DNetModel(params, folder, name), name, params

    @classmethod
    def conv_rnn_model(cls, folder):
        params = {
            'timesteps': -1,
            'filters': (128, 64, 32, 16),
            'kernels': (50, 10, 3, 3),
            'conv_regularizers_params': 0.02,
            'conv_regularizers': 'l2',
            'rnn_layers': (64, 128),
            'rnn_types': ('LSTM', 'LSTM'),
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.001},
            'batch_size': 64,
            'epochs': 1000,
            'train_repetitions': 100,
            'valid_repetitions': 10,
            'early_stop': 50,
            'reduce_factor': 0.8,
            'epochs_to_reduce': 10
        }
        name = 'conv_rnn_model_v4'
        return ConvolutionRNNModel(params, folder, name), name, params


if __name__ == "__main__":
    MODELS_FOLDER = "/home/alexander/Projects/Kaggle/LANLEarthquakePrediction/models/"
    TRAIN_DATA = "/home/alexander/Projects/Kaggle/LANLEarthquakePrediction/data/processed/train.pkl"
    VALID_DATA = "/home/alexander/Projects/Kaggle/LANLEarthquakePrediction/data/processed/validation.pkl"

    train_generator = DataGenerator(TRAIN_DATA)
    valid_generator = DataGenerator(VALID_DATA)

    model, name, params = ModelConfigurator.conv_rnn_model(MODELS_FOLDER)
    # TODO (kuznetsovav) Разнести параметры конфигурации и обучения по разным методам
    model.train(train_generator, valid_generator)

from src.models import Model,\
    LGBMRegressor,\
    ConvolutionModel,\
    DenseModel, RNNModel,\
    ConvolutionRNNModel,\
    Squeeze1DNetModel
import pickle


class ModelConfigurator(object):
    """Configure models via class methods"""

    @classmethod
    def lgbm_regressor(cls, folder) -> (Model, str):
        params = {
            'num_leaves': 54,
            #'min_data_in_leaf': 79,
            'objective': 'huber',
            'max_depth': -1,
            'learning_rate': 0.01,
            #"boosting": "gbdt",
            # "feature_fraction": 0.8354507676881442,
            #"bagging_freq": 3,
            #"bagging_fraction": 0.8126672064208567,
            "bagging_seed": 11,
            "metric": 'mae',
            "verbosity": -1,
            'reg_alpha': 1.1302650970728192,
            'reg_lambda': 0.3603427518866501
        }
        name = 'lgbm_regressor'
        return LGBMRegressor(params, folder, name), name, params

    @classmethod
    def convolution(cls, folder) -> (Model, str):
        params = {
            'timesteps': 50,
            'filters': (16, 32, 64),
            'kernels': (3, 3, 3),
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.001},
            'batch_size': 128,
            'epochs': 1000,
            'train_repetitions': 100,
            'valid_repetitions': 10,
            'early_stop': 50
        }
        name = 'convolution_model'
        return ConvolutionModel(params, folder, name), name, params

    @classmethod
    def dense_model(cls, folder) -> (Model, str):
        params = {
            'num_feature_parts': 20,
            'layers': (512, 256, 128),
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.001},
            'batch_size': 128,
            'epochs': 1000,
            'train_repetitions': 100,
            'valid_repetitions': 10,
            'early_stop': 50
        }
        name = 'dense_model'
        return DenseModel(params, folder, name), name, params

    @classmethod
    def rnn_model(cls, folder) -> (Model, str):
        params = {
            'timesteps': 100,
            'layers': (64, 128, 256),
            'layer_types': ('GRU', 'GRU', 'GRU'),
            'optimizer': 'SGD',
            'optimizer_params': {'lr': 0.01, 'decay': 1e-6, 'momentum': 0.9, 'nesterov': True},
            'batch_size': 128,
            'epochs': 1000,
            'train_repetitions': 100,
            'valid_repetitions': 10,
            'early_stop': 50
        }
        name = 'rnn_model_v2'
        return RNNModel(params, folder, name), name, params

    @classmethod
    def squeeze_1d_model(cls, folder) -> (Model, str):
        params = {
            'timesteps': 150,
            'filters': (32, 64, 128, 256),
            'kernels': (5, 5, 3, 3),
            'optimizer': 'SGD',
            'optimizer_params': {'lr': 0.01, 'decay': 1e-6, 'momentum': 0.9, 'nesterov': True}
        }
        name = 'squeeze_1d_model'
        return Squeeze1DNetModel(params, folder, name), name, params

    @classmethod
    def conv_rnn_model(cls, folder):
        params = {
            'timesteps': 150,
            'filters': (32, 64),
            'kernels': (3, 3),
            'rnn_types': ('GRU', 'GRU'),
            'rnn_layers': (64, 64),
            'optimizer': 'SGD',
            'optimizer_params': {'lr': 0.01, 'decay': 1e-6, 'momentum': 0.9, 'nesterov': True}
        }
        name = 'conv_rnn_model'
        return ConvolutionRNNModel(params, folder, name), name, params


if __name__ == "__main__":
    MODELS_FOLDER = "/home/alexander/Projects/LANLEarthquakePrediction/models/"
    TRAIN_DATA = "/home/alexander/Projects/LANLEarthquakePrediction/data/processed/train.pkl"
    VALID_DATA = "/home/alexander/Projects/LANLEarthquakePrediction/data/processed/validation.pkl"

    print("Loading data...")
    with open(TRAIN_DATA, 'rb') as f:
        train_data = pickle.load(f)

    with open(VALID_DATA, 'rb') as f:
        valid_data = pickle.load(f)
    print("Data was loaded")

    model, name, params = ModelConfigurator.rnn_model(MODELS_FOLDER)
    model.train(train_data, valid_data)

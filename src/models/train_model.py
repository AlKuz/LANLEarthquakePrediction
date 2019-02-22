from src.models import Model, LGBMRegressor, ConvolutionModel, DenseModel
import pickle


class ModelConfigurator(object):
    """Configure models via class methods"""

    @classmethod
    def lgbm_regressor(cls) -> (Model, str):
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
        return LGBMRegressor(params), 'lgbm_regressor.pkl'

    @classmethod
    def convolution(cls) -> (Model, str):
        params = {
            'input_size': 150000,
            'filters': (2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64),
            'kernels': (21, 13, 13, 7, 7, 7, 7, 5, 5, 5, 5, 5),
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.0005}
        }
        return ConvolutionModel(params), 'convolution_model.hdf5'

    @classmethod
    def dense_model(cls):
        params = {
            'num_feature_parts': 20,
            'layers': (256, 128),
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.001}
        }
        return DenseModel(params), 'dense_model_nfp-{nfp}_l-{layers}.hdf5'.format(
            nfp=params['num_feature_parts'],
            layers='-'.join(map(lambda x: str(x), params['layers']))
        )


if __name__ == "__main__":
    MODELS_FOLDER = "/home/alexander/Projects/LANLEarthquakePrediction/models/"
    TRAIN_DATA = "/home/alexander/Projects/LANLEarthquakePrediction/data/processed/train.pkl"
    VALID_DATA = "/home/alexander/Projects/LANLEarthquakePrediction/data/processed/validation.pkl"

    BATCH_SIZE = 64
    EPOCHS = 1000
    TRAIN_REPETITIONS = 50
    VALID_REPETITIONS = 10
    EARLY_STOP = 100

    with open(TRAIN_DATA, 'rb') as f:
        train_data = pickle.load(f)

    with open(VALID_DATA, 'rb') as f:
        valid_data = pickle.load(f)

    model, name = ModelConfigurator.dense_model()
    model.train(train_data, valid_data, MODELS_FOLDER+name, BATCH_SIZE, EPOCHS, TRAIN_REPETITIONS, VALID_REPETITIONS, EARLY_STOP)

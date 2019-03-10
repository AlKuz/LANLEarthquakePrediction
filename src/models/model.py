"""
Abstract model class
"""
from abc import ABC, abstractmethod
import numpy as np
import json
from src.data import DataGenerator


class Model(ABC):
    """
    Abstract class for all models
    """

    def __init__(self, params: dict, folder, name, seed=13):
        self._params = params
        self._model = None
        self._create_model()
        print("Model was created")
        self._name = name
        self._model_folder = folder

        np.random.seed(seed)
        with open(self._model_folder + self._name + '.info', 'w') as file:
            self._model.summary(print_fn=lambda x: file.write(x + '\n'))
            file.write('\n\n\n')
            file.write('epoch,\ttrain_loss,\ttest_loss\n')

        with open(self._model_folder + self._name + '.json', 'w') as file:
            json.dump(params, file)

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        raise Exception("Realize method")

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        raise Exception("Realize method")

    @abstractmethod
    def train(self, train: DataGenerator, valid: DataGenerator):
        pass

    @abstractmethod
    def _create_model(self) -> None:
        raise Exception("Realize method")

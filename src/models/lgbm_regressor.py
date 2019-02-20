"""
LGBM regressor
"""
from numpy import ndarray, min, max, mean, var, std, quantile, concatenate
import lightgbm as lgb
import pickle
import os

from src.models.model import Model


class LGBMRegressor(Model):
    """
    LGBM regressor
    """

    def _create_model(self, params):
        #self._model = lgb.LGBMRegressor(**params, n_estimators=20000, nthread=4, n_jobs=-1)
        self._model = lgb.LGBMRegressor(**params, n_jobs=-1)

    def predict(self, input_data: ndarray) -> ndarray:
        features = self._extract_features(input_data)
        return self._model.predict(features)

    def _fit_batch(self, train_x: ndarray, train_y: ndarray) -> float:
        features = self._extract_features(train_x)

        self._model.fit(features, train_y)

        model_result = self.predict(train_x)
        train_mae = self._calculate_mae(model_result, train_y)

        return train_mae

    def _extract_features(self, data: ndarray) -> ndarray:
        calc_statistics = lambda d: concatenate([
            min(d, axis=1, keepdims=True),
            max(d, axis=1, keepdims=True),
            mean(d, axis=1, keepdims=True),
            var(d, axis=1, keepdims=True),
            std(d, axis=1, keepdims=True),
            quantile(d, 0.25, axis=1, keepdims=True),
            quantile(d, 0.5, axis=1, keepdims=True),
            quantile(d, 0.75, axis=1, keepdims=True)
        ], axis=1)

        statistic_params_list = []
        statistic_params_list.append(calc_statistics(data))

        step = data.shape[1] // 10
        for i in range(0, data.shape[1], step):
            statistic_params_list.append(
                calc_statistics(data[:, i:i+step])
            )
        return concatenate(statistic_params_list, axis=1)

    def save_model(self, models_folder: str, model_name: str):
        with open(os.path.join(models_folder, model_name), 'wb') as f:
            pickle.dump(self._model, f)
        # TODO (kuznetsovav) Записывать вместе с моделью файл с логами и параметрами

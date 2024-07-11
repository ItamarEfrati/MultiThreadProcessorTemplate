import typing

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from sklearn.metrics import get_scorer

from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.training.hyperparameters_tuning.classification.classificationstudy import ClassificationStudy


class XGBootsStudy(ClassificationStudy):

    def __init__(self,
                 feature_type: str,
                 seed: int,
                 sampler: BaseSampler,
                 direction: str,
                 optimize_metric: str,
                 n_trials: int,
                 n_jobs: int,
                 cv: int,
                 n_jobs_forest: int,
                 xgb_objective: str,
                 booster: typing.Literal["gbtree", "dart"]):
        super().__init__(feature_type, seed, sampler, direction, optimize_metric, n_trials, n_jobs, cv)
        self.n_jobs_forest = n_jobs_forest
        self.xgb_objective = xgb_objective
        self.booster = booster
        #

    def get_study_name(self):
        return 'xgboots'

    def run_test(self, X_train, y_train, X_test, y_test, params):
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        model = XGBClassifier(**params)
        model.fit(np.vstack((X_train, X_test)), np.vstack((y_train.reshape(-1, 1), y_test.reshape(-1, 1))).reshape(-1))

        return {'train': [np.argmax(train_proba, -1), train_proba],
                'test': [np.argmax(test_proba, -1), test_proba]}, model

    def get_model(self, trial: optuna.Trial):
        param = {"objective": self.xgb_objective,
                 "n_estimators": trial.suggest_int('n_estimators', 10, 2000, step=10),
                 "max_depth": trial.suggest_int("max_depth", low=2, high=12),
                 "tree_method": trial.suggest_categorical("tree_method", ["exact", "auto"]),
                 "booster": self.booster,
                 "lambda": trial.suggest_float("lambda", 0, 10, step=0.01),
                 "alpha": trial.suggest_float("alpha", 0, 10, step=0.01),
                 # "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.01),
                 "seed": self.seed,
                 "nthread": self.n_jobs_forest,
                 "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 10, step=0.01),
                 "eta": trial.suggest_float("eta", 0.001, 0.5, step=0.001),
                 "gamma": trial.suggest_float("gamma", 0, 10, step=0.01),
                 # "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])}
                 }

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        model = XGBClassifier(**param)

        return model

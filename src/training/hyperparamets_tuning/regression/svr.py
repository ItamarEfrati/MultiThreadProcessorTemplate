import typing

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from sklearn.metrics import get_scorer, mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from src.classification.hyperparameters_tuning.abstract_study import Study


class SVRStudy(Study):

    def __init__(self,
                 feature_type: str,
                 seed: int,
                 sampler: BaseSampler,
                 direction: str,
                 optimize_metric: str,
                 n_trials: int,
                 n_jobs: int,
                 cv: int):
        super().__init__(feature_type, seed, sampler, direction, optimize_metric, n_trials, n_jobs)
        self.cv = cv
        self.params = {}

    def get_study_name(self):
        return 'svr'

    def run_test(self, X_train, y_train, X_test, y_test, params):
        model = SVR(**params)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        print("train mse ", mean_squared_error(y_train, train_preds))
        print("test mse ", mean_squared_error(y_test, test_preds))

        model = SVR(**params)
        model.fit(np.vstack((X_train, X_test)), np.concatenate([y_train, y_test]))

        return train_preds, train_preds, test_preds, test_preds, model

    def objective(self, trial: optuna.Trial, X, y, groups):
        C = trial.suggest_categorical('C', [1e-3, 1e-2, 1e-1, 1, 10, 50, 100])
        tol = trial.suggest_categorical('tol', [1e-4, 1e-3, 1e-2])
        epsilon = trial.suggest_categorical('epsilon', [1e-4, 1e-3,1e-2, 1e-1, 1, 5, 10])
        gamma = trial.suggest_categorical('gamma', [0.01, 0.1, 1, 'scale', 'auto'])
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])

        if kernel in ['poly', 'sigmoid']:
            degree = trial.suggest_int('degree', 2, 5)
            coef0 = trial.suggest_int('coef0', 0.0, 10.0) 
        else:
            coef0 = 1
            degree = 2
        
        model = SVR(C=C, gamma=gamma, kernel=kernel, degree=degree, epsilon=epsilon, coef0=coef0, tol=tol)

        if (kernel, C, tol, epsilon, gamma, degree, coef0) in self.params.keys():
            score = self.params[(kernel, C, tol, epsilon, gamma, degree, coef0)]
            trial.set_user_attr(self.optimize_metric, score)
        
        else:
            if self.cv > 1:
                score = cross_val_score(model, X, y, groups=groups, cv=self.cv, scoring=self.optimize_metric).mean()
            else:
                model.fit(X, y)
                y_pred = model.predict(X)
                score = get_scorer(self.optimize_metric)._score_func(y, y_pred)
            trial.set_user_attr(self.optimize_metric, score)
            self.params[(kernel, C, tol, epsilon, gamma, degree, coef0)] = score

        return score

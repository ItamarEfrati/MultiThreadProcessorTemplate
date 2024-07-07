import numpy as np
import optuna
from optuna.samplers import BaseSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.classification.hyperparameters_tuning.abstract_study import Study


class RandomForestRegressionStudy(Study):

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
                 leave_one_out: bool):
        super().__init__(feature_type, seed, sampler, direction, optimize_metric, n_trials, n_jobs)
        self.cv = cv
        self.n_jobs_forest = n_jobs_forest
        self.leave_one_out = leave_one_out

    def get_study_name(self):
        return 'random_forest'

    def run_test(self, X_train, y_train, X_test, y_test, params):
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=self.seed,
            ccp_alpha=params['ccp_alpha'],
            criterion=params['criterion'],
            n_jobs=self.n_jobs
        )

        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        return train_preds, test_preds, model

    def objective(self, trial: optuna.Trial, X, y, groups):
        n_estimators = trial.suggest_int('n_estimators', 10, 300, step=10)
        max_depth = trial.suggest_int('max_depth', 2, 16, log=True)
        min_samples_split = trial.suggest_float('min_samples_split', 0, 1.0, step=0.01)
        min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.1, 0.5, step=0.01)
        max_features = trial.suggest_int('max_features', 2, 16, log=True)
        ccp_alpha = trial.suggest_float('ccp_alpha', 0, 10, step=0.01)
        criterion = trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'poisson'])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.seed,
            ccp_alpha=ccp_alpha,
            criterion=criterion,
            n_jobs=self.n_jobs
        )
        cv = len(set(groups)) if self.leave_one_out else self.cv
        scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring=self.optimize_metric)
        score = scores.mean()
        trial.set_user_attr(self.optimize_metric, score)

        return score

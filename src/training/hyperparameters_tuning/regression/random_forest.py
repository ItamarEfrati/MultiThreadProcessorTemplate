import optuna
from optuna.samplers import BaseSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.training.hyperparameters_tuning.regression.regression_study import RegressionStudy


class RandomForestRegressionStudy(RegressionStudy):

    def __init__(self,
                 feature_type: str,
                 seed: int,
                 sampler: BaseSampler,
                 direction: str,
                 optimize_metric: str,
                 n_trials: int,
                 n_jobs: int,
                 cv: int,
                 n_jobs_forest: int):
        super().__init__(feature_type, seed, sampler, direction, optimize_metric, n_trials, n_jobs, cv)
        self.n_jobs_forest = n_jobs_forest

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

        return {'train': [train_preds], 'test': [test_preds]}, model

    def get_model(self, trial: optuna.Trial):
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

        return model

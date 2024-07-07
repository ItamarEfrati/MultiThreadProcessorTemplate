import numpy as np
import optuna
from optuna.samplers import BaseSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from classification.hyperparameters_tuning.abstract_study import Study


class RandomForestStudy(Study):

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
        super().__init__(feature_type, seed, sampler, direction, optimize_metric, n_trials, n_jobs)
        self.cv = cv
        self.n_jobs_forest = n_jobs_forest

    def get_study_name(self):
        return 'random_forest'

    def run_test(self, X_train, y_train, X_test, y_test, params):
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=self.seed,
            ccp_alpha=params['ccp_alpha'],
            class_weight=params['class_weight'],
            criterion=params['criterion'],
            n_jobs=self.n_jobs
        )

        model.fit(X_train, y_train)

        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        return np.argmax(train_proba, -1), train_proba, np.argmax(test_proba, -1), test_proba, model

    def objective(self, trial: optuna.Trial, X, y, groups):
        n_estimators = trial.suggest_int('n_estimators', 10, 10000, step=10)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)
        min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.1, 0.5)
        max_features = trial.suggest_int('max_features', 2, 16, log=True)
        ccp_alpha = trial.suggest_float('ccp_alpha', 0.01, 10)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None, 'balanced_subsample'])
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.seed,
            ccp_alpha=ccp_alpha,
            class_weight=class_weight,
            criterion=criterion,
            n_jobs=self.n_jobs
        )

        score = cross_val_score(model, X, y, groups=groups, cv=self.cv, scoring=self.optimize_metric).mean()
        trial.set_user_attr(self.optimize_metric, score)

        return score

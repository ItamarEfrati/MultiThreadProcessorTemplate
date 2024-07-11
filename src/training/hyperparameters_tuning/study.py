import logging
import os
import pickle
import shutil

from datetime import datetime

import optuna

from abc import ABC, abstractmethod
from functools import partial

from optuna.samplers import BaseSampler
from sklearn.model_selection import cross_val_score

from src.constants import *


class Study(ABC):

    def __init__(self,
                 feature_type: str,
                 seed: int,
                 sampler: BaseSampler,
                 direction: str,
                 optimize_metric: str,
                 n_trials: int,
                 n_jobs: int,
                 cv):
        self.seed = seed
        self.sampler = sampler
        self.direction = direction
        self.optimize_metric = optimize_metric
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.feature_type = feature_type
        self.cv = cv
        current_date = datetime.now()
        current_date = current_date.strftime("%d-%m-%Y_%H-%M-%S")
        self.working_dir = os.path.join(ASSETS_FOLDER, RUN_RESULTS, self.feature_type, self.get_study_name(),
                                        current_date)
        os.makedirs(self.working_dir, exist_ok=False)
        self.log = logging.getLogger()

    @abstractmethod
    def get_study_name(self):
        pass

    @abstractmethod
    def run_test(self, X_train, y_train, X_test, y_test, params):
        pass

    @abstractmethod
    def save_results(self, true, results, title):
        pass

    @abstractmethod
    def get_model(self, trial: optuna.Trial):
        pass

    def objective(self, trial: optuna.Trial, X, y, groups):
        model = self.get_model(trial)

        score = cross_val_score(model, X, y, groups=groups, cv=self.cv, scoring=self.optimize_metric).mean()
        trial.set_user_attr(self.optimize_metric, score)

        return score

    def run_study(self, data):
        X_train, y_train, train_groups, X_test, y_test = data
        storage_name = f"sqlite:///{os.path.join(self.working_dir, self.get_study_name())}.db"
        study = optuna.create_study(
            study_name=self.get_study_name(),
            sampler=self.sampler,
            storage=storage_name,
            pruner=optuna.pruners.MedianPruner(),
            direction=self.direction,
            load_if_exists=True
        )

        partial_objective = partial(self.objective, X=X_train, y=y_train, groups=train_groups)
        study.optimize(partial_objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        best_trial = study.best_trial
        best_metric = best_trial.user_attrs[self.optimize_metric]
        self.log.info(f"Best {self.optimize_metric}: {best_metric}")
        self.log.info("Saving trials data")

        final_results, model = self.run_test(X_train, y_train, X_test, y_test, best_trial.params)

        with open(os.path.join(self.working_dir, f'best_model_{self.get_study_name()}.obj'), 'wb') as f:
            pickle.dump(model, f)

        self.save_results(y_train, final_results['train'], 'train')
        self.save_results(y_test, final_results['test'], 'test')

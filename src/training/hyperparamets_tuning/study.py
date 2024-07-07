import logging
import os
import pickle
import shutil

from datetime import datetime

import optuna

from abc import ABC, abstractmethod
from functools import partial

from optuna.samplers import BaseSampler

from src.constants import *


class Study(ABC):

    def __init__(self,
                 feature_type: str,
                 seed: int,
                 sampler: BaseSampler,
                 direction: str,
                 optimize_metric: str,
                 n_trials: int,
                 n_jobs: int):
        self.seed = seed
        self.sampler = sampler
        self.direction = direction
        self.optimize_metric = optimize_metric
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.feature_type = feature_type
        current_date = datetime.now()
        current_date = current_date.strftime("%d-%m-%Y_%H-%M-%S")
        self.working_dir = os.path.join(ASSETS_FOLDER, RUN_RESULTS, self.feature_type, self.get_study_name(),
                                        current_date)
        os.makedirs(self.working_dir, exist_ok=False)
        self.log = logging.getLogger()

    @abstractmethod
    def objective(self, trial: optuna.Trial, X, y, groups):
        pass

    @abstractmethod
    def get_study_name(self):
        pass

    @abstractmethod
    def run_test(self, X_train, y_train, X_test, y_test, params):
        pass

    @abstractmethod
    def save_results(self, true, preds, prob, title):
        pass

    def run_study(self, data):
        X_train, y_train, X_test, y_test, train_groups = data
        storage_name = f"sqlite:///{self.get_study_name()}.db"
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

        train_preds, train_prob, test_preds, test_prob, model = self.run_test(X_train, y_train, X_test, y_test,
                                                                              best_trial.params)

        with open(os.path.join(self.working_dir, f'best_model_{self.get_study_name()}.obj'), 'wb') as f:
            pickle.dump(model, f)

        self.save_results(y_train, train_preds, train_prob, 'train')
        self.save_results(y_test, test_preds, test_prob, 'test')

        shutil.move(f"{self.get_study_name()}.db", self.working_dir)

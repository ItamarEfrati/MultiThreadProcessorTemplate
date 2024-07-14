import itertools
import os
import pathlib

import numpy as np

from abc import ABC, abstractmethod
from typing import Literal

from src.constants import DATA_FOLDER, PREPROCESS, PREPROCESS_OUTCOMES


class DataLoader(ABC):

    def __init__(self,
                 seed: int,
                 split_data: bool,
                 pipeline_type: str,
                 labels_path: str,
                 pattern: str):
        self.pattern = pattern
        self.seed = seed
        self.split_data = split_data
        self.pipeline_type = pipeline_type
        self.labels_path = labels_path
        np.random.seed(self.seed)

    @abstractmethod
    def _load_features_file(self, features_file):
        pass

    @abstractmethod
    def _format_features(self, features) -> np.ndarray:
        pass

    @abstractmethod
    def _load_train_groups(self):
        pass

    @abstractmethod
    def _get_train_indices(self):
        pass

    @abstractmethod
    def _load_labels(self, split: Literal["split", "all"]):
        pass

    def _load_features(self, split: Literal["split", "all"]):
        data_dir = pathlib.Path(DATA_FOLDER, PREPROCESS, PREPROCESS_OUTCOMES + '_' + self.pipeline_type)
        train_indices = self._get_train_indices()

        features = []
        indices = []
        for features_file in data_dir.rglob("**" + os.sep + self.pattern):
            features.append(self._load_features_file(features_file))
            indices.append(features_file.relative_to(data_dir).parts[0])

        X = self._format_features(features)

        if split == 'split':
            train_mask = np.isin(indices, train_indices)
            test_mask = ~train_mask
            X_train = X[train_mask]
            X_test = X[test_mask]

            return X_train, X_test
        else:
            return X

    def load_data(self):
        data = []
        if self.split_data:
            data.append(self._load_features("split"))
            data.append(self._load_labels("split"))
            data.append(self._load_train_groups())
        else:
            data.append(self._load_features("all"))
            data.append(self._load_labels("all"))
        return tuple(itertools.chain.from_iterable(data))

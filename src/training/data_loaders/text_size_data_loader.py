import random
from typing import Literal

import numpy as np

from src.training.data_loaders.data_loader import DataLoader


class TextSizeDataLoader(DataLoader):

    def _load_features_file(self, features_file):
        features = []
        with open(features_file, 'r') as f:
            size = f.readlines()
            features.append(int(size[0]))
        return features

    def _format_features(self, features) -> np.ndarray:
        return np.concat(features).reshape(-1, 1)

    def _get_train_indices(self):
        return list(['1', '2', '3'])

    def _load_labels(self, split: Literal["train", "val", "test", "all", "predict"]):
        """
        some random binary labels for example
        :param split:
        :return:
        """
        y = np.array([random.randint(0, 1) for _ in range(5)])
        if split == 'split':
            y_train = np.array(y[:3])
            y_test = np.array(y[3:])

            return y_train, y_test
        else:
            return y

    def _load_train_groups(self):
        return (np.array([0, 1, 2]),)

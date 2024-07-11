import numpy as np

from abc import ABC, abstractmethod
from typing import Literal


class DataLoader(ABC):

    def __init__(self,
                 seed: int,
                 split_data: bool,
                 load_test: bool,
                 load_train: bool,
                 load_predict: bool,
                 pipeline_type: str,
                 labels_path: str):
        self.seed = seed
        self.split_data = split_data
        self.load_test = load_test
        self.load_train = load_train
        self.load_predict = load_predict
        self.pipeline_type = pipeline_type
        self.labels_path = labels_path
        np.random.seed(self.seed)

    @abstractmethod
    def _load_features(self, split: Literal["train", "test", "all", "predict"]):
        pass

    @abstractmethod
    def _load_labels(self, split: Literal["train", "test", "all", "predict"]):
        pass

    def load_data(self):
        data = []
        if self.split_data:
            if self.load_train:
                data.append(self._load_features("train"))
                data.append(self._load_labels("train"))
                data.append(list(range(100))) # group
            if self.load_test:
                data.append(self._load_features("test"))
                data.append(self._load_labels("test"))

        elif self.load_predict:
            data.append(self._load_features("predict"))
        else:
            data.append(self._load_features("all"))
            data.append(self._load_labels("all"))
        return tuple(data)

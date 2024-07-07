import numpy as np

from abc import ABC, abstractmethod
from typing import Literal


class DataLoader(ABC):

    def __init__(self,
                 seed: int,
                 split_data: bool,
                 load_test: bool,
                 load_train: bool,
                 load_predict: bool):
        self.seed = seed
        self.split_data = split_data
        self.load_test = load_test
        self.load_train = load_train
        self.load_predict = load_predict
        np.random.seed(self.seed)

    @abstractmethod
    def _load_features(self, split: Literal["train", "val", "test", "all", "predict"]):
        pass

    @abstractmethod
    def _load_labels(self):
        pass

    @abstractmethod
    def _get_split_indices(self):
        pass

    def load_data(self):
        data = ()
        if self.split_data:
            if self.load_train:
                data += self._load_features("train")
                data += self._load_features("val")
            if self.load_test:
                data += self._load_features("test")

        elif self.load_predict:
            data = self._load_features("predict")
        else:
            data = self._load_features("all")
        return data

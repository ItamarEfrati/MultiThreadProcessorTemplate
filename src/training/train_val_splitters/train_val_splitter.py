from abc import abstractmethod

from sklearn.model_selection import BaseCrossValidator


class TrainValSplitter(BaseCrossValidator):

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    @abstractmethod
    def split(self, X, y=None, groups=None):
        pass

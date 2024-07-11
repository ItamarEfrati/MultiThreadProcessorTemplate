from src.training.train_val_splitters.train_val_splitter import TrainValSplitter


class TextSizeTrainValSplitter(TrainValSplitter):
    def split(self, X, y=None, groups=None):
        yield list(range(50)), list(range(50, 100))
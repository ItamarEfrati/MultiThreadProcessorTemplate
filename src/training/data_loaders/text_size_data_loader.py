import os
import pathlib
import random
from typing import Literal

import numpy as np

from src.constants import *

from src.training.data_loaders.data_loader import DataLoader


class TextSizeDataLoader(DataLoader):

    def _load_features(self, split: Literal["train", "val", "test", "all", "predict"]):
        data_dir = pathlib.Path(DATA_FOLDER, PREPROCESS, PREPROCESS_OUTCOMES + '_' + self.pipeline_type)

        pattern = 'text_counter_length.txt'
        features = {}
        for features_file in data_dir.rglob("**" + os.sep + pattern):
            with open(features_file, 'r') as f:
                size = f.readlines()
                features[features_file.parent.name] = int(size[0])

        for i in range(2, 100):
            # some random features
            features[i] = random.randint(0, 100)

        return np.array(list(features.values())).reshape(-1, 1)

    def _load_labels(self, split: Literal["train", "val", "test", "all", "predict"]):
        """
        some random binary labels for example
        :param split:
        :return:
        """
        return np.array([random.randint(0, 1) for _ in range(100)])


import json
import os

from time import gmtime, strftime

import pandas as pd
import numpy as np

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

from src.dataloading import DataLoader, download_all

from sklearn.model_selection import train_test_split


class BenchMark:

    def __init__(self, dataset_exclude=None, model_exclude=None,
                 output_dir=None):

        self.output_dir = self._set_output_dir(output_dir)
        self.datasets = download_all(dataset_exclude)

        all_models = ['ABOD', 'CBLOF', 'FeatureBagging', 'HBOS', 'IForest',
                      'MCD', 'OCSVM', 'PCA']
        self.models = list(set(all_models) - set(model_exclude))


        df_columns = ['Data', 'nSamples', 'nDimensions', 'OutlierPerc', 'ABOD',
                      'CBLOF', 'FB', 'HBOS', 'IForest', 'MCD', 'OCSVM', 'PCA']

        # initialize the container for saving the results
        self.df_auc = pd.DataFrame(columns=df_columns)
        self.df_topn = pd.DataFrame(columns=df_columns)
        self.df_train_time = pd.DataFrame(columns=df_columns)
        self.df_test_time = pd.DataFrame(columns=df_columns)

        self.iter_count = 0

        # TODO: copy from https://github.com/yzhao062/pyod/blob/master/notebooks/benchmark.py
        # TODO: same format as https://pyod.readthedocs.io/en/latest/benchmark.html

    def run(self, n_iterations):

        for dataset in self.datasets:

            dataloader = DataLoader(dataset)
            X, y = dataloader.get_X(), dataloader.get_y()

            for i in range(n_iterations):

                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=0.4)

    @staticmethod
    def _set_output_dir(output_dir):

        if output_dir is None:
            suffix = strftime("%Y-%m-%d-%H-%M", gmtime())
            output_dir = os.path.join("../benchmark", suffix)

        os.makedirs(output_dir, exist_ok=True)

        return output_dir







if __name__ == "__main__":

    datasets = download_all()
    print(datasets)

    print(np.random.RandomState(1))
    exit()

    print(datasets)

    benchmark = BenchMark(output_dir="../benchmark")

    print('0')

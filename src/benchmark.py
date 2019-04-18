
import json
import os

from time import time, gmtime, strftime

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

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score

from src.dataloading import DataLoader, download_all

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BenchMark:

    def __init__(self, dataset_exclude=None, output_dir=None):

        self.output_dir = self._set_output_dir(output_dir)
        self.datasets = download_all(dataset_exclude)
        self.models = dict()

        df_columns = ['Data', 'nSamples', 'nDimensions', 'OutlierPerc', 'ABOD',
                      'CBLOF', 'FB', 'HBOS', 'IForest', 'MCD', 'OCSVM', 'PCA']

        # initialize the container for saving the results
        self.df_auc = pd.DataFrame(columns=df_columns)
        self.df_topn = pd.DataFrame(columns=df_columns)
        self.df_train_time = pd.DataFrame(columns=df_columns)
        self.df_test_time = pd.DataFrame(columns=df_columns)

        self.iter_count = 0

    @staticmethod
    def _draw_param_set(param_grid):
        return {k: sampler() for k, sampler in param_grid.items()}

    @staticmethod
    def _fit_classifier(model, param_grid, X_train):

        # Draw parameter set
        param_set = BenchMark._draw_param_set(param_grid)

        # Make model and set parameters
        clf = model()
        clf.set_params(**param_set)

        # Fit model and measure training time
        t_start = time()
        clf.fit(X_train)
        t_stop = time()

        return clf, round(t_stop - t_start, ndigits=4)

    @staticmethod
    def _test_classifier(clf, y_test):

        # Apply model and measure test time
        t_start = time()
        y_fit = clf.decision_function(y_test)
        t_stop = time()

        return y_fit, round(t_stop - t_start, ndigits=4)

    @staticmethod
    def _eval_classifier(y_test, y_fit):

        roc = round(roc_auc_score(y_test, y_fit), ndigits=4)
        topn = round(precision_n_scores(y_test, y_fit), ndigits=4)

        return roc, topn

    @staticmethod
    def _check_models(models):

        # Check if models are already added to benchmarking experiment
        if not models:
            raise Exception('No models added yet! --> use .add_model()')

        # Check if all models have a parameter grid specified
        no_param_grid = []
        for model_name, model_holder in models.items():
            if not model_holder["param_grid"]:
                no_param_grid.append(model_name)
        if no_param_grid:
            raise Exception("No param_grid added for: {}"
                            .format(no_param_grid))

        # Generate random dummy data and check if models can be applied with
        # the current parameter grid (single draw)
        dummy_X = np.random.randn(100, 6)

        for model_name, model_holder in models.items():

            # Extract model and parameter grid
            model = model_holder["model"]
            param_grid = model_holder["param_grid"]

            # Fit classifier, measure train time
            clf, train_time = \
                BenchMark._fit_classifier(model, param_grid, dummy_X)

            # Apply classifier, measure test time
            _, test_time = BenchMark._test_classifier(clf, dummy_X)

            print("Check {} model on dummy data, train time: {}, test time: {}"
                  .format(model_name, train_time, test_time))

    def run(self, n_iterations):

        self._check_models(self.models)
        exit()

        for dataset in self.datasets:

            dataloader = DataLoader(dataset)
            X, y = dataloader.get_X(), dataloader.get_y()

            for i in range(n_iterations):

                random_state = np.random.RandomState(self.iter_count)

                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=0.4,
                                     random_state=random_state)

                # Scale train and test data, fit only based on train data!
                X_scaler = StandardScaler()
                X_train = X_scaler.fit_transform(X_train)
                X_test = X_scaler.transform(X_test)

                for model_name, model_holder in self.models.items():

                    # Extract model and parameter grid
                    model = model_holder["model"]
                    param_grid = model_holder["param_grid"]

                    # Fit classifier, measure train time
                    clf, train_time = \
                        self._fit_classifier(model, param_grid, X_train)

                    # Apply classifier, measure test time
                    y_fit, test_time = self._test_classifier(clf, X_test)

                    roc, topn = self._eval_classifier(y_test, y_fit)



    def add_model(self, model_name, model):

        self.models[model_name] = {
            "model": model,
            "param_grid": dict()
        }

    def add_model_param(self, model_name, param_name, param_generator):

        self.models[model_name]["param_grid"][param_name] = param_generator


    @staticmethod
    def _set_output_dir(output_dir):

        if output_dir is None:
            suffix = strftime("%Y-%m-%d-%H-%M", gmtime())
            output_dir = os.path.join("../benchmark", suffix)

        os.makedirs(output_dir, exist_ok=True)

        return output_dir



if __name__ == "__main__":

    benchmark = BenchMark(output_dir="../benchmark")
    benchmark.add_model("Isolation Forest", IForest)
    #benchmark.add_model_param("Isolation Forest", "behaviour", lambda: "new")
    benchmark.add_model_param("Isolation Forest", "n_estimators", lambda: np.random.randint(50, 1000))
    benchmark.add_model("uCBLOF", CBLOF)
    benchmark.add_model_param("uCBLOF", "n_clusters", lambda: np.random.randint(3, 100))
    benchmark.run(10)

    exit()

    print('0')

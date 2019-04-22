
import json
import os
import glob

from time import time, gmtime, strftime

import pandas as pd
import numpy as np

from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score

from src.dataloading import DataLoader, download_all

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# TODO: write docstrings for exposed functions


class BenchMark:

    def __init__(self, dataset_exclude=None, output_dir=None):

        self.output_dir = self._set_output_dir(output_dir)
        self.datasets = download_all(dataset_exclude)

        self.models = dict()
        self._latest_added_model = None
        self.results = []

    def load_previous_result(self, dir=None):

        if not dir:
            parent_dir = "../benchmark"
            benchmark_dirs = glob.glob(parent_dir + "/*")
            benchmark_dirs.sort()
            dir = benchmark_dirs[-2]  # Before last element (current is last)

        fp = os.path.join(dir, "backup.json")

        with open(fp, "r") as json_file:
            self.results = json.load(json_file)

        print("Loaded {} runs from previous benchmark"
              .format(len(self.results)))

        prev_models = set([x["model"] for x in self.results])
        prev_datasets = set([x["dataset"] for x in self.results])
        print("--> Previous models: {}".format(prev_models))
        print("--> Previous datasets: {}\n".format(prev_datasets))

    @staticmethod
    def _set_output_dir(output_dir):

        if output_dir is None:
            suffix = strftime("%Y-%m-%d-%H-%M", gmtime())
            output_dir = os.path.join("../benchmark", suffix)

        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    @staticmethod
    def _draw_param_set(param_grid):
        return {k: sampler() for k, sampler in param_grid.items()}

    @staticmethod
    def _fit_classifier(model, param_grid, X_train, max_tries=10):

        for i in range(max_tries):

            try:

                # Draw parameter set
                param_set = BenchMark._draw_param_set(param_grid)

                # Make model and set parameters
                clf = model()
                clf.set_params(**param_set)

                # Fit model and measure training time
                t_start = time()
                clf.fit(X_train)
                t_stop = time()

                return clf, param_set, round(t_stop - t_start, ndigits=4)

            except ValueError:

                print("Fit model {} failed: {} try".format(model, i + 1))

        raise ValueError("Fit step model {} failed due to incorrect params"
                         .format(model))

    @staticmethod
    def _test_classifier(clf, y_test):

        # Apply model and measure test time
        t_start = time()
        y_fit = clf.decision_function(y_test)
        t_stop = time()

        return y_fit, round(t_stop - t_start, ndigits=4)

    @staticmethod
    def _eval_classifier(y_test, y_fit):

        auc = round(roc_auc_score(y_test, y_fit), ndigits=4)
        topn = round(precision_n_scores(y_test, y_fit), ndigits=4)

        return auc, topn

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

            # Draw new parameter set, fit classifier, measure train time
            clf, param_set, train_time = \
                BenchMark._fit_classifier(model, param_grid, dummy_X)

            # Apply classifier, measure test time
            _, test_time = BenchMark._test_classifier(clf, dummy_X)

            print("Check on dummy data: train time: {}s, test time: {}s -- {}"
                  .format(train_time, test_time, model_name))

    def _add_result(self, model_name, data_name, auc, topn, train_time,
                    test_time, param_set):

        result = {
            "model": model_name,
            "dataset": data_name,
            "auc": auc,
            "topn": topn,
            "train_time": train_time,
            "test_time": test_time,
            "parameters": param_set
        }

        self.results.append(result)

        file_path = os.path.join(self.output_dir, "backup.json")
        with open(file_path, "w") as json_file:
            json.dump(self.results, json_file, indent=4)

    def _write_results(self):

        path_dir = os.path.join(self.output_dir, "results")
        os.mkdir(path_dir)

        fp = os.path.join(path_dir, "full.json")
        with open(fp, "w") as json_file:
            json.dump(self.results, json_file, indent=4)

        metric_cols = ["model", "dataset", "auc", "topn", "train_time",
                       "test_time"]
        df_metrics = pd.DataFrame(self.results, columns=metric_cols)
        df_metrics.to_csv(os.path.join(path_dir, "metrics.csv"))

        for model_name in self.models.keys():

            param_dfs = []
            for result in self.results:
                if result["model"] == model_name:
                    param_df = pd.io.json.json_normalize(result)
                    param_dfs.append(param_df)

            df_param = pd.concat(param_dfs, axis=1)
            fp = os.path.join(path_dir, "param_{}.csv".format(model_name))
            df_param.to_csv(fp)


    def run(self, n_iterations):

        self._check_models(self.models)

        for data_name in self.datasets:

            dataloader = DataLoader(data_name)
            X, y = dataloader.get_X(), dataloader.get_y()

            for _ in range(n_iterations):

                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=0.4)

                # Scale train and test data, fit only based on train data!
                X_scaler = StandardScaler()
                X_train = X_scaler.fit_transform(X_train)
                X_test = X_scaler.transform(X_test)

                for model_name, model_holder in self.models.items():

                    # Extract model and parameter grid
                    model = model_holder["model"]
                    param_grid = model_holder["param_grid"]

                    # Fit classifier, measure train time
                    clf, param_set, train_time = \
                        self._fit_classifier(model, param_grid, X_train)

                    # Apply classifier, measure test time
                    y_fit, test_time = self._test_classifier(clf, X_test)

                    auc, topn = self._eval_classifier(y_test, y_fit)

                    self._add_result(model_name=model_name,
                                     data_name=data_name,
                                     auc=auc,
                                     topn=topn,
                                     train_time=train_time,
                                     test_time=test_time,
                                     param_set=param_set)

        self._write_results()

    def add_model(self, model_name, model):

        self.models[model_name] = {
            "model": model,
            "param_grid": dict()
        }

        self._latest_added_model = model_name

        return self

    def add_param(self, param_name, param_generator, model_name=None):

        if not model_name:

            if not self._latest_added_model:
                raise Exception("First add model with .add_model()")

            model_name = self._latest_added_model

        self.models[model_name]["param_grid"][param_name] = param_generator

        return self

    def plot_params(self, model_name=None, n_draws=1000, max_plot_cols=3):

        if not model_name:

            if not self._latest_added_model:
                raise Exception("First add model with .add_model()")

            model_name = self._latest_added_model

        # Extract model and parameter grid
        param_grid = self.models[model_name]["param_grid"]

        rows_list = []
        for _ in range(n_draws):
            param_dict = self._draw_param_set(param_grid)
            rows_list.append(param_dict)
        df_params = pd.DataFrame(rows_list)

        n_plots = len(param_grid)
        n_plt_cols = int(min(n_plots, max_plot_cols))
        n_plt_rows = int(np.ceil(n_plots / max_plot_cols))

        num_cols = df_params._get_numeric_data().columns

        plt.figure(figsize=(n_plt_cols * 4 + 3, n_plt_rows * 3 + 2))
        plt.subplots_adjust(wspace=.4, hspace=.3)
        plt.suptitle("Parameters for {}".format(model_name))

        for i, col in enumerate(df_params):

            plt.subplot(n_plt_rows, n_plt_cols, i + 1)
            plt.title(col)

            if col in num_cols:
                df_params[col].astype(float).plot.hist()
            else:
                df_params[col].value_counts().plot.bar()
                plt.ylabel('Frequency')

        plt.show()


if __name__ == "__main__":

    from pyod.models.cblof import CBLOF
    from pyod.models.iforest import IForest

    dataset_exclude = ["annthyroid", "ecoli", "kdd-http", "kdd-smtp",
                       "shuttle", "forest-cover", "mammography", "glass",
                       "lympho"]
    benchmark = BenchMark(dataset_exclude=dataset_exclude)


    # Isolation forest
    benchmark.add_model("Isolation Forest", IForest)
    benchmark.add_param("contamination", lambda: round(np.random.uniform(0.01, 0.3), ndigits=3))
    benchmark.add_param("n_estimators", lambda: np.random.randint(50, 1000))
    benchmark.add_param("max_features", lambda: np.random.randint(1, 4))
    benchmark.add_param("bootstrap", lambda: int(np.random.choice([0, 1])))

    # uCBLOF
    benchmark.add_model("uCBLOF", CBLOF)
    benchmark.add_param("n_clusters", lambda: np.random.randint(3, 50))

    benchmark.run(3)

    exit()

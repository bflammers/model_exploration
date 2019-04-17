
import pandas as pd
import numpy as np
import os
import json
import requests
import scipy.io
import h5py

class DataLoader:

    def __init__(self, dataset_name, data_dir="../data"):
        """Initialize DataLoader object.

            Keyword arguments:
            dataset_name -- kdd-smtp, kdd-http, forest-cover or shuttle
        """
        print("Dataset name: {dset}".format(dset=dataset_name))

        self._data_dir = data_dir
        self._dataset_dir = os.path.join(data_dir, dataset_name)
        self._dataset_name = dataset_name

        self.odds_datasets = ["lympho", "glass", "vowels", "thyroid",
                              "shuttle", "kdd-http", "forest-cover",
                              "kdd-smtp", "mammography", "annthyroid", "ecoli"]

        if not os.path.isdir(data_dir):
            raise Exception("{} directory not present".format(data_dir))

        print("--> reading json with dataset attributes")
        with open(os.path.join(data_dir, "datasets.json"), "r") as json_file:
            data_attributes = json.load(json_file)

        self._URL = data_attributes[dataset_name]["URL"]
        self._file_type = data_attributes[dataset_name]["file_type"]
        self._unlabelled = bool(data_attributes[dataset_name]["is_unlabelled"])

        self._raw_data_file = os.path.join(self._dataset_dir,
                                          "raw_data" + self._file_type)
        self._data_present = self._check_data_present()

        if not self._data_present:
            print("--> no data present")
            self._download_data()
        else:
            print("--> data is present")

        self._df = None
        self._read_data()
        print("--> data loaded, shape: {}".format(self._df.shape))

    def __repr__(self):
        return ("DataLoader object for {dset} dataset with {rows} rows and "
                "{cols} columns").format(dset=self._dataset_name,
                                         rows=self._df.shape[0],
                                         cols=self._df.shape[1])

    @staticmethod
    def _check_dir_present(dir_path):
        if os.path.isdir(dir_path):
            print("--> {} directory present".format(dir_path))
        else:
            os.mkdir(dir_path)

    def _check_data_present(self):
        self._check_dir_present(self._data_dir)
        self._check_dir_present(self._dataset_dir)
        return os.path.exists(self._raw_data_file)

    def _download_data(self):

        print("--> downloading {} dataset from {}"
              .format(self._dataset_name, self._URL))

        r = requests.get(self._URL)
        data_file = r.content

        print("--> writing raw data to {}".format(self._raw_data_file))
        if self._file_type == ".mat":
            with open(self._raw_data_file, 'wb') as out_file:
                out_file.write(data_file)
        elif self._file_type == ".csv":
            raise NotImplementedError("filetype: .csv")
        else:
            raise Exception("Unknown filetype: {}".format(self._file_type))

        self._data_present = True

    def _reads_odds_stonybrook_data(self):

        try:
            raw = scipy.io.loadmat(self._raw_data_file)
        except NotImplementedError:
            with h5py.File(self._raw_data_file, 'r') as in_file:
                raw = {"X": np.transpose(np.array(in_file["X"])),
                       "y": np.transpose(np.array(in_file["y"]))}

        self._df = pd.DataFrame(raw['X'])
        self._df["y"] = raw["y"]

    def _read_data(self):

        if not self._data_present:
            raise Exception("data not present")
        elif self._dataset_name in self.odds_datasets:
            self._reads_odds_stonybrook_data()
        else:
            raise Exception("Dataset present but _read_data not supported")

    def get_X(self):

        if not self._data_present:
            raise Exception("Data not present")

        return self._df.drop(columns=["y"])

    def get_y(self):

        if not self._data_present:
            raise Exception("Data not present")
        elif self._unlabelled:
            raise Exception("Dataset present but not labelled")

        return self._df["y"]


def download_all(exclude=None):

    print("--> reading json with dataset attributes")
    with open("../data/datasets.json", "r") as json_file:
        data_attributes = json.load(json_file)

    # Drop datasets without name
    data_attributes.pop("", None)

    for k in data_attributes.keys():
        if exclude is None or k not in exclude:
            DataLoader(dataset_name=k)

    return list(data_attributes.keys())


if __name__ == "__main__":

    dataloader = DataLoader('kdd-smtp')
    print("X shape: ", dataloader.get_X().shape)
    print("y shape: ", dataloader.get_y().shape)

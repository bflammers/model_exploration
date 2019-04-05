
import pandas as pd
import numpy as np
import os
import json

class DataLoader:

    def __init__(self, dataset_name, data_dir="../data"):
        """Initialize DataLoader object.

            Keyword arguments:
            dataset_name -- kdd-smtp, kdd-http, forest-cover or shuttle
        """
        self.data_dir = data_dir
        self.dataset_dir = os.path.join(data_dir, dataset_name)
        self.dataset_name = dataset_name
        self.data_present = False

        if not os.path.isdir(data_dir):
            raise Exception("{} directory not present".format(data_dir))

        print("--> reading data attributes json")
        with open(os.path.join(data_dir, "data.json"), "r") as json_file:
            data_attributes = json.load(json_file)

        self.dataset_attributes = data_attributes[dataset_name]

        self._check_data_present()


    @staticmethod
    def _check_dir_present(dir_path):
        if os.path.isdir(dir_path):
            print("--> {} directory present".format(dir_path))
        else:
            os.mkdir(dir_path)

    def _check_data_present(self):
        self._check_dir_present(self.data_dir)
        self._check_dir_present(self.dataset_dir)

        X_present = os.path.exists(os.path.join(self.dataset_dir, "X"))
        y_present = os.path.exists(os.path.join(self.dataset_dir, "y"))

        if not X_present and not y_present:
            print("--> no data present")
            self._download_data()
        elif X_present and y_present:
            print("--> data is present")
            self.data_present = True
        else:
            raise Exception("Only X or y present, not both.")

    def _download_data(self):
        print("--> downloading {} dataset from {}"
              .format(self.dataset_name, self.dataset_attributes["URL"]))
        pass

    def get_X(self):
        if not self.data_present:
            raise Exception("Data not present")
        pass

    def get_y(self):
        if not self.data_present:
            raise Exception("Data not present")
        pass


if __name__ == "__main__":
    print('Yess')

    dataloader = DataLoader('kdd')
    print(dataloader._check_dir_present())
    dataloader.get_X()
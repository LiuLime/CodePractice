import os

import numpy as np
import pandas as pd
import sys

sys.path.append("..")

from dpm.utils import common, log
from dpm.model_ML import dataloader as load

logger = log.logger()

config_ml = common.read_yaml("../config.yaml")
# batch_subgroup_mean_metric save path
subgroup_mean_metric_xlsx = os.path.join(config_ml["path"]["metric1"],
                                         config_ml["file_name"]["subgroup_mean_metric_xlsx"])
subgroup_mean_metric_csv = os.path.join(config_ml["path"]["metric1"],
                                        config_ml["file_name"]["subgroup_mean_metric_csv"])
all_mean_metric_path = os.path.join(config_ml["path"]["metric1"],
                                    config_ml["file_name"]["all_mean_metric"])

# train small datasets path
minimum_value = config_ml["screen"]["minimum_value"]
train_dataset_path = os.path.join(config_ml["path"]["metric1"], config_ml["file_name"]["train_dataset"])
train_dataset_path = common.insert_str(train_dataset_path, str(minimum_value), ".")
dataset_size_df = common.read_csv(train_dataset_path)
# train small datasets size dictionary
dataset_size = dataset_size_df.set_index("system_pairs")["counts"].to_dict()


def load_features_labels():
    train_dataset, descriptors, gradients, metadata = load.read_dataset()
    print("----load finish")
    d = load.DatasetArray(descriptors, gradients, metadata, train_dataset)
    groups, features, labels = d.get_features_labels()

    # dataset_tensor = load.CustomDataset(descriptors, gradients, metadata, train_dataset)
    generator = d.data_generator(groups, features, labels)
    return generator


"""Performance evaluation"""


class Calculate:
    def __init__(self, init_rt, target_rt, preds_rt, epsilon=1e-10):
        """calculate RE, MRE"""
        self.init_rt = np.array(init_rt)
        self.true_rt = np.array(target_rt)
        self.preds_rt = preds_rt
        self.epsilon = epsilon
        print("init rt-------", self.init_rt.shape)
        print("true rt-------", self.true_rt.shape)

    def MRE(self) -> float:
        """return average value of normalized absolute errors for all prediction"""
        return np.mean(self.RE())

    def RE(self) -> np.ndarray:
        """return list of normalized absolute errors for each prediction"""
        initial_error = self.true_rt - self.init_rt
        prediction_error = self.true_rt - self.preds_rt
        RE = np.abs(prediction_error / (initial_error + self.epsilon))
        return RE

    def initial_error(self) -> np.ndarray:
        initial_error = self.true_rt - self.init_rt
        return initial_error


def metric_mean(s):
    std = s.std(numeric_only=True)
    mean = s.mean(numeric_only=True)
    return format(mean, ".3f") + "±" + format(std, ".3f")


"""collate 5-fold into average form in each datasets"""


class Statistic:
    def __init__(self, metric_data):
        self.metric_data = pd.DataFrame(data=metric_data)
        self.metric_slice = self.metric_data.loc[:, ['MSE_eval', 'RMSE_eval',
                                                     'MAE_eval', 'MedAE_eval',
                                                     'R2_eval', 'personr_eval',
                                                     "sub_group"]].copy()
        self.pred_slice = self.metric_data.loc[:, ["sub_group", "X_test_idx", "y_test", "y_preds"]].copy()
        # self.pred_slice = (pred_slice.groupby(by=["sub_group"]).apply(Statistic.concate_fold)
        #                    .reset_index())  # 合并同一个subgroup的list
        self.generator = load_features_labels()

    # @staticmethod
    # def concate_fold(groups):
    #     for idx, rows in groups.iterrows():
    #         concate_list=[]
    #     concate_list =list(*)
    #     # print(x)

    def subgroup_mean_metric(self) -> pd.DataFrame:
        """return metrics mean and std value across records by sub_group"""
        df = self.metric_slice.groupby(by=["sub_group"]).agg([np.mean, np.std], axis=0).reset_index()
        return df

    def all_mean_metric(self) -> pd.DataFrame:
        """return metrics description dataframe across all records"""
        metric_slice = self.metric_slice.drop(columns=["sub_group"])
        return metric_slice.agg(["mean", "std"], axis=0)

    def get_xlabel_in_generator(self, subgroup):
        """return X_labels (init_rt) for each subgroup"""
        for g, f, l in self.generator:
            if g == subgroup:
                return l

    def collate_preds(self, ):
        """calculate RE, MRE"""
        for idx, row in self.pred_slice.iterrows():
            subgroup = row["sub_group"][0]
            X_labels = self.get_xlabel_in_generator(subgroup)
            print(type(row["y_test"]))
            calc = Calculate(X_labels, row["y_test"], row["y_preds"])
            MRE = calc.MRE()
            print(MRE)
            break


def batch_subgroup_mean_metric(algorithm_list: list, save_file=True, read_save_file=True):
    """collate mean metric for algorithms, return mean metric dataframe summarizing all datasets or individual
    datasets"""
    if common.file_is_exist(subgroup_mean_metric_csv) and read_save_file is True:
        logger.debug("--load subgroup_mean_metric.csv")
        subgroup_mean_metric_merge = common.read_csv(subgroup_mean_metric_csv, delimiter_=",", header_=[0, 1])
        all_mean_metric_merge = common.read_csv(all_mean_metric_path, delimiter_=",", header_=[0, 1])
    else:
        arrays = [[item for item in algorithm_list for _ in range(2)],
                  ['mean', 'std'] * 7]
        columns = pd.MultiIndex.from_arrays(arrays, names=('algorithm', 'stat'))

        all_mean_metric_list = []
        subgroup_mean_metric_list = []
        for model in algorithm_list:
            metric_path = f"../model_ML/output/{model}_metric.json"
            metric_data = common.read_json(metric_path)
            s = Statistic(metric_data)
            subgroup_mean_metric = s.subgroup_mean_metric()
            subgroup_mean_metric.loc[:, ("algorithm", "")] = model
            subgroup_mean_metric.loc[:, ("dataset_size", "")] = subgroup_mean_metric.loc[:, ("sub_group", "")].map(
                dataset_size)
            subgroup_mean_metric_list.append(subgroup_mean_metric)
            all_mean_metric_list.append(s.all_mean_metric().T)
            if save_file:
                common.save_xlsx(subgroup_mean_metric, subgroup_mean_metric_xlsx, sheet_name_=model, index_=True)

        all_mean_metric_merge = pd.concat(all_mean_metric_list, join="outer", axis=1).set_axis(columns, axis=1)
        subgroup_mean_metric_merge = pd.concat(subgroup_mean_metric_list, axis=0).sort_values(by="dataset_size",
                                                                                              ascending=True)
        if save_file:
            common.save_csv(all_mean_metric_merge, all_mean_metric_path, _index=True)
            common.save_csv(subgroup_mean_metric_merge, subgroup_mean_metric_csv, _index=True)
    print(subgroup_mean_metric_merge.columns)
    return all_mean_metric_merge, subgroup_mean_metric_merge

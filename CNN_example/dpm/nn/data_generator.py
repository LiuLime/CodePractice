import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset

from dpm.config_ import DLParamConfig, PrepConfig
from dpm.feature import feature
from dpm.utils import log, common

logger = log.logger()

from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os


class ScalerHandler:
    def __init__(self, train_dataset, scaler_save_path, **kwargs):
        """
        A handler to fit scalers to training data and apply them to evaluation data.
        """
        self.train_dataset = train_dataset
        self.scalers = {}
        self.scaler_save_path = scaler_save_path
        os.makedirs(self.scaler_save_path, exist_ok=True)

    def fit_scalers(self):
        """
        Fit scalers on the training dataset and save them.
        """
        descriptor_feature = self.train_dataset.dataset.descriptor_feature
        metadata_feature_x = self.train_dataset.dataset.metadata_feature_x
        metadata_feature_y = self.train_dataset.dataset.metadata_feature_y

        self.scalers = {
            "scaler_descriptor": StandardScaler().fit(descriptor_feature),
            "scaler_metadata_x": StandardScaler().fit(metadata_feature_x),
            "scaler_metadata_y": StandardScaler().fit(metadata_feature_y),
        }

        # Transform and update the training dataset
        self.train_dataset.dataset.descriptor_feature = self.scalers["scaler_descriptor"].transform(descriptor_feature)
        self.train_dataset.dataset.metadata_feature_x = self.scalers["scaler_metadata_x"].transform(metadata_feature_x)
        self.train_dataset.dataset.metadata_feature_y = self.scalers["scaler_metadata_y"].transform(metadata_feature_y)

        # Save the fitted scalers
        self._save_scalers()

        return self.train_dataset

    def transform_dataset(self, dataset):
        """
        Transform features of a dataset using the fitted scalers.
        """
        if not self.scalers:
            self.scalers = self._load_scalers()

        dataset.dataset.descriptor_feature = self.scalers["scaler_descriptor"].transform(
            dataset.dataset.descriptor_feature)
        dataset.dataset.metadata_feature_x = self.scalers["scaler_metadata_x"].transform(
            dataset.dataset.metadata_feature_x)
        dataset.dataset.metadata_feature_y = self.scalers["scaler_metadata_y"].transform(
            dataset.dataset.metadata_feature_y)

        return dataset

    def _save_scalers(self):
        """
        Save the scalers to the specified path.
        """
        for name, scaler in self.scalers.items():
            dump(scaler, os.path.join(self.scaler_save_path, f'{name}.joblib'))

    def _load_scalers(self):
        """
        Load scalers from the specified path.
        """
        return {
            "scaler_descriptor": load(os.path.join(self.scaler_save_path, 'scaler_descriptor.joblib')),
            "scaler_metadata_x": load(os.path.join(self.scaler_save_path, 'scaler_metadata_x.joblib')),
            "scaler_metadata_y": load(os.path.join(self.scaler_save_path, 'scaler_metadata_y.joblib')),
        }


class CustomDataset(Dataset):
    def __init__(self, config_path: dict, config_prep: PrepConfig, feat_filename: str):
        """User defined Dataset(iterable) object achieving `torch.utils.data.Dataset` abstract class.
        Manage data in various tensor matrices like descriptor_feature, metadata_feature_x, metadata_feature_y, and rt_feature_x, label
        """
        d = feature.DatasetFeatures(config_path, config_prep, feat_filename)
        matrix_dict = d.get_features_labels()
        self.labels = matrix_dict["labels"]
        self.descriptor_feature = matrix_dict["descriptor_feature"]
        self.metadata_feature_x = matrix_dict["metadata_feature_x"]
        self.metadata_feature_y = matrix_dict["metadata_feature_y"]
        self.rt_feature_x = matrix_dict["rt_feature_x"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """concate descriptors, gradients, metadata according to x, y id"""

        input_descriptor_feature = torch.tensor(self.descriptor_feature[idx], dtype=torch.float32)  # shape(851，)
        input_metadata_feature_x = torch.tensor(self.metadata_feature_x[idx], dtype=torch.float32)
        input_metadata_feature_y = torch.tensor(self.metadata_feature_y[idx], dtype=torch.float32)
        input_rt_feature_x = torch.tensor(self.rt_feature_x[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # shape(1，)

        # print("data_generator: CustomDataset 测试|descriptor_feature shape", self.descriptor_feature.shape)
        # print("data_generator: CustomDataset 测试|input_descriptor_feature shape", input_descriptor_feature.shape)
        return input_descriptor_feature, input_metadata_feature_x, input_metadata_feature_y, input_rt_feature_x, label


class DatasetTensor:
    def __init__(self, config_path: dict, config_prep: PrepConfig, config_dl: DLParamConfig, feat_filename: str):
        """manages loading data, scaling data, splitting datasets, and random sampling.
        - loading data and manage scaler at different mode.
        - return iterable item in tensor format.
        - split indices of train CustomDataset into train and evaluate Subset
        - random sampler smaller sets for train and evaluate Subset

        Parameters
        ----------
        config_path:
            dict, the file path of saving dataset, for initial CustomDataset class and set saving path for split dataset.
        config_prep:
            PrepConfig object in config_.py, for initial CustomDataset class
        config_dl:
            DLParamConfig object in config_.py
        feat_filename:
            str, Path-like '.npz' feature file, for initial CustomDataset class.
            It will load defined feature file if it exists, or create feature file with this name.
        """
        self.config_dl = config_dl
        self.dataset = CustomDataset(config_path, config_prep, feat_filename)
        self.split_dataset_path = config_path['random_split_idx_path']
        self.train_indices = None
        self.eval_indices = None
        self.train_dataset, self.eval_dataset = None, None

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def save_split_indices(self, train_dataset, eval_dataset, name):
        # 保存数据集的索引
        path = os.path.join(self.split_dataset_path, name)
        train_indices = train_dataset.indices
        eval_indices = eval_dataset.indices
        torch.save({'train_indices': train_indices, 'eval_indices': eval_indices}, path)

    def load_split_indices(self, name):
        # 加载数据集的索引
        path = os.path.join(self.split_dataset_path, name)
        logger.debug(f"load split indices fom path:{path}")
        indices = torch.load(path)
        self.train_indices = indices['train_indices']
        self.eval_indices = indices['eval_indices']
        train_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)
        eval_dataset = torch.utils.data.Subset(self.dataset, self.eval_indices)
        return train_dataset, eval_dataset

    def random_split(self) -> tuple[Dataset, Dataset]:
        """if train evaluate split dataset exist, load saved dataset. If not exist, conduct spliting and save"""
        train_ratio = self.config_dl.cnn_params["train_ratio"]
        seed = self.config_dl.cnn_params["seed"]
        split_name = f"split_train_{int(train_ratio * 100)}_seed_{seed}.pth"
        full_path = os.path.join(self.split_dataset_path, split_name)
        is_exist = DatasetTensor.dataset_exist(full_path)
        if is_exist:
            self.train_dataset, self.eval_dataset = self.load_split_indices(name=split_name)
        else:
            self.train_dataset, self.eval_dataset = self._random_sampler(train_ratio, seed)
            self.save_split_indices(self.train_dataset, self.eval_dataset, name=split_name)
        return self.train_dataset, self.eval_dataset

    def _random_sampler(self, train_ratio, seed) -> tuple[Dataset, Dataset]:
        train_size = int(train_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, eval_dataset = random_split(self.dataset, [train_size, test_size],
                                                   generator=generator)

        return train_dataset, eval_dataset

    @staticmethod
    def small_set(set, seed, range_num):
        np.random.seed(seed)
        random_indices = np.random.randint(0, len(set), size=range_num).tolist()
        small_dataset = Subset(set, random_indices)
        return small_dataset

    def random_sampler_sd(self, train_size, eval_size, train_seed, eval_seed):
        """randomly sampling small subset according to data size passed"""
        train_dataset_small = DatasetTensor.small_set(self.train_dataset, train_seed, train_size)
        eval_dataset_small = DatasetTensor.small_set(self.eval_dataset, eval_seed, eval_size)
        return train_dataset_small, eval_dataset_small

    @staticmethod
    def dataset_exist(path):
        is_exist = common.file_is_exist(path)
        if is_exist:
            logger.debug(f"Split Dataset already exist.")
            return True
        else:
            logger.debug(f"Split Dataset did not exist.")
            return False

# %%

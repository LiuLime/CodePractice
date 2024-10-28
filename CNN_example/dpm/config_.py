"""
Config parameter

# Example usage
>>>config = PathConfig(mode='train', approach=None)
>>>print("Paths:", config.paths)
"""

import os
import sys

sys.path.append("..")


class PathConfig:
    def __init__(self, mode, approach=None):
        """
        Parameters
        ----------
        mode:
            str, 'train', 'test', 'preds', 'prepr'
        approach:
            str, 'ML', 'DL', None
        """
        self.mode = mode
        self.approach = approach
        self.version='nn_v6'

    def _validate_params(self):
        if self.mode not in ['train', 'test', 'preds', 'prepr']:
            raise ValueError("Mode should be 'train', 'test', 'preds', 'prepr'.")
        if self.approach not in ['ML', 'DL', None]:
            raise ValueError("Approach should be 'ML', 'DL', or None.")

    @property
    def base_path(self):
        # Define general settings that are shared across modes and approaches
        
        os.makedirs(f'dpm/nn/save/{self.version}/', exist_ok=True)
        base_paths = {
            'meta_d_fpath': './collate_data/{mode}_metadata.csv',
            'descriptor_d_fpath': './collate_data/{mode}_descriptor.csv',
            'gradient_d_fpath': './collate_data/{mode}_gradient.json',
            'pool_fpath': './collate_data/{mode}.csv',
            'inputs_ML_fpath': 'collate_data/ML_inputs/inputs_ML.npz',
            'inputs_DL_fpath': 'collate_data/DL_inputs/inputs_DL.npz',
            'inputs_fpath': 'collate_data/inputs/{mode}_inputs.npz',
            'random_split_idx_path': 'collate_data/split_idx/',
            'nn_output_path': f'dpm/nn/save/{self.version}/',
            'nn_model_fpath': f'dpm/nn/save/{self.version}/{self.version}_checkpoint.pt',
            'nn_metric_fpath': f'dpm/nn/save/{self.version}/{self.version}_metric.json',
            'nn_lrfinder_plotpath':f'dpm/nn/save/{self.version}/{self.version}_lrfinder.pdf',
            'scaler_path': "dpm/scalers/"
        }
        # Replace placeholders in base_paths with actual mode values
        return {
            key: value.format(mode=self.mode, approach=self.approach)
            for key, value in base_paths.items()
        }

    def __getitem__(self, item):
        return self.base_path[item]


# preprocessing configaration
class PrepConfig:
    """
    rt_encoding : Dict[str, Any]
        Settings for the retention time encoder.
    meta_encoding : Dict[str, Any]
        Settings for the metadata encoder.
    compound_encoding : Dict[str, Any]
        Settings for the compound descriptor encoder.
    """
    rt_encoding = {
        "num_bits_rt": 27,
        "rt_min": 0,
        "rt_max": 7200
    }


class BaseParamConfig:
    """base config for ML and DL"""

    def __init__(self):
        self.train_ratio = 0.8
        self.seed = 42
        self.lr = 0.0001047
        self.lr_finder = 1e-7
        self.batch_size = 512
        self.num_epochs = 50
        self.world_size = 10
        self.save_every_n_epoch = 5
        self.test = False  # True will arise test set prediction
        self.early_stopping = True  # True will arise early stopping process
        self.patience = 10  # patience steps for early stopping
        self.run_from_ckp = False  # True will arise running from saved checkpoint
        self.run_small_set = False  # True will arise small set sampling
        self.save_dataset_idx = False  # for saving split dataset index
        self.small_train_set_seed = 10  # for small set down sampling
        self.small_eval_set_seed = 20  # for small set down sampling
        self.small_train_size = 1000  # for small set down sampling
        self.small_eval_size = 1000  # for small set down sampling
        self.scaling = True  # for feature sacling


# machine learning parameters
class MLParamConfig(BaseParamConfig):
    @property
    def ML_params(self):
        return {
            'method': ["BRR", "Lasso", "AB", "GB", "RF", "linearSVR", "SVR"],
            'cv': 5,
            'train_test_split_seed': self.seed,
            'subset_minimum_value': 50
        }


# deep learning parameters
class DLParamConfig(BaseParamConfig):
    @property
    def cnn_params(self):
        return {
            'train_ratio': self.train_ratio,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'learning_rate': self.lr,
            'learning_rate_finder': self.lr_finder,
            'scaling': self.scaling,
            'num_epochs': self.num_epochs,
            'save_every_n_epoch': self.save_every_n_epoch,
            'world_size': self.world_size,
            'filters': 30,
            'kernel_size': 3,
            'strides': 1,
            'padding_size': 0,
            'pool_size': 1,
            'pool_strides': 2,
            'num_descriptor_features': 286,
            'num_metadata_x_features': 282,
            'num_metadata_y_features': 282,
            'num_rt_x_features': 27,
            'embedding_dim': 855,
            'num_conv1d_blocks': 5,
            'descriptor_dense_dim': 128,
            'metadata_x_dense_dim': 128,
            'metadata_y_dense_dim': 128,
            'output_dim': 1,
            'test': self.test,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'run_from_ckp': self.run_from_ckp,
            'run_small_set': self.run_small_set,
            'save_dataset_idx': self.save_dataset_idx,
            'small_train_set_seed': self.small_train_set_seed,
            'small_eval_set_seed': self.small_eval_set_seed,
            'small_train_size': self.small_train_size,
            'small_eval_size': self.small_eval_size
        }

    def calc_concat_dim(self, input_dim: int):
        block_structure = {"block1": {"num_conv1d": 2, "num_maxpool": 1},
                           "block2": {"num_conv1d": 2, "num_maxpool": 1},
                        #    "block3": {"num_conv1d": 3, "num_maxpool": 1},
                        #    "block4": {"num_conv1d": 3, "num_maxpool": 1},
                        #    "block5": {"num_conv1d": 3, "num_maxpool": 1},
                           }

        for b, l in block_structure.items():
            num_conv1d = l["num_conv1d"]
            num_max_pool = l["num_maxpool"]
            input_dim = self.calc_conv_dim(input_dim=input_dim, num_conv1d=num_conv1d)
            input_dim = self.calc_maxpool_dim(input_dim, num_maxpool=num_max_pool)
        return input_dim

    def calc_conv_dim(self, input_dim: int, num_conv1d: int):
        if num_conv1d == 0:
            return input_dim
        else:
            input_dim = (input_dim + 2 * self.cnn_params["padding_size"] - self.cnn_params["kernel_size"]) // \
                        self.cnn_params["strides"] + 1
            num_conv1d -= 1
            return self.calc_conv_dim(input_dim, num_conv1d)

    def calc_maxpool_dim(self, input_dim: int, num_maxpool: int):
        if num_maxpool == 0:
            return input_dim
        else:
            input_dim = (input_dim - self.cnn_params["pool_size"]) // self.cnn_params["pool_strides"] + 1
            num_maxpool -= 1
        return self.calc_maxpool_dim(input_dim, num_maxpool)

# c = DLParamConfig().cnn_params
# input_dim = c.calc_concat_dim(input_dim=411)

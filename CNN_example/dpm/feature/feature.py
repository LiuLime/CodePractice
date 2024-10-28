"""load all features datasets"""

import sys

import pandas as pd

from dpm.config_ import PrepConfig
from dpm.feature import entry_utils as eutils
from dpm.utils import common, log
import numpy as np
import numba as nb

sys.path.append("../..")

logger = log.logger()


class DatasetFeatures:
    def __init__(self, config_path: dict, config_prep: PrepConfig, feat_filename: str):
        """ dataset object prepared for machine learning in array format"""
        self.config_path = config_path
        self.rt_encoding = config_prep.rt_encoding  # dict[]
        self.featurePool = eutils.FeatureObject(config_path)
        self.df_pool = self.featurePool.df_pool  # pd.Dataframe
        self.descriptors = self.featurePool.descriptor_pool  # dict[compound_identifier: np.ndarray]
        self.metadata = self.featurePool.metadata_pool  # dict[dataset_identifier: np.ndarray]
        self.gradient = self.featurePool.gradient_pool  # dict[dataset_identifier: np.ndarray]
        self.feat_filename = feat_filename

    def __len__(self):
        return len(self.df_pool)

    def get_features_labels(self):
        """Get features and labels, read saved [matrix].npz if it exist, else will conduct feature encoding from scratch.

        Returns
        --------
        matrix_dict:  {
            'groups': groups,
            'unique_idxs': unique_idxs,
            'descriptor_feature': descriptor_feature,
            'metadata_feature_x': metadata_feature_x,
            'metadata_feature_y': metadata_feature_y,
            'rt_feature_x': rt_feature_x,
            'labels': labels
        }

        """
        is_exist = common.file_is_exist(self.feat_filename)
        if is_exist:
            logger.debug(f"inputs matrix is already exist, loading from {self.feat_filename}")
            matrix_dict = common.read_arrays(self.feat_filename)

        else:
            logger.debug(f"inputs matrix is not exist, start processing")
            matrix_dict = self.generate_feature()
            common.save_arrays(self.feat_filename, **matrix_dict)
        return matrix_dict

    def generate_feature(self):
        """generate features
        - compound descriptor_feature, shape 286
        - metadata_feature_x/y: concatenated from system metadata_feature and gradient_feature
        - rt_feature_x: represented by 27big gray code.
        - labels(y_rts): float.

        Returns
        ---------
        matrix_dict:  {
            'groups': groups, x dataset idx & y dataset idx
            'unique_idxs': unique_idxs, x compound idx & y compound idx
            'descriptor_feature': descriptor_feature,
            'metadata_feature_x': metadata_feature_x,
            'metadata_feature_y': metadata_feature_y,
            'rt_feature_x': rt_feature_x,
            'labels': labels
        }
        """
        x_ids = self.df_pool["x_id"]  # pd.Series
        y_ids = self.df_pool["y_id"]
        x_rts = self.df_pool["x_rt"]
        y_rts = self.df_pool["y_rt"]
        x_systems = self.df_pool["x_dataset_id"]
        y_systems = self.df_pool["y_dataset_id"]
        groups = x_systems + "+" + y_systems
        unique_idxs = x_ids + "+" + y_ids

        # descriptor feature
        descriptor_feature = self._generate_descriptor_feature(x_ids)

        # metadata
        metadata_feature_x = self._generate_metadata_feature(x_systems)
        metadata_feature_y = self._generate_metadata_feature(y_systems)

        # gradient feature
        gradient_feature_x = self._generate_gradient_feature(x_systems)
        gradient_feature_y = self._generate_gradient_feature(y_systems)
        # print("x gradient feature shape", x_gradient_feature.shape)

        # rt feature
        rt_feature_x = self._generate_rt_feature(x_rts)
        y_rt_features = np.stack(y_rts.values)

        logger.debug(f"x_rt_features:{rt_feature_x.shape}, "
                     f"y_rt_features(labels):{y_rt_features.shape}, "
                     f"x_descriptor_feature:{descriptor_feature.shape}, "
                     f"x_metadata_feature:{metadata_feature_x.shape}, "
                     f"x_gradient_feature:{gradient_feature_x.shape}")
        # x_rt_features:(974746, 27), y_rt_features(labels):(974746,), x_descriptor_feature:(974746, 286), x_metadata_feature:(974746, 174), x_gradient_feature:(974746, 108)

        # combine system metadata feature and gradient feature
        metadata_feature_x = np.hstack([metadata_feature_x, gradient_feature_x])
        metadata_feature_y = np.hstack([metadata_feature_y, gradient_feature_y])

        labels = np.expand_dims(y_rt_features, axis=1)
        matrix_dict = {
            'groups': groups,
            'unique_idxs': unique_idxs,
            'descriptor_feature': descriptor_feature,
            'metadata_feature_x': metadata_feature_x,
            'metadata_feature_y': metadata_feature_y,
            'rt_feature_x': rt_feature_x,
            'labels': labels
        }
        return matrix_dict

    def _generate_rt_feature(self, rts: pd.Series) -> np.ndarray:
        """coding retention time into 27bit gray code"""
        return np.stack(rts.apply(binary_encode, args=(self.rt_encoding["rt_min"],
                                                       self.rt_encoding["rt_max"],
                                                       self.rt_encoding["num_bits_rt"]
                                                       )
                                  )
                        .values)

    def _generate_descriptor_feature(self, descriptor_ids: pd.Series) -> np.ndarray:
        return np.stack(descriptor_ids.map(self.descriptors).values)

    def _generate_metadata_feature(self, dataset_ids: pd.Series) -> np.ndarray:
        return np.stack(dataset_ids.map(self.metadata).values)

    def _generate_gradient_feature(self, dataset_ids):
        return np.stack(dataset_ids.map(self.gradient).values)


def _gray_code(value: int, num_bits: int) -> np.array:
    # referring bittremieux-Gleams
    # https://github.com/bittremieux/GLEAMS/blob/13ebc74490399b69ca13c9f182d64ad6a09c67c1/gleams/feature/encoder.py#L295
    """
    Return the Gray code for a given integer, given the number of bits to use
    for the encoding.

    Parameters
    ----------
    value : int
        The integer value to be converted to Gray code.
    num_bits : int
        The number of bits of the encoding. No checking is done to ensure a
        sufficient number of bits to store the encoding is specified.

    Returns
    -------
    ss.csr_matrix
        A sparse array of individual bit values as floats.
    """
    # Gray encoding: https://stackoverflow.com/a/38745459
    return np.array(list(f'{value ^ (value >> 1):0{num_bits}b}'),
                    dtype=np.float32)


@nb.njit
def _get_bin_index(value: float, min_value: float, max_value: float,
                   num_bits: int) -> int:
    # referring bittremieux-Gleams
    # https://github.com/bittremieux/GLEAMS/blob/13ebc74490399b69ca13c9f182d64ad6a09c67c1/gleams/feature/encoder.py#L319
    """
    Get the index of the given value between a minimum and maximum value given
    a specified number of bits.

    Parameters
    ----------
    value : float
        The value to be converted to a bin index.
    min_value : float
        The minimum possible value.
    max_value : float
        The maximum possible value.
    num_bits : int
        The number of bits of the encoding.

    Returns
    -------
    int
        The integer bin index of the value between the given minimum and
        maximum value using the specified number of bits.
    """
    # Divide the value range into equal intervals
    # and find the value's integer index.
    num_bins = 2 ** num_bits
    bin_size = (max_value - min_value) / num_bins
    bin_index = int((value - min_value) / bin_size)
    # Clip to min/max.
    bin_index = max(0, min(num_bins - 1, bin_index))
    return bin_index


def binary_encode(value: float, min_value: float, max_value: float,
                  num_bits: int) -> np.array:
    # referring bittremieux-Gleams
    # https://github.com/bittremieux/GLEAMS/blob/13ebc74490399b69ca13c9f182d64ad6a09c67c1/gleams/feature/encoder.py#L352
    """
    Return the Gray code for a given value, given the number of bits to use
    for the encoding. The given number of bits equally spans the range between
    the given minimum and maximum value.
    If the given value is not within the interval given by the minimum and
    maximum value it will be clipped to either extremum.

    Parameters
    ----------
    value : float
        The value to be converted to Gray code.
    min_value : float
        The minimum possible value.
    max_value : float
        The maximum possible value.
    num_bits : int
        The number of bits of the encoding.

    Returns
    -------
    np.array
        A array of individual bit values as floats.
    """
    bin_index = _get_bin_index(value, min_value, max_value, num_bits)
    return _gray_code(bin_index, num_bits)

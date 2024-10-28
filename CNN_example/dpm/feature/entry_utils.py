""" create MSEntry class, creating instance will including dataset id, compound id, compound descriptors, rt,
system metainformation and system gradient

@ Liu Yuting | Niigata university 2024-9-13"""
import functools
import json
from typing import Hashable

import numpy as np
import pandas as pd
from numpy import ndarray

from dpm.config_ import PathConfig
from dpm.utils import common


# configpath = config.path()

#
# class EntriesObject:
#     def __init__(self, config_path):
#
#     @functools.lru_cache(maxsize=1)
#     def _load_data(self):
#         self.pool_path = config_path["data_pool_fpath"]
#         self.descriptors_path = config_path["descriptor_d_fpath"]
#         self.gradient_path = config_path["gradient_d_fpath"]
#         self.metadata_path = config_path["meta_d_fpath"]
#         train_dataset = common.read_csv(self.pool_path, datatype={"x_id": "object",
#                                                                   "y_id": "object",
#                                                                   "x_dataset_id": "str",
#                                                                   "y_dataset_id": "str"})

class FeatureObject:
    def __init__(self, config_path: dict):
        self.pool_path = config_path["pool_fpath"]
        self.descriptors_path = config_path["descriptor_d_fpath"]
        self.gradient_path = config_path["gradient_d_fpath"]
        self.metadata_path = config_path["meta_d_fpath"]

    @functools.lru_cache(maxsize=1)
    def _load_df(self) -> pd.DataFrame:

        train_dataset = common.read_csv(self.pool_path, datatype={"x_id": "object",
                                                                  "y_id": "object",
                                                                  "x_dataset_id": "str",
                                                                  "y_dataset_id": "str"})
        return train_dataset

    @property
    def df_pool(self):
        return self._load_df()

    @functools.lru_cache(maxsize=1)
    def _map_rt(self) -> dict[Hashable, float]:
        df_pool = self.df_pool
        x_rt_dict = df_pool.set_index("x_id")["x_rt"].to_dict()
        y_rt_dict = df_pool.set_index("y_id")["y_rt"].to_dict()
        return {**x_rt_dict, **y_rt_dict}

    @property
    def rt_map(self) -> dict[Hashable, float]:
        return self._map_rt()

    @functools.lru_cache(maxsize=1)
    def _map_superclass(self) -> dict[Hashable, str]:
        df_pool = self.df_pool
        superclass_dict = df_pool.set_index("x_id")["x_superclass"].to_dict()
        return superclass_dict

    @property
    def superclass_map(self) -> dict[Hashable, str]:
        return self._map_superclass()

    @functools.lru_cache(maxsize=1)
    def _map_descriptors(self) -> dict[Hashable, ndarray]:
        descriptors = common.read_csv(self.descriptors_path, datatype={"id": "str"}).fillna(value=0)
        descriptors = descriptors.set_index(keys="id")
        descriptors_dict = {index: row.to_numpy(dtype=np.float32) for index, row in descriptors.iterrows()}
        return descriptors_dict

    @property
    def descriptor_pool(self) -> dict[Hashable, ndarray]:
        return self._map_descriptors()

    @functools.lru_cache(maxsize=1)
    def _map_gradients(self) -> json:
        gradients = common.read_json(self.gradient_path)
        for key, value in gradients.items():
            value = FeatureObject.gradient_padding_flatten(value)
            gradients[key] = value
        return gradients

    @property
    def gradient_pool(self):
        return self._map_gradients()

    @functools.lru_cache(maxsize=1)
    def _map_metadata(self) -> dict[Hashable, ndarray]:
        metadata = common.read_csv(self.metadata_path, datatype={"id": "str"}).fillna(value=0)
        metadata.loc[:, "id"] = metadata["id"].apply(FeatureObject.format_id)
        metadata = metadata.map(FeatureObject.code_unit)
        metadata = metadata.drop(columns=["column.name", "column.usp.code"]).set_index(keys="id")
        metadata_dict = {index: row.to_numpy(dtype=np.float32) for index, row in metadata.iterrows()}
        return metadata_dict

    @property
    def metadata_pool(self):
        return self._map_metadata()

    @staticmethod
    def format_id(num):
        return f'{num:04}'

    @staticmethod
    def code_unit(x):
        code_unit_dict = {"%": 1,
                          "mM": 2,
                          "µM": 3}
        try:
            return code_unit_dict[x]
        except KeyError:
            return x

    @staticmethod
    def gradient_padding_flatten(gradient) -> np.ndarray:
        """Padding to (1, 108)"""
        gradient_array = np.array(gradient, dtype=np.float32).flatten()
        length = gradient_array.shape[0]
        diff = 108 - length

        # 如果需要填充
        if diff > 0:
            gradient_array_padding = np.pad(gradient_array, (0, diff), mode='constant', constant_values=0)
        else:
            gradient_array_padding = gradient_array

        # 增加一个维度，变成 (1, 108)
        # gradient_array_padding = np.expand_dims(gradient_array_padding, axis=0)

        return gradient_array_padding


class MsmsEntry:
    def __init__(self,
                 config_path: PathConfig,
                 dataset_identifier: str,
                 compound_identifier: str | None = None,
                 ):
        self.dataset_identifier = dataset_identifier
        self.compound_identifier = compound_identifier
        self.features = FeatureObject(config_path)

    def __getstate__(self):
        return {

            "dataset_identifier": self.dataset_identifier,
            "compound_superclass": self.superclass_,
            "compound_descriptor": self.compound_descriptor,
            "system_gradient": self.system_gradient,
            "system_parameter": self.system_parameter
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def retention_time(self) -> float:
        return self.features.rt_map[self.compound_identifier]

    @property
    def compound_descriptor(self) -> np.ndarray:
        return self.features.descriptor_pool[self.compound_identifier]

    @property
    def system_parameter(self) -> np.ndarray:
        return self.features.metadata_pool[self.dataset_identifier]

    @property
    def system_gradient(self) -> np.ndarray:
        return self.features.gradient_pool[self.dataset_identifier]

    @property
    def superclass_(self) -> str:
        superclass_ = self.features.superclass_map[self.compound_identifier]
        return superclass_

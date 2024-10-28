"""
Refer the code from github repository: bittremieux/GLEAMS
https://github.com/bittremieux/GLEAMS/blob/13ebc74490399b69ca13c9f182d64ad6a09c67c1/gleams/feature/encoder.py#L18

@ Liu Yuting|Niigata university 2024-9-13
"""

import abc
import itertools
from typing import List
import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),
                                              os.pardir, os.pardir)))
import numba as nb
import numpy as np
import scipy.sparse as ss
from entry_utils import MsmsSpectrum, MsmsSystem

import numpy as np


class EntryEncoder(metaclass=abc.ABCMeta):
    """
    Abstract superclass for system encoders.
    """

    feature_names = None

    def __init__(self):
        pass

    @abc.abstractmethod
    def encode(self, entry: MsmsSystem) -> ss.csr_matrix:
        """
        Encode the given Entry information.

        Parameters
        ----------
        information : np.array, information array collated by RepoRT repository.
            The system parameter or spectrum information to be encoded.

        Returns
        -------
        ss.csr_matrix
            Encoded system features and spectrum features.
        """
        pass


class MetaEnconder(EntryEncoder):
    """
    Represents a system as metadata features and gradient features: 1-D matrix.
    """

    def __init__(self, ):
        """
        Instantiate a MetaEncoder.

        """
        super().__init__()
        self.feature_names = [f'meta_1']

    def encode(self, entry: MsmsSystem) -> ss.csr_matrix:
        """
        Encode the system meta information as system parameter and gradient as csr_matrix format

        Returns
        -------
        ss.csr_matrix
        """
        return ss.hstack([ss.csr_matrix(entry.system_parameter), ss.csr_matrix(entry.system_gradient)])


class RTEncoder(EntryEncoder):
    """
    Represents a retention time as gray encoding

    """

    def __init__(self, num_bits_rt: int,
                 rt_min: float,
                 rt_max: float,
                 ):
        """
        Instantiate a RTEncoder.

        Parameters
        ----------
        num_bits_rt : int
            The number of bits to use to encode the retention time.
        rt_min : float
            The minimum value between which to scale the precursor m/z.
        rt_max : float
            The maximum value between which to scale the precursor m/z.

        """
        super().__init__()

        self.num_bits_rt = num_bits_rt
        self.rt_min = rt_min
        self.rt_max = rt_max
        self.feature_names = [f'rt_{i}' for i in range(self.num_bits_rt)]

    def encode(self, entry: MsmsSystem) -> ss.csr_matrix:
        """
        Encode the retention time into 27 bit gray code.

        Parameters
        ----------
        rt : float
            The retention time to be encoded.

        Returns
        -------
        ss.csr_matrix
        """
        rt = entry.retention_time
        gray_code_rt = binary_encode(
            rt, self.rt_min, self.rt_max, self.num_bits_rt)
        return gray_code_rt


class CompoundEnconder(EntryEncoder):
    """
    Represents a compound as descriptor features: 1-D matrix.
    """

    def __init__(self, ):
        """
        Instantiate a CompoundEncoder.

        Parameters
        ----------


        """
        super().__init__()

        self.feature_names = [f'descriptor_1']

    def encode(self, entry: MsmsSystem) -> ss.csr_matrix:
        """
        Encode the compound descriptor into parse matrix

        Parameters
        ----------
        entry : MSEntry
            The entry to be encoded.

        Returns
        -------
        ss.csr_matrix
        """
        return ss.csr_matrix(entry.compound_descriptor)


class MultipleEncoder(EntryEncoder):
    """
    Combines multiple child encoders.
    """

    def __init__(self, encoders: List[EntryEncoder]):
        """
        Instantiate a MultipleEncoder with the given child encoders.

        Parameters
        ----------
        encoders : List[SpectrumEncoder]
            The child encoders to do the actual spectrum encoding.
        """
        super().__init__()

        self.encoders = encoders

        self.feature_names = list(itertools.chain.from_iterable(
            [enc.feature_names for enc in self.encoders]))

    def encode(self, entry: MsmsSystem) -> ss.csr_matrix:
        """
        Encode the given spectrum using the child encoders.

        Parameters
        ----------
        spec : MsmsSpectrum
            The spectrum to be encoded.

        Returns
        -------
        ss.csr_matrix
            Concatenated spectrum features produced by all child encoders.
        """
        return ss.hstack([enc.encode(entry) for enc in self.encoders])


def _gray_code(value: int, num_bits: int) -> ss.csr_matrix:
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
    return ss.csr_matrix(list(f'{value ^ (value >> 1):0{num_bits}b}'),
                         dtype=np.float32)


@nb.njit
def _get_bin_index(value: float, min_value: float, max_value: float,
                   num_bits: int) -> int:
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
                  num_bits: int) -> ss.csr_matrix:
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
    ss.csr_matrix
        A sparse array of individual bit values as floats.
    """
    bin_index = _get_bin_index(value, min_value, max_value, num_bits)
    return _gray_code(bin_index, num_bits)

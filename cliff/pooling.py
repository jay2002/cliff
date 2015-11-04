"""
    This module is used to do feature pooling
"""

# pylint: disable=E1101

import numpy as np


def max_pooling(features):
    """ Max pooling
    :param features: local features before pooling
    :type features: list, each element is a n-by-m ndarray, where n is the
        number of features, m is feature dimension
    :return: features after max pooling
    :rtype: list, each element is a n-by-m ndarray
    """

    return [np.max(i, axis=0) for i in features]


def mean_pooling(features):
    """ Mean pooling
    :param features: local features before pooling
    :type features: list, each element is a n-by-m ndarray, where n is the
        number of features, m is feature dimension
    :return: features after mean pooling
    :rtype: list, each element is a n-by-m ndarray
    """

    return [np.mean(i, axis=0) for i in features]


def top_k_mean_pooling(features, k):
    """ Top k mean pooling
    :param features: local features before pooling
    :type features: list, each element is a n-by-m ndarray, where n is the
        number of features, m is feature dimension
    :param k: for each dimension, choose largest k values
    :type k: integer
    :return: features after top k mean pooling
    :rtype: list, each element is a n-by-m ndarray
    """

    return [np.mean(np.sort(i, axis=0)[-k:, :], axis=0) for i in features]

"""
    This module includes normalization functions
"""

# pylint: disable=E1101

import numpy as np


def l2_norm(features):
    """ l2 normalization
    :param features: features before normalization
    :type features: list
    :return: features after normalization
    :rtype: list
    """

    transformed_feature = []
    for i in features:
        tmp = i.copy()
        for j in range(tmp.shape[0]):
            tmp[j] = tmp[j] / np.sqrt(np.dot(tmp[j], tmp[j]))
        transformed_feature.append(tmp)
    return transformed_feature

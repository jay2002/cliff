"""
    This module includes activation functions.
"""


def relu(features):
    """ Rectified Linear Unit
    :param features: features before activation
    :type features: list
    :return: features after activation
    :rtype: list
    """

    transformed_feature = []
    for i in features:
        tmp = i.copy()
        tmp[tmp < 0] = 0
        transformed_feature.append(tmp)
    return transformed_feature

"""
    This module includes normalization functions
"""


def l2(features):
    """ l2 normalization
    :param features: features before normalization
    :type features: list
    :return: features after normalization
    :rtype: list
    """

    transformed_feature = []
    for i in features:
        tmp = i.copy()
        tmp = tmp / np.sqrt(np.dot(tmp, tmp))
        transformed_feature.append(tmp)
    return transformed_feature

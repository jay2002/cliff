"""
    This module is used to apply Principal Component Analysis on features
"""

# pylint: disable=E1101

import numpy as np
import sklearn.decomposition


def fit(features, n_dimensions, whiten=False):
    """ train PCA
    :param features: features used to train PCA
    :type features: list. Each element is a k-by-m ndarray, where k is the
        number of local features and varies from image to images, and m is
        feature dimension.
    :param n_dimentions: feature dimension after PCA
    :type n_dimensions: integer
    :param whiten: whether to do whitening
    :type whiten: boolean
    :return: trained pca
    :rtype: sklearn.decomposition.PCA
    """

    pca = sklearn.decomposition.PCA(n_components=n_dimensions, whiten=whiten)
    pca.fit(np.vstack(features))
    return pca


def transform(features, pca):
    """ use trained PCA to transform data
    :param features: features to be transformed
    :type features: list. Each element is a k-by-m ndarray, where k is the
        number of local features and varies from image to images, and m is
        feature dimension.
    :param pca: trained PCA
    :type pca: sklearn.decomposition.PCA
    :return: transformed features
    :rtype: list
    """

    return [pca.transform(i) for i in features]

"""
    This module is used to aggregate features
"""

# pylint: disable=E1101

import numpy as np
import sklearn.cluster


def vlad_fit(features, n_centers):
    """ train VLAD
        :param features: features used to train VLAD
        :type features: list
        :param n_centers: number of centers
        :type n_centers: int
        :return: trained VLAD
        :rtype: sklearn.cluser.k_means_.KMeans
    """

    kmeans = sklearn.cluster.KMeans(n_clusters=n_centers)
    kmeans.fit(np.vstack(features))
    print type(kmeans)
    return kmeans


def vlad_transform(features, kmeans):
    """ use trained VLAD to transform data
        :param features: features to be transformed
        :type features: list
        :param kmeans: trained VLAD
        :type kmeans: sklearn.cluser.k_means_.KMeans
        :return: transformed features
        :rtype: list
    """

    transformed_features = []
    for i, feat in enumerate(features):
        print 'No.', i
        transformed_feat = np.zeros(kmeans.cluster_centers_.shape)
        if feat is not None:
            labels = kmeans.predict(feat)
            for j, label in enumerate(labels):
                transformed_feat[label] = \
                    transformed_feat[label] + \
                    feat[j] - kmeans.cluster_centers_[label]
            for j in range(transformed_feat.shape[0]):
                norm = np.linalg.norm(transformed_feat[j])
                if norm > 0:
                    transformed_feat[j] = transformed_feat[j] / norm
                else:
                    transformed_feat[j] = 0 * transformed_feat[j]
        transformed_features.append(transformed_feat)
    return transformed_features

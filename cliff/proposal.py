"""
    This module is used to generate object proposals
"""

# pylint: disable=E1101
# pylint: disable=F0401

import os
import numpy as np


def selective_search(images, rcnn=False, batch_size=128):
    """
        Generate object proposals using selective search algorithm
        :param images: image names
        :type images: list, each element is a string
        :param rcnn: False for a few quick proposals, True for R-CNN
            configuration for more coverage.
        :type rcnn: boolean
        :param batch_size: number of images sent to the outer program each time
        :type batch_size: integer
        :return: proposals for each image, each proposal is located by two
            points' coordinates
        :rtype: list, each element is a k-by-4 ndarray, k is the number of
            proposals and varies among images
    """

    import selective_search_ijcv_with_python.selective_search as ss

    n_images = len(images)
    n_batches = int(np.ceil(1.0 * n_images / batch_size))
    proposals = []
    real_names = [os.path.realpath(i) for i in images]
    for i in range(n_batches):
        print 'Batch: ', i, '/', n_batches
        print 'Image: ', i * batch_size, '/', n_images
        batch = real_names[i * batch_size:min((i + 1) * batch_size, n_images)]
        if rcnn:
            tmp = ss.get_windows(batch, 'selective_search_rcnn')
        else:
            tmp = ss.get_windows(batch)
        proposals = proposals + tmp
    return proposals

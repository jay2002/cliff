"""
    This module is used to generate object proposals
"""

# pylint: disable=F0401

import os


def selective_search(images):
    """
        Generate object proposals using selective search algorithm
        :param images: image names
        :type images: list, each element is a string
        :return: proposals for each image
        :rtype: list, each element is a k-by-4 ndarray, k is the number of
            proposals
    """

    import selective_search_ijcv_with_python.selective_search as ss
    real_names = [os.path.realpath(i) for i in images]
    proposals = ss.get_windows(real_names)
    print proposals
    return proposals

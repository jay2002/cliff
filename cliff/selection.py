"""
    This module is used to select features or proposals
"""


def select_preceding(features, k):
    """ select preceding k features or proposals for each image
    :param k: preceding k features or proposals for each image are selected
    :type k: integer
    :return: selected features or proposals
    :rtype: list. Each element is a k'-by-m ndarray, where m is feature
        dimension or 4 for proposals. If there are enough features or proposals
        for selection, then k' = k, else all features or proposals are
        selected.
    """

    return [i[:k] for i in features]

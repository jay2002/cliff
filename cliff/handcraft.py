"""
    This module is used to extract handcraft features
"""

# pylint: disable=E1101

import os
import cv2


def sift(images, n_features=None):
    """ Extract SIFT features
        :param images: image names. The function extracts features for each
            image.
        :type images: list
        :param n_features: number of features. If n_features == None, all
            features will be extracted.
        :type n_features: int or None
    """

    if cv2.__version__.startswith('2.'):
        if n_features is None:
            detector = cv2.SIFT()
        else:
            detector = cv2.SIFT(n_features)
    elif cv2.__version__.startswith('3.'):
        if n_features is None:
            detector = cv2.xfeatures2d.SIFT_create()
        else:
            detector = cv2.xfeatures2d.SIFT_create(n_features)
    else:
        print "Unknown OpenCV version."

    features = []
    for i, image_name in enumerate(images):
        print 'No.', i
        print image_name
        real_name = os.path.realpath(image_name)
        image = cv2.imread(real_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, feat = detector.detectAndCompute(gray, None)
        features.append(feat)

    return features

"""
    This module is used to extract convolutional neural network features
"""
# pylint: disable=E1101
# pylint: disable=F0401
# pylint: disable=R0912
# pylint: disable=R0913
# pylint: disable=R0914

import os
import numpy as np
import caffe
import cliff.config


def caffe(images, proposals, model, layer, device=0, batch_size=128):
    """ Extract CNN features based on Caffe models
    :param images: image names. The function extracts features for each
        image.
    :type images: list, each element is a string
    :param proposals: proposals or None. The function extracts features for
        each proposal of each image. If it is None, the function will extract
        global features.
    :type proposals: None, or list, each element is a n-by-4 ndarray, n is
        number of proposals
    :param model: the CNN model name used to extract features
    :type model: string
    :param layer: layer in the CNN model. The function will extract features
        from that layer
    :type layer: string
    :param ravel: whether to ravel the features to a vector
    :type ravel: boolean
    :param device: gpu device number. If cpu is used, it should be a negtive
        number.
    :param batch_size: number of features extracted at one pass
    :type batch_size: integer
    :return: image features
    :rtype: list, each element is a n-by-m ndarray, m is feature dimension
    """

    if device < 0:
        caffe.set_cpu_mode()
    else:
        caffe.set_device(device)
        caffe.set_mode_gpu()

    caffe_root = cliff.config.Config().caffe_root

    if model == 'CaffeNet':
        net = caffe.Net(os.path.join(caffe_root, 'models',
                                     'bvlc_reference_caffenet',
                                     'deploy.prototxt'),
                        os.path.join(caffe_root, 'models',
                                     'bvlc_reference_caffenet',
                                     'bvlc_reference_caffenet.caffemodel'),
                        caffe.TEST)
        net.blobs['data'].reshape(batch_size, 3, 227, 227)
        transformer = caffe.io.Transformer(
            {'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data',
                             np.load(
                                 os.path.join(caffe_root, 'python',
                                              'caffe', 'imagenet',
                                              'ilsvrc_2012_mean.npy'))
                             .mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
    else:
        print 'Unknown CNN model.'
        return None

    feat = []
    n_features = 0
    for i, image_name in enumerate(images):
        print 'No.', i
        print image_name
        real_name = os.path.realpath(image_name)
        image = caffe.io.load_image(real_name)
        if proposals is None:
            windows = [[0, 0, image.shape[0], image.shape[1]]]
        else:
            windows = proposals[i]
        for window in windows:
            net.blobs['data'].data[...] = transformer.preprocess(
                'data', image[window[0]:window[2], window[1]:window[3]])
            n_features = n_features + 1
            if n_features == batch_size:
                n_features = 0
                net.forward()
                for j in range(batch_size):
                    feat.append(net.blobs[layer].data[j].ravel().copy())
    if n_features > 0:
        net.forward()
        for j in range(n_features):
            feat.append(net.blobs[layer].data[j].ravel().copy())

    feat = np.vstack(feat)

    if proposals is None:
        features = list(feat)
    else:
        features = []
        index = 0
        for i in range(len(images)):
            features.append(feat[index:index + proposals[i].shape[0]])
            index = index + proposals[i].shape[0]

    return features

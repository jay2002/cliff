"""
    This module offers a brief io interface
"""

import cPickle


def read_list(filename):
    """ Read list file
    :param filename: list filename
    :type: string
    :return: content read from the file
    :rtype: list
    """

    list_file = open(filename, 'r')
    content = [i.strip() for i in list_file.readlines()]
    list_file.close()
    return content


def read_pickle(filename):
    """ Read pickle file
    :param filename: pickle filename
    :type filename: string
    :return: content read from the file
    """

    pickle_file = open(filename, 'rb')
    content = cPickle.load(pickle_file)
    pickle_file.close()
    return content


def write_pickle(content, filename):
    """ Write pickle file
    :param content: content to write
    :param filename: pickle filename
    :type filename: string
    """
    pickle_file = open(filename, 'wb')
    cPickle.dump(content, pickle_file)
    pickle_file.close()

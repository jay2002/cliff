"""
    Setup configuration for Cliff.
"""

# pylint: disable=too-few-public-methods

import os
import sys
import ConfigParser


class Config(object):
    """ Setup configuration for Cliff
    """

    __path_caffe_python = ""
    __path_selective_search = ""

    def __init__(self):
        """ Read configurations from "cliff.conf"
        """

        lib_path = os.path.dirname(os.path.realpath(__file__))
        config_parser = ConfigParser.ConfigParser()
        config_parser.read(os.path.join(lib_path, 'cliff.conf'))

        self.__path_caffe_python = os.path.join(
            config_parser.get('path', 'caffe_python'))
        if self.__path_caffe_python not in sys.path:
            sys.path.append(self.__path_caffe_python)

        self.__path_selective_search = os.path.join(
            config_parser.get('path', 'selective_search'))
        if self.__path_selective_search not in sys.path:
            sys.path.append(self.__path_selective_search)

    @property
    def caffe_root(self):
        """ Get Caffe root
            return: Caffe root
            rtype: string
        """

        caffe_root = os.path.dirname(self.__path_caffe_python)
        return caffe_root

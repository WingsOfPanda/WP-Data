from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, random, os, shutil
import hashlib

from wputils.utils.fops import fsrh, fflt
from wputils.utils.utils import pe, pt


def fstt(fp, clist):
    """
    check quntatity of data, spliting the image file, the label file, and check the health of dataset
    :param clist:
    :param fp:
    :return:
    """

    flst = fflt(fp)
    csrh, cnon = [], []
    for f in [fsrh(f, clist) for f in flst]:
        if None in f:
            cnon.append(f)
        else:
            csrh.append(f)

    return csrh, cnon


def htck(fp):
    """
    file health and duplication check
    :param fp:
    :return:
    """

    return None


def tvsp(flst):
    random.seed(666)
    random.shuffle(flst)
    spt = int(len(flst) * 0.8)
    return flst[:spt], flst[spt::]


def gmd5(file):
    if os.path.isfile(file):
        with open(file, 'rb') as fp:
            data = fp.read(4096)
            td = bytearray(data)
            td[9] = 0x03
            data = bytes(td)
            result = hashlib.md5(data)
        return result.hexdigest()
    else:
        return None

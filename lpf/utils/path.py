import os.path as osp
import lpf


def get_module_dpath(dname):
    droot_pkg = osp.dirname(lpf.__file__)
    dpath = osp.join(droot_pkg, dname)
    return dpath
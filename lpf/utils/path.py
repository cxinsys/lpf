import os.path as osp
import lpf


def get_module_dpath(dname):
    droot_pkg = osp.dirname(lpf.__file__)
    dpath = osp.join(droot_pkg, dname)
    return dpath


def get_template_fpath(ladybird):
    dpath_data = osp.join(get_module_dpath("data"), ladybird)
    dpath_template = osp.join(dpath_data, "template")
    fpath_template = osp.join(dpath_template, "ladybird.png")
    return fpath_template


def get_mask_fpath(ladybird):
    dpath_data = osp.join(get_module_dpath("data"), ladybird)
    dpath_template = osp.join(dpath_data, "template")
    fpath_mask = osp.join(dpath_template, "mask.png")
    return fpath_mask

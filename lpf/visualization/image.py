import os
import os.path as osp

import numpy as np
from PIL import Image


def add_margins(img, margin_top, margin_right, margin_bottom, margin_left, bg_color):
    width, height = img.size
    new_width = width + margin_right + margin_left
    new_height = height + margin_top + margin_bottom
    img_new = Image.new(img.mode, (new_width, new_height), bg_color)
    img_new.paste(img, (margin_left, margin_top))
    return img_new


def merge_images(*rows, bg_color=(0, 0, 0, 0), alignment=None):

    if not alignment:
        alignment = (1.0, 1.0)

    rows = [[img.convert('RGBA') for img in row] for row in rows]
    heights = [max(img.height for img in row) for row in rows]
    widths = [max(img.width for img in column) for column in zip(*rows)]

    canvas = Image.new(
        'RGBA',
        size=(sum(widths), sum(heights)),
        color=bg_color
    )

    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            y = sum(heights[:i]) + int((heights[i] - img.height) * alignment[1])
            x = sum(widths[:j]) + int((widths[j] - img.width) * alignment[0])
            canvas.paste(img, (x, y))
        # end of for
    # end of for

    return canvas


def merge_timeseries(dpath_input,
                     dpath_output,
                     n_cols,
                     infile_header="ladybird",
                     outfile_header="frame",
                     bg_color="white",
                     ratio_resize=0.5,
                     margin_top=10,
                     margin_right=10,
                     margin_bottom=10,
                     margin_left=10,
                     verbose=0):

    """Merge multiple ladybird images over time.
    """

    if not osp.isdir(dpath_input):
        raise NotADirectoryError("%s doest not exists."%(dpath_input))

    os.makedirs(dpath_output, exist_ok=True)

    dirs_model = []
    for dname in os.listdir(dpath_input):
        dpath = osp.join(dpath_input, dname)
        if not dname.startswith("model_"):
            continue
        else:
            if not osp.isdir(dpath):
                continue
        # if-else

        dirs_model.append(dpath)

    # end of for

    if len(dirs_model) == 0:
        raise NotADirectoryError('There is no directory that starts with "model".')

    fnames = []
    dpath_model = dirs_model[0]
    for entity in os.listdir(dpath_model):
        if not entity.starstwith(infile_header):
            continue

        fpath = osp.join(dpath_model, entity)
        if not osp.isfile(fpath):
            continue

        fnames.append(entity)
    # end of for

    fnames.sort()

    # Generate images.
    n_models = len(dirs_model)
    n_iters = len(fnames)
    n_pad_zeros = int(np.floor(np.log10(n_iters))) + 1
    fstr_outfile = "{}_%0{}d.png".format(outfile_header, n_pad_zeros)

    for i, fname in enumerate(fnames):

        images = [[]]
        cnt_rows = 0

        for dpath in dirs_model:
            fpath = osp.join(dpath, fname)

            img = Image.open(fpath)
            img = img.resize((int(ratio_resize * img.width), int(ratio_resize * img.height)))
            img = add_margins(img,
                              margin_top=margin_top,
                              margin_right=margin_right,
                              margin_bottom=margin_bottom,
                              magin_left=margin_left,
                              bg_color=bg_color)

            images[cnt_rows].append(img)

            if verbose > 0:
                print("[INPUT IMAGE]", fpath)

            if len(images[cnt_rows]) >= n_cols:
                cnt_rows += 1
                if n_models > (cnt_rows * n_cols):
                    images.append([])
        # end of for

        # images.pop()

        img_output = merge_images(*images, bg_color=bg_color, alignment=(1, 1))
        img_output.save(osp.join(dpath_output, fstr_outfile%(i)))

    # end of for

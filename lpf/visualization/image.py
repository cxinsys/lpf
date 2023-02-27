import os
import os.path as osp

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def add_margins(img, margin_top, margin_right, margin_bottom, margin_left, bg_color):
    width, height = img.size
    new_width = width + margin_right + margin_left
    new_height = height + margin_top + margin_bottom
    img_new = Image.new(img.mode, (new_width, new_height), bg_color)
    img_new.paste(img, (margin_left, margin_top))
    return img_new


def merge_image_rows(*rows, bg_color=(0, 0, 0, 0), alignment=None):

    if not alignment:
        alignment = (1.0, 1.0)

    # rows = [[img.convert('RGBA') for img in row] for row in rows]  
    
    n_rows = len(rows)
    n_cols = len(rows[0])
    
    
    arr2d_img = []
    
    # The max height of a row
    arr2d_height = np.zeros((n_rows, n_cols), dtype=np.int32)  
    
    # The max width of a column
    arr2d_width = np.zeros((n_rows, n_cols), dtype=np.int32)   
    
    for i, row in enumerate(rows):
        arr2d_img.append([])
        for j, img in enumerate(row):
            img = img.convert("RGBA")
            arr2d_img[i].append(img)
            arr2d_width[i][j] = img.width
            arr2d_height[i][j] = img.height
    
    
    # heights = [max(img.height for img in row) for row in rows]
    # widths = [max(img.width for img in column) for column in zip(*rows)]
    
    heights = arr2d_height.max(axis=1)
    widths = arr2d_width.max(axis=0)

    canvas = Image.new(
        'RGBA',
        size=(np.sum(widths), np.sum(heights)),
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


def merge_multiple(imgs=None,
                   dpath_input=None,
                   indir_header="model",
                   infile_header="ladybird",
                   n_cols=2,
                   bg_color="white",
                   ratio_resize=1.0,
                   margin_top=10,
                   margin_right=10,
                   margin_bottom=10,
                   margin_left=10,
                   text_format=None,
                   text_color=None,
                   font=None,
                   font_size=None,
                   text_margin_ratio=None,
                   verbose=0):

    """Merge the images of multiple models.
    """

    if not imgs and not dpath_input:
        raise ValueError("Either imgs or dpath_input should be given.")

    if dpath_input:
        if not osp.isdir(dpath_input):
            raise NotADirectoryError("%s doest not exists." % (dpath_input))

        dpaths = []
        for entity in os.listdir(dpath_input):
            if not entity.startswith(indir_header):
                continue

            dpath = osp.join(dpath_input, entity)
            if not osp.isdir(dpath):
                continue

            dpaths.append(dpath)
        # end of for

        dpaths.sort()

        fpaths = []
        for dpath in dpaths:
            for entity in os.listdir(dpath):
                if not entity.startswith(infile_header):
                    continue

                fpath = osp.join(dpath, entity)
                if not osp.isfile(fpath):
                    continue

                fpaths.append(fpath)
            # end of for
        # end of for
    # end of if

    if text_format:
        if not text_color:
            text_color = (0, 0, 0)

        if not font_size:
            font_size = 16
        elif not isinstance(font_size, int):
            raise ValueError("font_size should be integer type, not %s" % (type(font_size)))

        if not font:
            font = ImageFont.truetype("arial.ttf", font_size, encoding="UTF-8")

        if not text_margin_ratio:
            text_margin_ratio = 0.2
        elif not isinstance(text_margin_ratio, float):
            raise ValueError("font_size should be float type, not %s" % (type(text_margin_ratio)))

    # Generate images.
    tmr = text_margin_ratio
    imgs_out = [[]]
    cnt_rows = 0

    imgs_in = []
    if dpath_input:
        for i, fpath in enumerate(fpaths):
            img = Image.open(fpath)
            img = img.resize((int(ratio_resize * img.width), int(ratio_resize * img.height)))

            fname, _ = osp.splitext(osp.basename(fpath))
            _, iden = fname.split("_")  # Identifier in the file name

            imgs_in.append((iden, img))

        # end of for
    elif imgs:
        for i, img in enumerate(imgs):
            img = img.resize((int(ratio_resize * img.width), int(ratio_resize * img.height)))
            imgs_in.append((i, img))


    n_imgs = len(imgs_in)
    for i, (iden, img) in enumerate(imgs_in):
        if text_format:
            str_text = "{}{:,}".format(text_format, int(iden))
            bbox = font.getbbox(str_text)

            # bbox: (left, top, right, bottom)
            text_width = bbox[2]
            text_height = bbox[3]

            text_margin_width = int((max(text_width - img.width, 0) / 2)) \
                                + int(tmr * img.width)
            text_margin_height = int(text_height + tmr * text_height)

            mt = margin_top
            mr = margin_right + text_margin_width
            mb = margin_bottom + text_margin_height
            ml = margin_left + text_margin_width

            text_x = int((ml + img.width / 2) - (text_width / 2))
            text_y = int(margin_top + img.height + text_height * tmr)

            img = add_margins(img,
                              margin_top=mt,
                              margin_right=mr,
                              margin_bottom=mb,
                              margin_left=ml,
                              bg_color=bg_color)

            draw = ImageDraw.Draw(img)
            text_pos = (text_x, text_y)
            draw.text(text_pos, str_text, font=font, fill=text_color)
        else:
            img = add_margins(img,
                              margin_top=margin_top,
                              margin_right=margin_right,
                              margin_bottom=margin_bottom,
                              margin_left=margin_left,
                              bg_color=bg_color)

        imgs_out[cnt_rows].append(img)

        if verbose > 0:
            print("[INPUT IMAGE]", fpath)

        if len(imgs_out[cnt_rows]) >= n_cols:
            cnt_rows += 1
            if n_imgs > (cnt_rows * n_cols):
                imgs_out.append([])
    # end of for

    img_out = merge_image_rows(*imgs_out, bg_color=bg_color, alignment=(1, 1))

    return img_out


def merge_single_timeseries(dpath_input,
                            n_cols,
                            infile_header="ladybird",
                            bg_color="white",
                            ratio_resize=1.0,
                            margin_top=10,
                            margin_right=10,
                            margin_bottom=10,
                            margin_left=10,
                            text_format=None,
                            text_color=None,
                            font=None,
                            font_size=None,
                            text_margin_ratio=None,
                            verbose=0):
    """Merge the images of a single morph over time.
    """

    if not osp.isdir(dpath_input):
        raise NotADirectoryError("%s doest not exists."%(dpath_input))


    fpaths = []

    for entity in os.listdir(dpath_input):
        if not entity.startswith(infile_header):
            continue

        fpath = osp.join(dpath_input, entity)
        if not osp.isfile(fpath):
            continue

        fpaths.append(fpath)
    # end of for

    fpaths.sort()
    n_images = len(fpaths)

    if text_format:
        if not text_color:
            text_color = (0, 0, 0)

        if not font_size:
            font_size = 10
        elif not isinstance(font_size, int):
            raise ValueError("font_size should be integer type, not %s"%(type(font_size)))

        if not font:
            font = ImageFont.truetype("arial.ttf", font_size, encoding="UTF-8")

        if not text_margin_ratio:
            text_margin_ratio = 0.2
        elif not isinstance(text_margin_ratio, float):
            raise ValueError("font_size should be float type, not %s"%(type(text_margin_ratio)))
        

    # Generate images.
    tmr = text_margin_ratio
    _imgs = [[]]
    cnt_rows = 0
    for i, fpath in enumerate(fpaths):
        img = Image.open(fpath)
        img = img.resize((int(ratio_resize * img.width), int(ratio_resize * img.height)))

        if text_format:
            fname, _ = osp.splitext(osp.basename(fpath))
            _, iden = fname.split("_")  # identifier in the file name
            
            # bbox: (left, top, right, bottom)         
            str_text = "{}{:,}".format(text_format, int(iden))
            bbox = font.getbbox(str_text) 
            
            text_width = bbox[2]
            text_height = bbox[3]

            text_margin_width = int((max(text_width - img.width, 0) / 2)) + int(tmr*img.width)
            text_margin_height = int(text_height + tmr * text_height)
            
            mt = margin_top
            mr = margin_right + text_margin_width
            mb = margin_bottom + text_margin_height
            ml = margin_left + text_margin_width
            
            text_x = int((ml + img.width / 2) - (text_width / 2))
            text_y = int(margin_top + img.height + text_height * tmr)
            
            img = add_margins(img,
                              margin_top=mt,
                              margin_right=mr,
                              margin_bottom=mb,
                              margin_left=ml,
                              bg_color=bg_color)
                        
            draw = ImageDraw.Draw(img)
            text_pos = (text_x, text_y)
            draw.text(text_pos, str_text, font=font, fill=text_color)
        else:
            img = add_margins(img,
                              margin_top=margin_top,
                              margin_right=margin_right,
                              margin_bottom=margin_bottom,
                              margin_left=margin_left,
                              bg_color=bg_color)

        _imgs[cnt_rows].append(img)

        if verbose > 0:
            print("[INPUT IMAGE]", fpath)

        if len(_imgs[cnt_rows]) >= n_cols:
            cnt_rows += 1
            if n_images > (cnt_rows * n_cols):
                _imgs.append([])
    # end of for

    img_output = merge_image_rows(*_imgs, bg_color=bg_color, alignment=(1, 1))
    # img_output.save(osp.join(dpath_output, fstr_outfile%(i)))

    return img_output


def merge_multiple_timeseries(dpath_input,
                              n_cols,
                              dpath_output=None,
                              infile_header="ladybird",
                              outfile_header="frame",
                              bg_color="white",
                              ratio_resize=0.5,
                              margin_top=10,
                              margin_right=10,
                              margin_bottom=10,
                              margin_left=10,
                              verbose=0):
    """Merge the images of multiple morphs over time.
    """

    if not osp.isdir(dpath_input):
        raise NotADirectoryError("%s doest not exists."%(dpath_input))

    if dpath_output:
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

    imgs = []
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
                              margin_left=margin_left,
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

        img_output = merge_image_rows(*images, bg_color=bg_color, alignment=(1, 1))
        imgs.append(img_output)
        if dpath_output:
            img_output.save(osp.join(dpath_output, fstr_outfile%(i)))

    # end of for
    return imgs
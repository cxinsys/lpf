import os
import os.path as osp

_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}


def create_video(dpath_frames,
                 fpath_output,
                 fps=8,
                 duration=0.01,
                 verbose=0,
                 **kwargs):
    if not osp.isdir(dpath_frames):
        raise NotADirectoryError("%s does not exist."%(dpath_frames))

    import moviepy.editor as mpy

    fpaths = []
    for entity in sorted(os.listdir(dpath_frames)):
        fpath = osp.join(dpath_frames, entity)
        ext = osp.splitext(entity)[1].lower()
        if ext not in _IMAGE_EXTENSIONS:
            continue

        if verbose > 0:
            print("[INPUT IMAGE]", fpath)

        fpaths.append(fpath)
    # end of for

    clips = []
    for fpath in fpaths:
        img = mpy.ImageClip(fpath).set_duration(duration)
        clips.append(img)

    concat_clip = mpy.concatenate_videoclips(clips,
                                             bg_color=(255, 255, 255),
                                             method="compose")
    # _, ext = osp.splitext(fpath_output)
    # ext = ext.lower()
    concat_clip.write_videofile(fpath_output, fps=fps, **kwargs)
    concat_clip.close()

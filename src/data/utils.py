import numpy as np


def apply_window(window, hu_img):
    """
    Windowing function.

    Argument:
        window (dict): key ("W", "L")
        hu_img (array): rescaled pixel array
    """
    l, w = window["L"], window["W"]
    window_min = l - (w // 2)
    window_max = l + (w // 2)
    img = np.clip(hu_img, window_min, window_max)
    img = 255 * ((img - window_min) / w)
    img = img.astype(np.uint8)
    return img


def pad_slices(idx, nslices, l):
    """
    Argument:
        idx (list): list of slice indexes
        nslices (int): number of slices sampled from a study
        l (int): length of a study
    """
    idx = np.arange(l)
    pad = (nslices - l) // 2
    idx = np.pad(idx, (pad, pad + l % 2), mode='edge')
    return idx
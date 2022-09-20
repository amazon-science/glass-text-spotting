import collections
import numpy as np


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(dictionary, sep='.'):
    out_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = out_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return out_dict


def rgb2grey(image: np.ndarray, three_channels=False):
    """
    Converts an RGB image to greyscale. We rely on human color perception,
    taken from http://poynton.ca/PDFs/ColorFAQ.pdf.
    For further reference, you can observe the transform Y'709 at https://en.wikipedia.org/wiki/HSL_and_HSV
    :param image: An RGB image (H, W, C) with uint8 data type
    :param three_channels: Returns an image of (H, W, 3), duplicated along it's third dimension three times
    :return: A greyscale image (H, W) as numpy ndarray with uint8 data type
    """
    grey_image = 0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.0721 * image[:, :, 2]
    grey_image = np.uint8(grey_image)
    if three_channels:
        return np.repeat(grey_image[:, :, None], 3, axis=2)
    else:
        return grey_image


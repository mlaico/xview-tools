"""
"""
# Standard Library imports:
import io as _io

# 3rd Party imports:
import skimage.io as io
from PIL import Image

"""
Functions for visualizing images and annotations.
"""

def convertToJpeg(im):
    """
    (copied from tfr_util.py, so we don't have to import tensorflow)
    Converts an image array into an encoded JPEG string.
    Args:
        im: an image array
    Output:
        an encoded byte string containing the converted JPEG image.
    """
    with _io.BytesIO() as f:
        im = Image.fromarray(im)
        im.save(f, format="JPEG")
        return f.getvalue()
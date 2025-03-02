"""Module for encoding and decoding data structures to and from raw bytes"""

import imageio.v3 as iio
import io
import scipy


def deserialize_image(bytes: bytes) -> scipy.ndimage:
    """Decode the image from raw bytes using PIL."""
    buffer = io.BytesIO(bytes)
    return iio.imread(buffer)


def serialize_image(image: scipy.ndimage) -> bytes:
    """Encode the image as raw bytes using PIL."""
    buffer = io.BytesIO()
    iio.imwrite(buffer, image, format="PNG")
    return buffer.getvalue()

import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


from typing import Any


def to_pil_preserve(images: Any, scale_float: bool = True):
    """
    Convert (possibly nested) numpy image arrays back to PIL.Image WITHOUT changing spatial shape
    or nesting structure.

    Accepts:
      - np.ndarray with shape (H, W, C), C in {1,3,4}, dtype uint8 or float
      - PIL.Image.Image (returned as-is)
      - Nested list / tuple structures containing the above

    Guarantees:
      - No resize / pad / crop performed
      - Returns an object with the SAME nesting layout (list -> list, tuple -> tuple)
      - Only dtype (float -> uint8) and channel-mode adaptation may happen
        * float arrays assumed in [0,1] if scale_float=True (scaled *255 + clip)
    Args:
      images: input object / sequence
      scale_float: whether to scale float images in [0,1] to uint8
    Returns:
      Mirrored structure with all leaf nodes as PIL.Image.Image
    """

    def _convert(obj):
        # Nested containers
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_convert(x) for x in obj)

        # PIL stays
        if isinstance(obj, Image.Image):
            return obj

        # numpy -> PIL
        if isinstance(obj, np.ndarray):
            arr = obj
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D array (H,W,C), got shape={arr.shape}")
            if arr.shape[2] not in (1, 3, 4):
                raise ValueError(f"Channel count must be 1/3/4, got {arr.shape[2]}")
            if np.issubdtype(arr.dtype, np.floating):
                if scale_float:
                    arr = np.clip(arr, 0.0, 1.0)
                    arr = (arr * 255.0 + 0.5).astype(np.uint8)
                else:
                    raise TypeError("Float array provided but scale_float=False")
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            # Single channel -> 'L'
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
                return Image.fromarray(arr, mode="L")
            # 3 channels -> RGB, 4 -> RGBA
            mode = "RGB" if arr.shape[2] == 3 else "RGBA"
            return Image.fromarray(arr, mode=mode)

        raise TypeError(f"Unsupported element type: {type(obj)}")

    return _convert(images)

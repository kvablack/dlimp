from typing import Tuple

import tensorflow as tf


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes an image using Lanczos3 interpolation. Expects & returns uint8."""
    assert image.dtype == tf.uint8
    image = tf.image.resize(image, size, method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image


def resize_depth_image(depth_image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes a depth image using bilinear interpolation. Expects & returns float32 in arbitrary range."""
    assert depth_image.dtype == tf.float32
    if len(depth_image.shape) < 3:
        depth_image = tf.image.resize(
            depth_image[..., None], size, method="bilinear", antialias=True
        )[..., 0]
    else:
        depth_image = tf.image.resize(
            depth_image, size, method="bilinear", antialias=True
        )
    return depth_image


def read_resize_encode_image(path: str, size: Tuple[int, int]) -> tf.Tensor:
    """Reads, decodes, resizes, and then re-encodes an image."""
    data = tf.io.read_file(path)
    image = tf.image.decode_jpeg(data)
    image = resize_image(image, size)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return tf.io.encode_jpeg(image, quality=95)

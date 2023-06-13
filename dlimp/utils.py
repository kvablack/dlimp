from typing import Tuple
import tensorflow as tf


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes an image using Lanczos3 interpolation. Assumes uint8 images."""
    assert image.dtype == tf.uint8
    image = tf.image.resize(image, size, method="lanczos3")
    return tf.cast(tf.round(image), tf.uint8)


def read_resize_encode_image(path: str, size: Tuple[int, int]) -> tf.Tensor:
    """Reads, decodes, resizes, and then re-encodes an image."""
    data = tf.io.read_file(path)
    image = tf.image.decode_jpeg(data)
    image = resize_image(image, size)
    return tf.io.encode_jpeg(image, quality=95)

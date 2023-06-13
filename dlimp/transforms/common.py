from typing import Any, Dict, TypeAlias, Callable
import tensorflow as tf
from absl import logging

Transform: TypeAlias = Callable[[Dict[str, Any]], Dict[str, Any]]


def add_next_obs(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a trajectory with a key "obs", add the key "next_obs". Discards the last value of all other
    keys. "obs" may be nested.
    """
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["next_obs"] = tf.nest.map_structure(lambda x: x[1:], traj["obs"])
    return traj_truncated


def flatten_dict(d: Dict[str, Any], sep="/") -> Dict[str, Any]:
    """Given a nested dictionary, flatten it by concatenating keys with sep."""
    flattened = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v, sep=sep).items():
                flattened[k + sep + k2] = v2
        else:
            flattened[k] = v
    return flattened


def unflatten_dict(d: Dict[str, Any], sep="/") -> Dict[str, Any]:
    """Given a flattened dictionary, unflatten it by splitting keys by sep."""
    unflattened = {}
    for k, v in d.items():
        keys = k.split(sep)
        if len(keys) == 1:
            unflattened[k] = v
        else:
            if keys[0] not in unflattened:
                unflattened[keys[0]] = {}
            unflattened[keys[0]][sep.join(keys[1:])] = v
    return unflattened


def decode_images(x: Dict[str, Any], decode=False) -> Dict[str, Any]:
    """Can operate on nested dicts. Decodes any leaves that are tf.string and have "image" anywhere in their path."""
    for key in x:
        if isinstance(x[key], dict):
            x[key] = decode_images(x[key], decode=decode or "image" in key)
        elif x[key].dtype == tf.string and (decode or "image" in key):
            if len(x[key].shape) == 0:
                x[key] = tf.io.decode_image(x[key])
            else:
                logging.warning(
                    "Using tf.map_fn to decode images. This is slow. "
                    "Using decode_images as a frame_transform instead is recommended, "
                    "even if it means repeating decode operations."
                )
                x[key] = tf.map_fn(
                    tf.io.decode_image, x[key], fn_output_signature=tf.uint8
                )
            # convert to float and normalize to [-1, 1]
            x[key] = tf.cast(x[key], tf.float32) / 127.5 - 1.0
    return x

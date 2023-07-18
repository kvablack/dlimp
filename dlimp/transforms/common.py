from functools import partial
from typing import Any, Dict, Tuple, Callable, Union, Literal, Iterable
import tensorflow as tf
from absl import logging
from dlimp.augmentations import augment_image
from dlimp.utils import resize_image

Transform = Union[Callable[[Dict[str, Any]], Dict[str, Any]], Literal["cache"]]


def apply_transforms(
    ds: tf.data.Dataset,
    transforms: Iterable[Transform],
    num_parallel_calls: int = tf.data.AUTOTUNE,
    deterministic: bool = False,
):
    for transform in transforms:
        if transform == "cache":
            ds = ds.cache()
        else:
            ds = ds.map(
                transform,
                num_parallel_calls=num_parallel_calls,
                deterministic=deterministic,
            )
    return ds


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


def selective_tree_map(
    x: Dict[str, Any],
    match_fn: Callable[[str, Any], bool],
    map_fn: Callable,
    *,
    _keypath: str = "",
) -> Dict[str, Any]:
    """Maps a function over a nested dictionary, only applying it leaves that match a criterion.

    Args:
        x (Dict[str, Any]): The dictionary to map over.
        match (Callable[[str, Any], bool]): A function that takes a full key path separated by slashes as well as a
            value and returns True if the function should be applied to that value.
        fn (Callable): The function to apply.
    """
    for key in x:
        if isinstance(x[key], dict):
            x[key] = selective_tree_map(x[key], match_fn, map_fn, _keypath=_keypath + key + "/")
        elif match_fn(_keypath + key, x[key]):
            x[key] = map_fn(x[key])
    return x


def decode_images(x: Dict[str, Any], match_str: str = "image") -> Dict[str, Any]:
    """Can operate on nested dicts. Decodes any leaves that have `match_str` anywhere in their path."""

    def map_fn(value):
        if len(value.shape) == 0:
            return tf.io.decode_image(value, expand_animations=False)
        else:
            logging.warning(
                "Using tf.map_fn to decode images. This is slow. "
                "Using decode_images as a frame_transform instead is recommended, "
                "even if it means repeating decode operations."
            )
            value = tf.map_fn(
                partial(tf.io.decode_image, expand_animations=False),
                value,
                fn_output_signature=tf.uint8,
            )

    return selective_tree_map(
        x,
        lambda keypath, value: match_str in keypath and value.dtype == tf.string,
        map_fn,
    )


def resize_images(x: Dict[str, Any], match_str: str = "image", size: Tuple[int, int] = (128, 128)) -> Dict[str, Any]:
    """Can operate on nested dicts. Resizes any leaves that have `match_str` anywhere in their path. Takes uint8 images
    as input and returns float images (still in [0, 255]).
    """

    def map_fn(value):
        if len(value.shape) == 3:
            return resize_image(value, size=size)
        else:
            logging.warning(
                "Using tf.map_fn to resize images. This is slow. "
                "Using resize_images as a frame_transform instead is recommended, "
                "even if it means repeating resize operations."
            )
            value = tf.map_fn(
                partial(resize_image, size=size),
                value,
                fn_output_signature=tf.uint8,
            )

    return selective_tree_map(
        x,
        lambda keypath, value: match_str in keypath and value.dtype == tf.uint8,
        map_fn,
    )


def augment(
    x: Dict[str, Any],
    match_str: str = "image",
    traj_identical: bool = True,
    augment_kwargs: dict = {},
) -> Dict[str, Any]:
    """
    Augments the input dictionary `x` by applying image augmentation to all values whose keypath contains `match_str`.

    Args:
        x (Dict[str, Any]): The input dictionary to augment.
        match_str (str, optional): The string to match in keypaths. Defaults to "image".
        traj_identical (bool, optional): Whether to use the same random seed for all images in a trajectory.
        augment_kwargs (dict, optional): Additional keyword arguments to pass to the `augment_image` function.
    """

    def map_fn(value):
        if traj_identical:
            seed = [x["_i"], x["_i"]]
        else:
            seed = None
        return augment_image(value, seed=seed, **augment_kwargs)

    return selective_tree_map(
        x,
        lambda keypath, value: match_str in keypath,
        map_fn,
    )

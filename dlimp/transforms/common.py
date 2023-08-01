import inspect
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import tensorflow as tf

from dlimp.augmentations import augment_image
from dlimp.utils import resize_image


class Transform:
    class TransformType(Enum):
        TRAJ = "traj"
        FRAME = "frame"
        CACHE = "cache"

    def __init__(
        self,
        type: TransformType,
        func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        if type == Transform.TransformType.CACHE:
            assert func is None
        self.type = type
        self.func = func

    def __call__(self, *args, **kwargs):
        if self.type == Transform.TransformType.CACHE:
            raise ValueError("Cannot call a cache transform.")

        return self.func(*args, **kwargs)

    def __repr__(self):
        if self.func is None:
            func_str = "None"
        else:
            func_str = f"{self.func.__name__} defined in "
            f"{inspect.getsourcefile(self.func)} at line {inspect.getsourcelines(self.func)[1]}"

        return f"Transform(type={self.type}, func={func_str})"


CACHE = Transform(Transform.TransformType.CACHE)


def traj_transform(func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Transform:
    """Wrapper that creates a trajectory-level transform."""
    return Transform(Transform.TransformType.TRAJ, func)


def frame_transform(func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Transform:
    """Wrapper that creates a frame-level transform."""
    return Transform(Transform.TransformType.FRAME, func)


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
    out = {}
    for key in x:
        if isinstance(x[key], dict):
            out[key] = selective_tree_map(
                x[key], match_fn, map_fn, _keypath=_keypath + key + "/"
            )
        elif match_fn(_keypath + key, x[key]):
            out[key] = map_fn(x[key])
        else:
            out[key] = x[key]
    return out


# --- Below are some common utility transforms --- #


@traj_transform
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


@traj_transform
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


@traj_transform
def add_next_obs(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a trajectory with a key "obs", add the key "next_obs". Discards the last value of all other
    keys. "obs" may be nested.
    """
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["next_obs"] = tf.nest.map_structure(lambda x: x[1:], traj["obs"])
    return traj_truncated


@frame_transform
def decode_images(x: Dict[str, Any], match_str: str = "image") -> Dict[str, Any]:
    """Can operate on nested dicts. Decodes any leaves that have `match_str` anywhere in their path."""

    return selective_tree_map(
        x,
        lambda keypath, value: match_str in keypath and value.dtype == tf.string,
        partial(tf.io.decode_image, expand_animations=False),
    )


@frame_transform
def resize_images(
    x: Dict[str, Any], match_str: str = "image", size: Tuple[int, int] = (128, 128)
) -> Dict[str, Any]:
    """Can operate on nested dicts. Resizes any leaves that have `match_str` anywhere in their path. Takes uint8 images
    as input and returns float images (still in [0, 255]).
    """
    return selective_tree_map(
        x,
        lambda keypath, value: match_str in keypath and value.dtype == tf.uint8,
        partial(resize_image, size=size),
    )


@frame_transform
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

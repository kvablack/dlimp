from typing import Any, Callable, Dict, Sequence, Union


def selective_tree_map(
    x: Dict[str, Any],
    match: Union[str, Sequence[str], Callable[[str, Any], bool]],
    map_fn: Callable,
    *,
    _keypath: str = "",
) -> Dict[str, Any]:
    """Maps a function over a nested dictionary, only applying it leaves that match a criterion.

    Args:
        x (Dict[str, Any]): The dictionary to map over.
        match (str or Sequence[str] or Callable[[str, Any], bool]): If a string or list of strings, `map_fn` will only
        be applied to leaves whose key path contains one of `match`. If a function, `map_fn` will only be applied to
        leaves for which `match(key_path, value)` returns True.
        map_fn (Callable): The function to apply.
    """
    if not callable(match):
        if isinstance(match, str):
            match = [match]
        match_fn = lambda keypath, value: any([s in keypath for s in match])
    else:
        match_fn = match

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

from typing import Any, Dict

import tensorflow as tf


def add_next_obs(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a trajectory with a key "obs", add the key "next_obs". Discards the last value of all other
    keys. "obs" may be nested.
    """
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["next_obs"] = tf.nest.map_structure(lambda x: x[1:], traj["obs"])
    return traj_truncated

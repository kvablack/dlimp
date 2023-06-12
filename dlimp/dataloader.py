from typing import Any, Callable, Dict, Sequence
import tensorflow as tf
from functools import partial
from dlimp import goal_relabeling
from dlimp.utils import unflatten_dict, add_next_obs, decode_images


transform = Callable[[Dict[str, Any]], Dict[str, Any]]


def make_dataset(
    path: str,
    *,
    seed: int = 0,
    batch_size: int,
    shuffle_buffer_size: int = 25000,
    cache: bool = False,
    traj_transforms: Sequence[transform] = (
        unflatten_dict,
        add_next_obs,
        partial(goal_relabeling.uniform, reached_proportion=0.1),
    ),
    frame_transforms: Sequence[transform] = (decode_images,),
) -> tf.data.Dataset:
    """Get a tf.data.Dataset from a directory of tfrecord files.

    Args:
        path (str): Path to a directory containing tfrecord files. seed (int, optional): Random seed. Defaults to 0.
        seed (int, optional): Random seed for shuffling.
        batch_size (int): Batch size. shuffle_buffer_size (int, optional): Size of the shuffle buffer. Defaults to
            25000. Set to 1 to disable shuffling.
        cache (bool, optional): Whether to cache the dataset in memory. Defaults to False.
        traj_transforms (Sequence[transform], optional): A sequence of functions to apply at the trajectory level.
        frame_transforms (Sequence[transform], optional): A sequence of functions to apply at the frame level.
    """
    # get the tfrecord files
    paths = tf.io.gfile.glob(tf.io.gfile.join(path, "*.tfrecord"))

    # extract the type spec from the first file
    type_spec = _get_type_spec(paths[0])

    # read the tfrecords (yields raw serialized examples)
    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.AUTOTUNE)

    # decode the examples (yields trajectories)
    dataset = dataset.map(
        partial(_decode_example, type_spec=type_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # cache all the dataloading
    if cache:
        dataset = dataset.cache()

    # apply trajectory transforms
    for transform in traj_transforms:
        dataset = dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)

    # unbatch to get individual frames
    dataset = dataset.unbatch()

    # apply frame transforms
    for transform in frame_transforms:
        dataset = dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.repeat()

    # shuffle the dataset
    if shuffle_buffer_size > 1:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

    # batch the dataset
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    # always prefetch last
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _decode_example(
    example_proto: tf.Tensor, type_spec: Dict[str, tf.TensorSpec]
) -> Dict[str, tf.Tensor]:
    features = {key: tf.io.FixedLenFeature([], tf.string) for key in type_spec.keys()}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], spec.dtype)
        for key, spec in type_spec.items()
    }

    for key in parsed_tensors:
        parsed_tensors[key] = tf.ensure_shape(parsed_tensors[key], type_spec[key].shape)

    return parsed_tensors


def _get_type_spec(path: str) -> Dict[str, tf.TensorSpec]:
    """Get a type spec from a tfrecord file.

    Args:
        path (str): Path to a single tfrecord file.

    Returns:
        dict: A dictionary mapping feature names to tf.TensorSpecs.
    """
    data = next(iter(tf.data.TFRecordDataset(path))).numpy()
    example = tf.train.Example()
    example.ParseFromString(data)

    out = {}
    for key, value in example.features.feature.items():
        data = value.bytes_list.value[0]
        tensor_proto = tf.make_tensor_proto([])
        tensor_proto.ParseFromString(data)
        dtype = tf.dtypes.as_dtype(tensor_proto.dtype)
        shape = [d.size for d in tensor_proto.tensor_shape.dim]
        shape[0] = None  # first dimension is trajectory length, which is variable
        out[key] = tf.TensorSpec(shape=shape, dtype=dtype)

    return out

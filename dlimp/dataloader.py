from typing import Dict, Sequence
import tensorflow as tf
from functools import partial
import dlimp as dl


def make_dataset(
    path: str,
    *,
    seed: int = 0,
    batch_size: int,
    shuffle_buffer_size: int = 1,
    transforms: Sequence[dl.transforms.Transform] = (
        dl.transforms.unflatten_dict,
        dl.transforms.decode_images,
        dl.transforms.add_next_obs,
        dl.CACHE,
    ),
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from a directory of tfrecord files.

    The dataset is customizable by passing in a sequence of transforms. A transform can be one of three things:

    1) a function to apply at the trajectory level
    2) a function to apply at the frame level
    3) a caching operation

    For 1) and 2), the function should take a dictionary of tensors and return a dictionary of tensors. To indicate
    that the function is a trajectory-level transform, wrap it with `dl.traj_transform`. To indicate that the function
    is a frame-level transform, wrap it with `dl.frame_transform`. For caching, simply use `dl.CACHE`.

    Be useful when caching the dataset, as it will freeze any randomness from previous transforms. It may also use a lot
    of memory.

    Args:
        path (str): Path to a directory containing tfrecord files. seed (int, optional): Random seed. Defaults to 0.
        seed (int, optional): Random seed for shuffling.
        batch_size (int): Batch size.
        shuffle_buffer_size (int, optional): Size of the shuffle buffer. Default 1, which disables shuffling.
        transforms (Sequence[Transform], optional): Sequence of transforms to apply to the dataset. See above.
    """
    # get the tfrecord files
    paths = tf.io.gfile.glob(tf.io.gfile.join(path, "*.tfrecord"))

    if shuffle_buffer_size > 1:
        paths = tf.random.shuffle(paths, seed=seed)

    # extract the type spec from the first file
    type_spec = _get_type_spec(paths[0])

    # read the tfrecords (yields raw serialized examples)
    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.AUTOTUNE)

    # set options
    options = tf.data.Options()
    options.autotune.enabled = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_fusion = True
    dataset = dataset.with_options(options)

    # decode the examples (yields trajectories)
    dataset = dataset.map(
        partial(_decode_example, type_spec=type_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # add a unique index and length metadata to each trajectory. also broadcasts any scalar elements to every frame.
    dataset = dataset.enumerate().map(
        _add_traj_metadata, num_parallel_calls=tf.data.AUTOTUNE
    )

    # apply transforms
    for transform in transforms:
        # in case the transform is a partial, unwrap it
        unwrapped = transform
        while isinstance(unwrapped, partial):
            unwrapped = unwrapped.func

        if not isinstance(unwrapped, dl.transforms.Transform):
            print(unwrapped)
            raise ValueError(
                f"Expected a Transform, got {unwrapped} instead."
                " Did you wrap your function with dl.traj_transform or dl.frame_transform?"
            )

        if unwrapped.type == dl.transforms.Transform.TransformType.CACHE:
            dataset = dataset.cache()
        elif unwrapped.type == dl.transforms.Transform.TransformType.TRAJ:
            dataset = dataset.map(
                transform,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif unwrapped.type == dl.transforms.Transform.TransformType.FRAME:
            dataset = dataset.map(
                lambda traj: tf.data.Dataset.from_tensor_slices(traj)
                .map(transform, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(tf.cast(traj["_len"][0], tf.int64))
                .get_single_element(),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            raise ValueError(f"Unknown transform type {transform.type}")

    # unbatch to get individual frames
    dataset = dataset.unbatch()

    # shuffle the dataset
    if shuffle_buffer_size > 1:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

    dataset = dataset.repeat()

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
        if shape:
            shape[0] = None  # first dimension is trajectory length, which is variable
        out[key] = tf.TensorSpec(shape=shape, dtype=dtype)

    return out


def _add_traj_metadata(i: tf.Tensor, x: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    # get the length of each dict entry
    traj_lens = {k: tf.shape(v)[0] if len(v.shape) > 0 else None for k, v in x.items()}

    # take the maximum length as the canonical length (elements should either be the same length or length 1)
    traj_len = tf.reduce_max([l for l in traj_lens.values() if l is not None])

    for k in x:
        # broadcast scalars to the length of the trajectory
        if traj_lens[k] is None:
            x[k] = tf.repeat(x[k], traj_len)
            traj_lens[k] = traj_len

        # broadcast length-1 elements to the length of the trajectory
        if traj_lens[k] == 1:
            x[k] = tf.repeat(x[k], traj_len, axis=0)
            traj_lens[k] = traj_len

    asserts = [
        # make sure all the lengths are the same
        tf.assert_equal(tf.size(tf.unique(tf.stack(list(traj_lens.values()))).y), 1),
    ]

    assert "_i" not in x
    assert "_len" not in x
    x["_i"] = tf.repeat(i, traj_len)
    x["_len"] = tf.repeat(traj_len, traj_len)

    with tf.control_dependencies(asserts):
        return x

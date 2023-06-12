import tensorflow as tf


def random_resized_crop(image, scale, ratio, seed):
    if len(tf.shape(image)) == 3:
        image = tf.expand_dims(image, axis=0)
    batch_size = tf.shape(image)[0]
    # taken from https://keras.io/examples/vision/nnclr/#random-resized-crops
    log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]

    random_scales = tf.random.stateless_uniform((batch_size,), seed, scale[0], scale[1])
    random_ratios = tf.exp(
        tf.random.stateless_uniform((batch_size,), seed, log_ratio[0], log_ratio[1])
    )

    new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
    new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
    height_offsets = tf.random.stateless_uniform(
        (batch_size,), seed, 0, 1 - new_heights
    )
    width_offsets = tf.random.stateless_uniform((batch_size,), seed, 0, 1 - new_widths)

    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (height, width)
    )

    return tf.squeeze(image)


def random_rot90(image, seed):
    k = tf.random.stateless_uniform((), seed, 0, 4, dtype=tf.int32)
    return tf.image.rot90(image, k=k)


AUGMENT_OPS = {
    "random_resized_crop": random_resized_crop,
    "random_brightness": tf.image.stateless_random_brightness,
    "random_contrast": tf.image.stateless_random_contrast,
    "random_saturation": tf.image.stateless_random_saturation,
    "random_hue": tf.image.stateless_random_hue,
    "random_flip_left_right": tf.image.stateless_random_flip_left_right,
    "random_flip_up_down": tf.image.stateless_random_flip_up_down,
    "random_rot90": random_rot90,
}


def augment(image: tf.Tensor, seed: tf.Tensor, **augment_kwargs) -> tf.Tensor:
    # convert images to [0, 1]
    dtype = image.dtype
    if dtype == tf.float32:
        # assume images are in [-1, 1]
        image = image * 0.5 + 0.5
    elif dtype == tf.uint8:
        image = tf.cast(image, tf.float32) / 255.0
    else:
        raise ValueError(f"Invalid image dtype: {image.dtype}")

    for op in augment_kwargs["augment_order"]:
        if op in augment_kwargs:
            if hasattr(augment_kwargs[op], "items"):
                image = AUGMENT_OPS[op](image, seed=seed, **augment_kwargs[op])
            else:
                image = AUGMENT_OPS[op](image, seed=seed, *augment_kwargs[op])
        else:
            image = AUGMENT_OPS[op](image, seed=seed)
        image = tf.clip_by_value(image, 0, 1)

    # convert back to original dtype and scale
    if dtype == tf.float32:
        image = image * 2 - 1
    else:
        image = tf.cast(image * 255, dtype)

    return image

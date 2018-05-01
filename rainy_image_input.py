import glob
import os
import tensorflow as tf

GROUND_TRUTH_DIR = "ground truth"
RAINY_IMAGE_DIR = "rainy image"

def _get_indices(data_dir):
    """Get indices for input/output association.

    Args:
    data_dir: Path to the data directory.

    Returns:
    indices: List of numerical index values.

    Raises:
    ValueError: If no data_dir or no ground-truth dir.
    """

    if not tf.gfile.Exists(os.path.join(data_dir, GROUND_TRUTH_DIR)):
        raise ValueError("Failed to find ground-truth directory.")

    return [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(data_dir, GROUND_TRUTH_DIR, "*.jpg"))
    ]

def _get_input_files(data_dir, indices):
    """Get input files from indices.

    Args:
    data_dir: Path to the data directory.
    indices: List of numerical index values.

    Returns:
    Dictionary, keyed by index value, valued by string lists containing
    one or more filenames.

    Raises:
    ValueError: If no rainy-image dir.
    """

    directory = os.path.join(data_dir, RAINY_IMAGE_DIR)
    if not tf.gfile.Exists(directory):
        raise ValueError("Failed to find rainy-image directory.")

    return {
        i: glob.glob(os.path.join(directory, "{}_[0-9]*.jpg".format(i)))
        for i in indices
    }

def _get_output_files(data_dir, indices):
    """Get output files from indices.

    Args:
    data_dir: Path to the data directory.
    indices: List of numerical index values.

    Returns:
    outputs: Dictionary, keyed by index value, valued by stsring lists
    containing one or more filenames.

    Raises:
    ValueError: If no ground-truth dir.
    """

    directory = os.path.join(data_dir, GROUND_TRUTH_DIR)
    if not tf.gfile.Exists(directory):
        raise ValueError("Failed to find ground-truth directory.")

    return {
        i: os.path.join(directory, "{}.jpg".format(i)) for i in indices
    }

IMAGE_SIZE = 384

def dataset(data_dir, indices=None):
    """Construct dataset for rainy-image evaluation.

    Args:
    data_dir: Path to the data directory.
    indices: The input-output pairings to return. If None (the default), uses
    indices present in the data directory.

    Returns:
    dataset: Dataset of input-output images.
    """

    if not indices:
        indices = _get_indices(data_dir)

    fs_in = _get_input_files(data_dir, indices)
    fs_out = _get_output_files(data_dir, indices)

    ins = [
        fname for k, v in iter(sorted(fs_in.items()))
        for fname in v if k in indices
    ]

    outs = [v for sublist in [
        [fname] * len(fs_in[k])
        for k, fname in iter(sorted(fs_out.items()))
        if k in indices
    ] for v in sublist]

    def _parse_function(fname_in, fname_out):
        def _decode_resize(fname):
            return tf.image.resize_image_with_crop_or_pad(
                tf.image.decode_jpeg(fname), IMAGE_SIZE, IMAGE_SIZE,
            )

        return _decode_resize(fname_in), _decode_resize(fname_out)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(ins), tf.constant(outs)),
    ).map(_parse_function)

    return dataset

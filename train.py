import tensorflow as tf
import logging
from rainy_image_input import dataset, IMAGE_SIZE

tf.app.flags.DEFINE_string("checkpoint_dir", "/tmp/derain-checkpoint",
                           """Directory to write event logs and checkpointing
                           to.""")

tf.app.flags.DEFINE_string("data_dir",
                           "/tmp/derain_data",
                           """Path to the derain data directory.""")

tf.app.flags.DEFINE_integer("batch_size",
                            128,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer("max_steps",
                            1000000,
                            """Number of training batches to run.""")

LEVEL = tf.logging.DEBUG
FLAGS = tf.app.flags.FLAGS

LOG = logging.getLogger("derain-train")

def dataset_input_fn():
    # TODO: use FLAGS.batch_size
    return dataset(FLAGS.data_dir, range(1, 2)).batch(1)

MODEL_DEFAULT_PARAMS = {
    "learn_rate": 0.01,
}


def model_fn(features, labels, mode, params):
    inputs = features
    tf.summary.image("inputs", inputs)
    global_step = tf.train.get_or_create_global_step()

    params = {**MODEL_DEFAULT_PARAMS, **params}

    l = tf.keras.layers
    model = tf.keras.Sequential([
        # 512 kernels of 16x16x3
        l.Conv2D(
            512,
            (16, 16),
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            use_bias=True,
            activation=tf.nn.tanh,
            padding="same",
        ),
        # 512 kernels of 1x1x512
        l.Conv2D(
            512,
            (1, 1),
            use_bias=True,
            activation=tf.nn.tanh,
        ),
        # 3 kernels of 8x8x512 (one for each color channel)
        l.Conv2D(
            3,
            (8, 8),
            use_bias=True,
            padding="same",
        ),
    ])

    # TODO: handle each of ModeKeys.{EVAL,TRAIN,PREDICT}
    if mode == tf.estimator.ModeKeys.TRAIN:
        predictions = model(inputs, training=True)

        norm = tf.norm(inputs - predictions, ord="fro", axis=[-2, -1])
        loss = tf.reduce_mean(norm, 1)
        tf.summary.scalar("loss", loss)

        optimizer = tf.train.GradientDescentOptimizer(params["learn_rate"])

        train_op = optimizer.minimize(loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[
                tf.train.LoggingTensorHook(
                    ["loss", "inputs"],
                    # TODO: parameterize
                    every_n_iter=3,
                )
            ],
        )

    raise NotImplementedError

def train():
    regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.checkpoint_dir,
        # TODO
        config=None,
        params={},
    )

    regressor.train(
        input_fn=dataset_input_fn,
        steps=FLAGS.max_steps,
    )


def main(argv=None):
    if tf.gfile.Exists(FLAGS.checkpoint_dir):
        LOG.debug("Emptying checkpoint dir")
        tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)

    LOG.debug("Creating checkpoint dir")
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    train()

if __name__ == "__main__":
    tf.logging.set_verbosity(LEVEL)
    tf.app.run(main)

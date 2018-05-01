import tensorflow as tf
import logging

LEVEL = tf.logging.DEBUG
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(level=LEVEL)

tf.app.flags.DEFINE_string("checkpoint_dir", "/tmp/derain-checkpoint",
                           """Directory to write event logs and checkpointing
                           to.""")

tf.app.flags.DEFINE_string("data_dir",
                           "/tmp/derain_data",
                           """Path to the derain data directory.""")

tf.app.flags.DEFINE_integer("batch_size",
                            128,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer(
    "max_steps",
    1000000,
    """Number of training batches to run.""",
)

log = logging.getLogger("derain-train")


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        c = tf.constant("Hello")
        with tf.train.MonitoredTrainingSession(
                hooks=[
                    tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                ],
        ) as train_session:
            while not train_session.should_stop():
                res = train_session.run(c)
                log.debug(res)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.checkpoint_dir):
        log.debug("Emptying checkpoint dir")
        tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
    log.debug("Creating checkpoint dir")
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    train()


if __name__ == "__main__":
    log.info("FLAGS: {}".format(FLAGS.max_steps))
    tf.app.run(main)

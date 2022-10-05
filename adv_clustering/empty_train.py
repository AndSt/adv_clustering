import os
import json
import random

from absl import app
from absl import logging
from absl import flags

FLAGS = flags.FLAGS

# data flags
flags.DEFINE_string(
    "data_dir", default="",
    help="Data directory. Here, data is loaded for computation."
)

flags.DEFINE_string(
    "work_dir", default="",
    help="Working directory. Here, data is saved."
)

flags.DEFINE_string(
    "objective", default="accuracy",
    help="Working directory. Here, data is saved."
)

flags.DEFINE_integer(
    "train_parameter", default=1,
    help="Example of a parameter used for training"
)


def main(_):
    # data loading test
    logging.info("Load data.")

    # Place where training would happen
    logging.info("Here happens training.")
    logging.info(f"Value of training parameter: {FLAGS.train_parameter}")

    # save train metric
    acc = random.random()
    f1 = random.random()
    logging.info("Save metrics.")
    with open(os.path.join(FLAGS.work_dir, "test_metrics.json"), "w") as f:
        json.dump({FLAGS.objective: acc, "macro avg": {"f1-score": f1}}, f)


if __name__ == '__main__':
    app.run(main)

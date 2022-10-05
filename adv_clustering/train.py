import os
import json
import random

from absl import app
from absl import logging
from absl import flags

import torch

from adv_clustering.model import ARL
from adv_clustering.trainer import Trainer
from adv_clustering.loader import get_dataloader

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

flags.DEFINE_bool(
    "debug", default=False,
    help="Working directory. Here, data is saved."
)

flags.DEFINE_integer(
    "num_clusters", default=64,
    help="Number of clusters"
)

flags.DEFINE_integer(
    "num_epochs", default=100, help="Number of epochs."
)

flags.DEFINE_float(
    "lr", default=1e-3, help="Learning rate."
)


def main(_):
    # data loading test
    logging.info("Load data.")
    word_embs, train_loader = get_dataloader(os.path.join(FLAGS.data_dir))

    model = ARL(word_vecs=word_embs, num_clusters=FLAGS.num_clusters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = Trainer(model=model, device=device)
    t.train(data_loader=train_loader, lr=FLAGS.lr, num_epochs=FLAGS.num_epochs)

    # save train metric
    logging.info("Save metrics.")
    ev = t.evaluate(train_loader)
    with open(os.path.join(FLAGS.work_dir, "test_metrics.json"), "w") as f:
        json.dump({FLAGS.objective: ev["acc"], "full": ev}, f)


if __name__ == '__main__':
    app.run(main)

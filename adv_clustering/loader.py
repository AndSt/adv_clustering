import os
import json

import joblib
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class PosNegDataset(Dataset):
    def __init__(self, samples: torch.tensor, masks: torch.Tensor, labels: torch.Tensor, num_negatives: int = 16):
        self.samples = samples.long()
        self.masks = masks.float()
        self.labels = labels.float()

        self.num_negatives = num_negatives
        self.num_samples = len(self.samples)

        self.negatives = []
        self.initialize_negatives()

    def initialize_negatives(self):
        counts = np.arange(self.num_samples)
        for i in range(self.num_samples):
            without_i = np.delete(counts, i)
            self.negatives.append(np.random.choice(without_i, size=self.num_negatives, replace=False))
        self.negatives = torch.from_numpy(np.array(self.negatives))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pos = self.samples[idx]
        negs = self.samples[self.negatives[idx]]
        pos_mask = self.masks[idx]
        neg_masks = self.masks[self.negatives[idx]]
        label = self.labels[idx]
        sample = {"pos": pos, "pos_mask": pos_mask, "neg": negs, "neg_mask": neg_masks, "labels": label}
        return sample


def load_data(data_dir: str):
    vec = joblib.load(os.path.join(data_dir, "Event_word2vec_300.pkl"))
    with open(os.path.join(data_dir, "Event_word2wid.txt")) as f:
        event2id = json.load(f)

    word_embs = np.zeros((len(event2id), vec.get("break").shape[0]))

    for key, val in event2id.items():
        word_embs[val - 1] = vec.get(key)
    word_embs = torch.from_numpy(word_embs)

    with open(os.path.join(data_dir, "Event_docarr.txt")) as f:
        # lines = f.readlines()
        # lines = [json.loads(l) for l in lines]
        lines = json.load(f)

    padded_length = 0
    for line in lines:
        if len(line["tokenids"]) > padded_length:
            padded_length = len(line["tokenids"])

    samples = []
    masks = []
    labels = []

    for line in lines:
        labels.append(line["topic"])
        ids = line["tokenids"]
        ids = np.array([i - 1 for i in ids])
        samples.append(torch.from_numpy(ids))
        mask = np.zeros((padded_length,))
        mask[0:len(ids)] = 1
        masks.append(mask)

    labels = torch.from_numpy(np.array(labels))
    samples = torch.nn.utils.rnn.pad_sequence(samples).T
    masks = torch.from_numpy(np.array(masks))

    assert masks.shape == samples.shape
    return samples, masks, labels, word_embs


def get_dataloader(data_dir: str, num_negatives: int = 16, batch_size: int = 64):
    # load word id tensor
    # load text id tensor
    samples, masks, labels, word_embs = load_data(data_dir=data_dir)
    dataset = PosNegDataset(samples=samples, masks=masks, labels=labels, num_negatives=num_negatives)
    return word_embs, DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from adv_clustering.evaluate import full_evaluate


class Trainer:

    def __init__(self, model, device, num_epochs: int = 3):
        self.model = model
        self.num_epochs = num_epochs

        self.device = device
        self.model.to(self.device)

    def compute_loss(self, d, d_rec, d_neg, margin: float = 1):
        L_2 = - torch.cosine_similarity(d, d_rec, dim=-1).sum()

        d = d.unsqueeze(1).repeat(1, d_neg.shape[1], 1)
        d_rec = d_rec.unsqueeze(1).repeat(1, d_neg.shape[1], 1)

        phi_pos = torch.cosine_similarity(d, d_rec, dim=-1).flatten()
        phi_neg = torch.cosine_similarity(d, d_neg, dim=-1).flatten()

        margin_loss = nn.MarginRankingLoss(margin=margin, reduction="sum")
        # phi_neg: batch_size x num_neg
        L_1 = margin_loss(phi_pos, phi_neg, torch.ones(phi_pos.shape, device=self.device))# / d_neg.shape[1]

        return L_1 + L_2

    def batch_to_device(self, batch):
        return {key: item.to(self.device) for key, item in batch.items()}

    def train(self, data_loader, num_epochs: int = 2, lr: float = 1e-3):
        my_optim = torch.optim.Adam(params=self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0,
                                    amsgrad=False)

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=0.99)

        self.model.to(self.device)

        i = 0
        for epoch in range(num_epochs):
            for batch in tqdm(data_loader):
                labels = batch.pop("labels")
                batch = self.batch_to_device(batch)
                my_optim.zero_grad()

                d, _, d_reconstructed, d_neg = self.model(**batch)
                # loss
                loss = self.compute_loss(d, d_reconstructed, d_neg)
                loss.backward()
                # my_lr_scheduler.step(int(i / 50))
                my_optim.step()
                # my_lr_scheduler.step()
                i += 1

            if epoch % 10 == 0:
                my_lr_scheduler.step()
                print("Loss: ", loss.detach().cpu().numpy())
                ev = self.evaluate(data_loader=data_loader)

    def predict_cluster(self):
        pass

    def evaluate(self, data_loader):
        labels_true = []
        labels_pred = []
        for batch in tqdm(data_loader):
            labels_true.append(batch.pop("labels").detach().cpu().numpy())
            batch = self.batch_to_device(batch)
            with torch.no_grad():
                d, cluster_proba, d_reconstructed, d_neg = self.model(**batch)
                pred = cluster_proba.argmax(dim=-1)
                labels_pred.append(pred.detach().cpu().numpy())

        labels_true = np.concatenate(labels_true)
        labels_pred = np.concatenate(labels_pred)
        ev = full_evaluate(labels_true=labels_true, labels_pred=labels_pred)
        print(ev)
        return ev

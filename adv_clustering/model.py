import torch
import torch.nn as nn


def mask_emb_mult(samples, mask):
    samples = samples.transpose(-1, -2)
    mask = mask / mask.sum(axis=-1, keepdims=True)
    mask = mask.unsqueeze(-1)
    emb = torch.matmul(samples, mask)
    return emb.squeeze()


class ARL(nn.Module):
    def __init__(self, word_vecs, num_clusters: int = 300):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(word_vecs, freeze=False)
        self.centers = nn.Parameter(torch.empty((num_clusters, word_vecs.shape[1]), dtype=torch.float))
        nn.init.xavier_uniform_(self.centers.data)

    def forward(self, pos, pos_mask, neg, neg_mask):
        # batch_size x num_words x dim
        pos = self.embedding(pos).float()
        neg = self.embedding(neg).float()

        pos_mask = pos_mask.float()
        neg_mask = neg_mask.float()

        # batch_size x dim
        d = mask_emb_mult(pos, pos_mask)
        # batch_size x num_negatives x dim
        d_neg = mask_emb_mult(neg, neg_mask)

        # batch_size x num_clusters
        cluster_proba = torch.matmul(d, self.centers.T)
        cluster_proba = torch.softmax(cluster_proba, dim=1)

        # attention x /centers
        # batch_size x dim
        d_reconstructed = torch.matmul(cluster_proba, self.centers)

        return d, cluster_proba, d_reconstructed, d_neg

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class DistMult(nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()
        self.criterion = nn.Softplus()
        self.lmbda = args["lambda"]

    def _calc(self, h, t, r):
        return - torch.sum(h * t * r, -1)

    def loss(self, score, regul):
        return torch.mean(self.criterion(score * self.batch_y)) + self.lmbda * regul

    def forward(self, input):
        h, r, t, y = input
        self.batch_y = y
        score = self._calc(h, t, r)
        regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        return self.loss(score, regul)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()

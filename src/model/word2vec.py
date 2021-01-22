import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import logsigmoid

from utils import *


class SkipGram(nn.Module):
    def __init__(self, m1, norm):
        super(SkipGram, self).__init__()

    def forward(self, *input):
        emb_u, emb_v, emb_v_neg = input
        pos = torch.sum(emb_u * emb_v, dim=(1, 2))
        neg = torch.sum(emb_u * emb_v_neg, dim=2)
        return pos, neg

    def nce_loss(self, pos_dot, neg_dot, pos_log_k_negative_prob, neg_log_k_negative_prob, size_average=True, reduce=True):
        """
        https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf
        :param pos_dot:
        :param neg_dot:
        :param pos_log_k_negative_prob:
        :param neg_log_k_negative_prob:
        :param size_average:
        :param reduce:
        :return:
        """
        s_pos = pos_dot - pos_log_k_negative_prob
        s_neg = neg_dot - neg_log_k_negative_prob
        loss = - (torch.mean(logsigmoid(s_pos) + torch.sum(logsigmoid(-s_neg), dim=1)))

        if not reduce:
            return loss
        if size_average:
            return torch.mean(loss)
        return torch.sum(loss)

    def negative_sampling_loss(self, pos_dot, neg_dot, size_average=True, reduce=True):
        """
        :param pos_dot: The first tensor of SKipGram's output: (#mini_batches)
        :param neg_dot: The second tensor of SKipGram's output: (#mini_batches, #negatives)
        :param size_average:
        :param reduce:
        :return: a tensor has a negative sampling loss
        """
        loss = - (
                logsigmoid(pos_dot) + torch.sum(logsigmoid(-neg_dot), dim=1)
        )

        if not reduce:
            return loss
        if size_average:
            return torch.mean(loss)

        return torch.sum(loss)
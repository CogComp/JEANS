import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class MTransE(nn.Module):
    def __init__(self, args):
        super(MTransE, self).__init__()
        self.args = args
        self.m1 = args["m1"]
        self.norm = args["norm"]
        # self.criterion = nn.MarginRankingLoss(self.m1, False).cuda()

    def forward_KM_marginal(self, input):
        A_h_embs, A_rel_embs, A_t_embs, A_hn_embs, A_reln_embs, A_tn_embs = input
        A_pos = torch.norm(A_h_embs + A_rel_embs - A_t_embs, p=self.norm, dim=1)
        A_neg = torch.norm(A_hn_embs + A_reln_embs - A_tn_embs, p=self.norm, dim=1)
        tmp = A_pos + self.m1 - A_neg
        zero_pos = torch.FloatTensor([0]).expand_as(tmp)
        if self.args["cuda"]:
            zero_pos = zero_pos.cuda()
        self.triple_loss = torch.mean(torch.max(tmp, zero_pos))
        return self.triple_loss, torch.mean(A_pos), torch.mean(A_neg)

    def loss_edge(self, input):
        AM_rels1_embs, AM_rels2_embs = input
        rel1_norm = torch.norm(AM_rels1_embs, dim = -1)
        rel2_norm = torch.norm(AM_rels2_embs, dim = -1)
        loss = torch.mean(torch.abs(rel1_norm - rel2_norm))
        return loss

    def forward_KM_MarginRankingLoss(self, input):
        A_h_embs, A_rel_embs, A_t_embs, A_hn_embs, A_reln_embs, A_tn_embs = input
        A_pos = torch.norm(A_h_embs + A_rel_embs - A_t_embs, p=self.norm, dim=1)
        A_neg = torch.norm(A_hn_embs + A_reln_embs - A_tn_embs, p=self.norm, dim=1)
        self.loss = nn.MarginRankingLoss(self.m1)
        target = torch.ones(A_neg.size()).cuda()
        self.triple_loss = self.loss(A_pos, A_neg, target)
        return self.triple_loss, torch.mean(A_pos), torch.mean(A_neg)

    def forward_alignKM(self, input):
        A_h_embs, A_rel_embs, A_t_embs = input
        A_pos = torch.norm(A_h_embs + A_rel_embs - A_t_embs, self.norm, 1)
        loss = torch.nn.LogSigmoid(-A_pos)
        print(f"loss.shape: {loss.shape}")
        return - loss


    def forward_KM_limited(self, input):
        A_h_embs, A_rel_embs, A_t_embs, A_hn_embs, A_reln_embs, A_tn_embs = input
        pos_score = torch.norm(A_h_embs + A_rel_embs - A_t_embs, 2, 1)
        neg_score = torch.norm(A_hn_embs + A_reln_embs - A_tn_embs, 2, 1)
        tmp_pos = pos_score - self.args["lambda_1"]
        tmp_neg = self.args["lambda_2"] - neg_score
        zero_pos = torch.FloatTensor([0]).expand_as(tmp_pos)
        zero_neg = torch.FloatTensor([0]).expand_as(tmp_neg)
        if self.args["cuda"]:
            zero_pos = zero_pos.cuda()
            zero_neg = zero_neg.cuda()
        pos_loss = torch.mean(torch.max(tmp_pos, zero_pos))
        neg_loss = torch.mean(torch.max(tmp_neg, zero_neg))
        self.triple_loss = pos_loss + self.args["mu_1"] * neg_loss
        return self.triple_loss, torch.mean(pos_score), torch.mean(neg_score)

    def forward_AM(self, *input):
        AM_ent1_embs, AM_ent2_embs, mapping = input
        return mapping(AM_ent1_embs), AM_ent2_embs



    def loss_AM(self, *input):
        AM_ent1_embs_transformed, AM_ent2_embs = input
        self.AM_loss = torch.norm(AM_ent1_embs_transformed - AM_ent2_embs, p=self.norm, dim=-1)
        self.AM_loss = torch.mean(self.AM_loss)
        return self.AM_loss


















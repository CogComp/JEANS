import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import scipy
import scipy.linalg

class MUSE(nn.Module):
    def __init__(self, args):
        super(MUSE, self).__init__()

    def procrustes(self, *input):
        AM_ent1_embs, AM_ent2_embs, mapping = input
        A = AM_ent1_embs.data
        B = AM_ent2_embs.data
        W = mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
        return mapping

















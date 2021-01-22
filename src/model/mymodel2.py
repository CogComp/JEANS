import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import psutil
from torch.nn.parameter import Parameter
import time


# import faiss
# FAISS_AVAILABLE = True

sys.path.append("./")
from utils.utils import *
from model.MTransE import MTransE
from model.DistMult import DistMult
from model.SkipGram import SkipGram
from model.MUSE import MUSE
from model.GCN import GCN
from utils.test_funcs import *

logger = logging.getLogger(__name__)
ratio = 1

class mymodel(nn.Module):
    def __init__(self, args, args_SG, multiG):
        super(mymodel, self).__init__()
        self.args = args
        self.cuda_avl = args["cuda"]
        self.dim = args["dim"]
        self.num_ent1 = args["num_ent1"]
        self.num_ent2 = args["num_ent2"]

        # initialize embedding
        self.num_rels1, self.num_rels2 = args["num_rels1"], args["num_rels2"]
        if args["random_initialize"]:
            self.emb_ew1 = nn.Embedding(args["num_ew1"], self.dim)
            self.emb_ew2 = nn.Embedding(args["num_ew2"], self.dim)
            nn.init.xavier_normal(self.emb_ew1.weight)
            nn.init.xavier_normal(self.emb_ew2.weight)
        else:
            emb1, _, _ = load_vec(args["emb_ew1"])
            emb2, _, _ = load_vec(args["emb_ew2"])
            # self.embed1_w, self.embed2_w = nn.Parameter(torch.from_numpy(emb1[self.args["num_ent1"]:,:]).float()), nn.Parameter(torch.from_numpy(emb2[self.args["num_ent2"]:,:]).float())
            # self.embed1_e, self.embed2_e = nn.Parameter(torch.from_numpy(emb1[:self.args["num_ent1"],:]).float()), nn.Parameter(torch.from_numpy(emb2[:self.args["num_ent2"],:]).float())
            self.emb_ew1, self.num_ew1, self.dim = create_emb_layer(weights_matrix=emb1, max_norm = args["max_norm"])
            self.emb_ew2, self.num_ew2, self.dim = create_emb_layer(weights_matrix=emb2, max_norm = args["max_norm"])
            # self.emb_ew1, self.emb_ew2 = torch.from_numpy(emb1).float(), torch.from_numpy(emb2).float()
            # self.num_ew1, self.dim = self.emb_ew1.shape[0], self.emb_ew1.shape[1]
            # self.num_ew2, self.dim = self.emb_ew2.shape[0], self.emb_ew2.shape[1]
            del emb1, emb2


        if args["GCN"] is True:
            self.emb_rels1 = nn.Embedding(self.num_rels1, self.dim*2)
            self.emb_rels2 = nn.Embedding(self.num_rels2, self.dim*2)
        else:
            self.emb_rels1 = nn.Embedding(self.num_rels1, self.dim)
            self.emb_rels2 = nn.Embedding(self.num_rels2, self.dim)
        nn.init.xavier_normal(self.emb_rels1.weight)
        nn.init.xavier_normal(self.emb_rels2.weight)

        # GCN
        if args["GCN"] is True:
            self.GCN_model1 = GCN(nfeat=multiG.KG1.attr.shape[1], nhid=args["dim"], nclass=args["dim"], dropout=args["dropout"])
            self.GCN_model2 = GCN(nfeat=multiG.KG2.attr.shape[1], nhid=args["dim"], nclass=args["dim"], dropout=args["dropout"])
        self.emb_ew1_GCN = None
        self.emb_ew2_GCN = None

        # mapping
        if args["GCN"] is True:
            self.mapping = nn.Linear(self.dim*2, self.dim*2, bias=False)
            self.mapping.weight.data.copy_(torch.diag(torch.ones(self.dim*2)))
        else:
            self.mapping = nn.Linear(self.dim, self.dim, bias=False)
            self.mapping.weight.data.copy_(torch.diag(torch.ones(self.dim)))

            self.mapping_rel = nn.Linear(self.dim, self.dim, bias=False)
            self.mapping_rel.weight.data.copy_(torch.diag(torch.ones(self.dim)))

            for param in self.mapping_rel.parameters():
                param.requires_grad = False

            for param in self.mapping.parameters():
                param.requires_grad = False

        # model
        self.MtransE = MTransE(args)
        self.SkipGram = SkipGram()
        self.MUSE = MUSE(args)
        self.DistMult = DistMult(args)

        # data
        self.multiG = multiG

    def GCN4ent_embed(self,multiG):
        # logger.info("run GCN4ent_embed")
        # t0 = time.time()
        # features1, adj1, feature2, adj2 = multiG.KG1.attr, multiG.KG1.adj, multiG.KG2.attr, multiG.KG2.adj
        self.emb_ew1_GCN = self.GCN_model1(multiG.KG1.attr, multiG.KG1.adj)
        self.emb_ew2_GCN = self.GCN_model2(multiG.KG2.attr, multiG.KG2.adj)
        # self.emb_ew1.weight = ent_embed1
        # self.emb_ew2.weight = ent_embed2
        # logger.info(f"GCN generate entity embeddings, Time use: {time.time()-t0}")
        # t0 = time.time()
        # word_embed1 = self.emb_ew1.weight[ent_embed1.shape[0]:,:]
        # word_embed2 = self.emb_ew2.weight[ent_embed2.shape[0]:,:]
        # ew_embed1_tensor = torch.cat((ent_embed1, word_embed1),0)
        # ew_embed2_tensor = torch.cat((ent_embed2, word_embed2),0)
        # self.emb_ew1.weight = Parameter(ew_embed1_tensor)
        # self.emb_ew2.weight = Parameter(ew_embed2_tensor)
        # self.emb_ew1[:self.num_ent1,:], self.emb_ew2[:self.num_ent2,:] = ent_embed1, ent_embed2
        # self.emb_ew1 =torch.cat((ent_embed1, self.emb_ew1[self.num_ent1:,:]),0)
        # self.emb_ew2 = torch.cat((ent_embed2, self.emb_ew2[self.num_ent2:,:]),0)
        # logger.info(f"Combine embedding table, Time use: {time.time()-t0}")
        # del ent_embed1, ent_embed2

    def forward_KM_bootriple_t(self, KG_index, input):
        input = list(input)
        if self.cuda_avl:
            for i, v in enumerate(input.copy()):
                input[i] = v.cuda()
        pos_triples, neg_triples = input
        A_h_batch, A_rel_batch, A_t_batch = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        A_hn_batch, A_reln_batch, A_tn_batch = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        # for MTransE
        # emb_ent1, emb_ent12, emb_rel = (self.emb_ew1, self.emb_ew2, self.emb_rels1) if KG_index == 1 else (self.emb_ew2, self.emb_ew1, self.emb_rels1)
        A_h_embs = self.emb_ew2(A_h_batch)
        A_hn_embs = self.emb_ew2(A_hn_batch)
        A_t_embs = self.mapping(self.emb_ew1(A_t_batch))
        A_tn_embs = self.mapping(self.emb_ew1(A_tn_batch))

        # A_rel_embs = self.mapping_rel(self.emb_rels1(A_rel_batch))
        # A_reln_embs = self.mapping_rel(self.emb_rels1(A_reln_batch))
        A_rel_embs = self.emb_rels2(A_rel_batch)
        A_reln_embs = self.emb_rels2(A_reln_batch)
        return A_h_embs, A_rel_embs, A_t_embs, A_hn_embs, A_reln_embs, A_tn_embs


    def forward_KM_bootriple_h(self, KG_index, input):
        input = list(input)
        if self.cuda_avl:
            for i, v in enumerate(input.copy()):
                input[i] = v.cuda()
        pos_triples, neg_triples = input
        A_h_batch, A_rel_batch, A_t_batch = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        A_hn_batch, A_reln_batch, A_tn_batch = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        # for MTransE
        # emb_ent1, emb_ent12, emb_rel = (self.emb_ew1, self.emb_ew2, self.emb_rels1) if KG_index == 1 else (self.emb_ew2, self.emb_ew1, self.emb_rels1)
        A_h_embs = self.mapping(self.emb_ew1(A_h_batch))
        A_hn_embs = self.mapping(self.emb_ew1(A_hn_batch))
        A_t_embs = self.emb_ew2(A_t_batch)
        A_tn_embs = self.emb_ew2(A_tn_batch)

        # A_rel_embs = self.mapping_rel(self.emb_rels1(A_rel_batch))
        # A_reln_embs = self.mapping_rel(self.emb_rels1(A_reln_batch))
        A_rel_embs = self.emb_rels2(A_rel_batch)
        A_reln_embs = self.emb_rels2(A_reln_batch)
        return A_h_embs, A_rel_embs, A_t_embs, A_hn_embs, A_reln_embs, A_tn_embs


    def forward_KM(self, KG_index, input):
        input = list(input)
        if self.cuda_avl:
            for i, v in enumerate(input.copy()):
                input[i] = v.cuda()
        pos_triples, neg_triples = input
        A_h_batch, A_rel_batch, A_t_batch = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        A_hn_batch, A_reln_batch, A_tn_batch = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        # for MTransE
        emb_ent, emb_rel, emb_ent_GCN = (self.emb_ew1, self.emb_rels1, self.emb_ew1_GCN) if KG_index == 1 else (self.emb_ew2, self.emb_rels2, self.emb_ew2_GCN)
        # swj tensor EMBEDDING
        # A_h_embs = F.normalize(emb_ent[A_h_batch],2, 1)
        # A_hn_ememb_entbs = F.normalize(emb_ent[A_hn_batch],2, 1)
        # A_t_embs = F.normalize(emb_ent[A_t_batch],2, 1)
        # A_tn_embs = F.normalize(emb_ent[A_tn_batch],2, 1)

        # normalize
        A_h_embs = F.normalize(emb_ent(A_h_batch),2, 1)
        A_hn_embs = F.normalize(emb_ent(A_hn_batch),2, 1)
        A_t_embs = F.normalize(emb_ent(A_t_batch),2, 1)
        A_tn_embs = F.normalize(emb_ent(A_tn_batch),2, 1)
        # A_rel_embs = F.normalize(emb_rel(A_rel_batch),2, 1)
        # A_reln_embs = F.normalize(emb_rel(A_reln_batch),2, 1)

        # A_h_embs = emb_ent(A_h_batch)
        # A_hn_embs = emb_ent(A_hn_batch)
        # A_t_embs = emb_ent(A_t_batch)
        # A_tn_embs = emb_ent(A_tn_batch)
        A_rel_embs = emb_rel(A_rel_batch)
        A_reln_embs = emb_rel(A_reln_batch)
        if self.args["GCN"]:
            A_h_embs = emb_ent_GCN[A_h_batch]*ratio + A_h_embs
            A_hn_embs = emb_ent_GCN[A_hn_batch]*ratio + A_hn_embs
            A_t_embs = emb_ent_GCN[A_t_batch]*ratio + A_t_embs
            A_tn_embs = emb_ent_GCN[A_tn_batch]*ratio + A_tn_embs
            # A_h_embs = torch.cat((A_h_embs, emb_ent_GCN[A_h_batch]), dim = 1)
            # A_hn_embs = torch.cat((A_hn_embs, emb_ent_GCN[A_hn_batch]), dim = 1)
            # A_t_embs = torch.cat((A_t_embs, emb_ent_GCN[A_t_batch]), dim = 1)
            # A_tn_embs = torch.cat((A_tn_embs, emb_ent_GCN[A_tn_batch]), dim = 1)
            # torch.cat((e1_batch_emb, r1_batch_emb), dim=0)


        return A_h_embs, A_rel_embs, A_t_embs, A_hn_embs, A_reln_embs, A_tn_embs

    def forward_DistMult(self, KG_index, input):
        input = list(input)
        if self.cuda_avl:
            for i, v in enumerate(input.copy()):
                input[i] = v.cuda()
        y = []
        A_h_batch, A_rel_batch, A_t_batch, y = input
        # for MTransE
        emb_ent, emb_rel = (self.emb_ew1,self.emb_rels1) if KG_index == 1 else (self.emb_ew2, self.emb_rels2)
        # normalize
        # A_h_embs = F.normalize(emb_ent(A_h_batch),2, 1)
        # A_t_embs = F.normalize(emb_ent(A_t_batch),2, 1)
        # A_rel_embs = F.normalize(emb_rel(A_rel_batch),2, 1)

        A_h_embs = emb_ent(A_h_batch)
        A_t_embs = emb_ent(A_t_batch)
        A_rel_embs = emb_rel(A_rel_batch)

        return A_h_embs, A_rel_embs, A_t_embs, y


    def forward_AM(self, input, ht):
        input = list(input)
        if self.cuda_avl:
            for i, v in enumerate(input.copy()):
                input[i] = v.cuda()
        AM_ent1_batch, AM_ent2_batch = input
        if ht is True:
            AM_ent1_embs = self.emb_ew1(AM_ent1_batch)
            AM_ent2_embs = self.emb_ew2(AM_ent2_batch)
            if self.args["GCN"]:
                AM_ent1_embs = self.emb_ew1_GCN[AM_ent1_batch]*ratio + AM_ent1_embs
                AM_ent2_embs = self.emb_ew1_GCN[AM_ent2_batch]*ratio + AM_ent2_embs
                # AM_ent1_embs = torch.cat((AM_ent1_embs, self.emb_ew1_GCN[AM_ent1_batch]), dim = 1)
                # AM_ent2_embs = torch.cat((AM_ent2_embs, self.emb_ew1_GCN[AM_ent2_batch]), dim = 1)
            # swj
            # AM_ent1_embs = self.emb_ew1[AM_ent1_batch]
            # AM_ent2_embs = self.emb_ew2[AM_ent2_batch]
        else:
            AM_ent1_embs = self.emb_rels1(AM_ent1_batch)
            AM_ent2_embs = self.emb_rels2(AM_ent2_batch)
        # nomalize
        if ht is True:
            AM_ent1_embs = F.normalize(AM_ent1_embs, 2, 1)
            AM_ent2_embs = F.normalize(AM_ent2_embs, 2, 1)
        return AM_ent1_embs, AM_ent2_embs


    def forward_SG(self, lang_index, batch):
        inputs, contexts, negatives = batch
        num_minibatches = len(contexts)
        inputs = torch.LongTensor(list(map(int,inputs))).view(num_minibatches, 1)
        contexts = torch.LongTensor(list(map(int,contexts))).view(num_minibatches, 1)
        negatives = torch.LongTensor(negatives)
        if self.cuda_avl:
            inputs = inputs.cuda()
            contexts = contexts.cuda()
            negatives = negatives.cuda()
        emb = self.emb_ew1 if lang_index == 1 else self.emb_ew2

        in_vectors = emb(inputs)
        pos_context_vectors = emb(contexts)
        neg_context_vectors = emb(negatives)
        return in_vectors, pos_context_vectors, neg_context_vectors


    def test(self, KG1_all_ents, KG2_all_ents, selected_pairs=None, method = "nn", multiG = None, test_train = True):
        selected_pairs = torch.LongTensor(selected_pairs)
        if self.cuda_avl:
            selected_pairs = selected_pairs.cuda()
            KG1_all_ents = KG1_all_ents.cuda()
            KG2_all_ents = KG2_all_ents.cuda()
            if test_train is False:
                KG1_all_ents = KG1_all_ents[:len(multiG.aligned_KG1_ents_test)]
                KG2_all_ents = KG2_all_ents[:len(multiG.aligned_KG2_ents_test)]

        # test_ent1_vocab = selected_pairs[:, 0].clone()
        # test_ent2_vocab = selected_pairs[:, 1].clone()
        # test_ent1_vocab.sort()
        # test_ent2_vocab.sort()
        # swj
        # refs1_embed = F.normalize(self.mapping(self.emb_ew1[selected_pairs[:, 0]]), 2, 1)
        # refs2_embed = F.normalize(self.emb_ew2[KG2_all_ents], 2, 1)
        # refs1_embed = self.mapping(self.emb_ew1(selected_pairs[:, 0]))
        if not self.args["GCN"]:
            refs1_embed = self.mapping(self.emb_ew1.weight[KG1_all_ents])
            refs2_embed = self.emb_ew2(KG2_all_ents)
        else:
            refs1_embed = self.mapping(self.emb_ew1.weight[KG1_all_ents] + self.emb_ew1_GCN[KG1_all_ents]*ratio)
            refs2_embed = self.emb_ew2(KG2_all_ents) + self.emb_ew2_GCN[KG2_all_ents]*ratio
            # refs1_embed = self.mapping(torch.cat((self.emb_ew1.weight[KG1_all_ents], self.emb_ew1_GCN[KG1_all_ents]), dim = 1))
            # refs2_embed = torch.cat((self.emb_ew2(KG2_all_ents), self.emb_ew2_GCN[KG2_all_ents]), dim = 1)

        for m in method:
            precision_at_1 = eval_acc_mrr(refs1_embed, refs2_embed, selected_pairs, m, self.multiG.KG1.id2we, self.multiG.KG2.id2we, self.args["langs"])
        return precision_at_1

    def test_word(self, selected_pairs=None, method = "nn"):
        selected_pairs = torch.LongTensor(selected_pairs)
        if self.cuda_avl:
            selected_pairs = selected_pairs.cuda()
        # refs1_embed = F.normalize(self.mapping(self.emb_ew1[selected_pairs[:, 0]]), 2, 1)
        # refs2_embed = F.normalize(self.emb_ew2.data, 2, 1)
        # swj
        refs1_embed = self.mapping(self.emb_ew1.weight)
        refs2_embed = self.emb_ew2.weight
        # swj
        # refs2_embed = F.normalize(self.emb_ew2.data, 2, 1)
        eval_acc_word(refs1_embed, refs2_embed, selected_pairs, "nn")














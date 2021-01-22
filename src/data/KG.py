import pickle
import time
# from tqdm import tqdm
import sys
import scipy.sparse as sp
import numpy as np
import random
import torch

sys.path.append(".")
from utils.utils import *
from utils.utils_GCN import *

class KG(object):
    def __init__(self):
        self.train_triples = []
        self.train_triples_h = []
        self.train_triples_t = []
        self.train_triples_num = 0
        self.ents, self.rels = [], []
        self.we2id, self.id2we = {}, []
        self.ent2id, self.rel2id, self.id2ent, self.id2rel = {}, {}, [], []
        self.ent_num, self.rel_num = 0, 0

        # self.r_hs_train, self.r_ts_train, self.rel2htth = {}, {}, {}

    def load_data(self, dir, id, emb_path):
        # read ids
        id = id+1
        self.id2ent, self.ent2id = read_entid(f"{dir}/ent_ids_{id}")
        self.id2rel, self.rel2id = read_entid(f"{dir}/rel_ids_{id}")
        # load triple
        self.train_triples = list(read_triple_ids(f"{dir}/triples_{id}"))
        self.train_triples_h = list(read_triple_ids_h(f"{dir}/triples_{id}"))
        self.train_triples_t = list(read_triple_ids_t(f"{dir}/triples_{id}"))
        self.train_triples_num = len(self.train_triples)
        # load head tail
        heads = set([triple[0] for triple in self.train_triples])
        tails = set([triple[2] for triple in self.train_triples])
        self.ents = np.array(list(map(int,self.id2ent.keys())))
        self.tails = list(tails)
        self.heads = list(heads)
        self.ent_num = len(self.ent2id)
        self.rel_num = max(map(int,list(self.rel2id.values())))+1
        # load triple hrt dict
        self.generate_triple_dict()
        attrib_dir = dir.rsplit("/",1)[0]

        _, self.id2we, self.we2id = load_vec(emb_path)

        # # JAPE ONLY
        # attr = self.loadattr(f"{attrib_dir}/training_attrs_{id}")
        # self.attr = self.get_ae_input(attr)
        # adj = self.get_weighted_adj()
        # self.adj = self.preprocess_adj(adj)

    def preprocess_adj(self, adj):
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def loadattr(self, attrib_file):
        cnt = {}
        with open(attrib_file, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent_name = "dbpedia/" + th[0].split("/resource/")[-1]
                if ent_name not in self.ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1

        fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
        num_features = min(len(fre), 2000)
        self.attr2id = {}
        for i in range(num_features):
            self.attr2id[fre[i][0]] = i
        M = {}
        with open(attrib_file, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent_name = "dbpedia/" + th[0].split("/resource/")[-1]
                if ent_name in self.ent2id:
                    for i in range(1, len(th)):
                        if th[i] in self.attr2id:
                            M[(self.ent2id[ent_name], self.attr2id[th[i]])] = 1.0
        row = []
        col = []
        data = []
        for key in M:
            row.append(key[0])
            col.append(key[1])
            data.append(M[key])
        return sp.coo_matrix((data, (row, col)), shape=(len(self.id2ent), num_features))

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def normalize_adj(self,adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def get_ae_input(self, attr):
        attr = torch.FloatTensor(np.array(attr.todense()))
        return attr


    def get_weighted_adj(self):
        KG = self.train_triples
        r2f = func(KG)
        r2if = ifunc(KG)
        M = {}
        for tri in KG:
            if tri[0] == tri[2]:
                continue
            if (tri[0], tri[2]) not in M:
                M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
            else:
                M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
            if (tri[2], tri[0]) not in M:
                M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
            else:
                M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
        row = []
        col = []
        data = []
        for key in M:
            row.append(key[1])
            col.append(key[0])
            data.append(M[key])
        return sp.coo_matrix((data, (row, col)), shape=(self.ent_num, self.ent_num))

    def generate_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.train_triples:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set



    def corrupt_pos(self, t, pos, neg_sample = None, dic = None):
        hit = True
        res = None
        # t0 = time.time()
        res = np.copy(t)
        if neg_sample == "random":
            samp = random.choice(self.ents)
        elif neg_sample == "neighbour":
            candidates = dic.get(t[pos].tolist())
            # t1= time.time()
            # print("time 1: ", t1-t0)
            samp = random.choice(candidates)
            # t2 = time.time()
            # print("time 2: ", t2-t1)
        while samp == t[pos]:
            samp = random.choice(self.ents)
        res[pos] = samp
        # print("time 3: ", time.time() - t2)
        return res

    # bernoulli negative sampling
    def corrupt_batch(self, t_batch, neg_sample = None, dic = None, multi = None):
        neg_triples = []
        for t in t_batch:
            for num in range(multi):
                if np.random.uniform(0,1) < 0.5:
                    neg_triples.append(self.corrupt_pos(t, 2, neg_sample, dic))
                else:
                   neg_triples.append(self.corrupt_pos(t, 0, neg_sample, dic))
        return np.array(neg_triples)


    def save(self, filename):
        f = open(filename,'wb')
        #self.desc_embed = self.desc_embed_padded = None
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)
        print("===============\nCaution: need to reload desc embeddings.\n=====================")


if __name__ == '__main__':
    id2lang = {"1": "fr", "2": "en"}
    DIR = "../reference/JAPE/data/dbp15k/zh_en/0_3"
    KG1 = KG()
    KG2 = KG()
    id = 0
    KG1.load_data(dir = DIR, id = id, emb_path = f"/scratch/swj0419/joint_emb/data/wiki2vec/{id2lang[id]}wiki_300d_pro.txt")
    id = 1
    KG2.load_data(dir = DIR, id = id, emb_path = f"/scratch/swj0419/joint_emb/data/wiki2vec/{id2lang[id]}wiki_300d_pro.txt")







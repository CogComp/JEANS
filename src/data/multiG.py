import numpy as np
import pickle
import time
import sys
sys.path.append("./")
from data.KG import KG
import os
from collections import defaultdict
from utils.utils import *


class multiG(object):
    def __init__(self, KG1=None, KG2=None, lang1=None, lang2=None):
        if KG1 == None or KG2 == None:
            self.KG1 = KG()
            self.KG2 = KG()
        else:
            self.KG1 = KG1
            self.KG2 = KG2
        self.lang1 = lang1
        self.lang2 = lang2
        self.align_ents, self.align_rels, self.align_words, self.align_words_test = [],[],[], []
        self.n_align_ents, n_align_rels = 0, 0
        self.aligned_KG1_ents, self.aligned_KG2_ents = np.array([0]), np.array([0])
        self.aligned_KG1_rels, self.aligned_KG2_rels = np.array([0]), np.array([0])
        self.ent12, self.ent21 = defaultdict(set), defaultdict(set)
        self.rel12, self.rel21 = defaultdict(set), defaultdict(set)
        self.align_ents_test = []

    def load_align(self, KG_DIR):
        '''Load the dataset.'''
        self.n_align_ents = 0
        self.aligned_KG1_ents, self.aligned_KG2_ents = read_ref(os.path.join(KG_DIR, 'sup_ent_ids'))
        for e1, e2 in zip(self.aligned_KG1_ents, self.aligned_KG2_ents):
            self.align_ents.append((e1, e2))
            self.ent12[e1].add(e2)
            self.ent21[e2].add(e1)
        self.aligned_KG1_ents.sort()
        self.aligned_KG2_ents.sort()
        self.align_ents = np.array(self.align_ents)
        self.n_align_ents = len(self.align_ents)
        print("Loaded aligned entities from", KG_DIR, ". #pairs:", self.n_align_ents)

        # JAPE ONLY
        self.aligned_KG1_rels, self.aligned_KG2_rels = read_ref(os.path.join(KG_DIR, 'sup_rel_ids'))
        for e1, e2 in zip(self.aligned_KG1_rels, self.aligned_KG2_rels):
            self.align_rels.append((e1, e2))
            self.rel12[e1].add(e2)
            self.rel21[e2].add(e1)
        self.aligned_KG1_rels.sort()
        self.aligned_KG2_rels.sort()
        self.align_rels = np.array(self.align_rels)
        self.n_align_rels = len(self.align_rels)
        print("Loaded aligned rels from", KG_DIR, ". #pairs:", self.n_align_rels)

        self.align_ents_test = []
        self.aligned_KG1_ents_test, self.aligned_KG2_ents_test = read_ref(os.path.join(KG_DIR, 'ref_ent_ids'))
        for e1, e2 in zip(self.aligned_KG1_ents_test, self.aligned_KG2_ents_test):
            self.align_ents_test.append((e1, e2))
        self.aligned_KG1_ents_test.sort()
        self.aligned_KG2_ents_test.sort()
        self.align_ents_test = np.array(self.align_ents_test)
        print("Loaded aligned entities from", KG_DIR, ". #pairs:", len(self.align_ents_test))

    def load_bidict(self, DICT_DIR_train, DICT_DIR_test):
        self.aligned_w1, self.aligned_w2 = read_ref_word(DICT_DIR_train)
        for e1, e2 in zip(self.aligned_w1, self.aligned_w2):
            if e1 in self.KG1.we2id and e2 in self.KG2.we2id:
                self.align_words.append((self.KG1.we2id[e1], self.KG2.we2id[e2]))
        self.aligned_w1.sort()
        self.aligned_w2.sort()
        self.align_words = np.array(self.align_words)
        print("Loaded aligned words from", DICT_DIR_train, ". #pairs:", len(self.align_words))

        aligned_w1_test_word, aligned_w2_test_word = read_ref_word(DICT_DIR_test)
        self.aligned_w1_test, self.aligned_w2_test = [], []
        for e1, e2 in zip(aligned_w1_test_word, aligned_w2_test_word):
            if e1 in self.KG1.we2id and e2 in self.KG2.we2id:
                e1_id, e2_id = self.KG1.we2id[e1], self.KG2.we2id[e2]
                self.align_words_test.append((e1_id, e2_id))
                self.aligned_w1_test.append(e1_id)
                self.aligned_w2_test.append(e2_id)
            else:
                print("not found")
        self.aligned_w1_test.sort()
        self.aligned_w2_test.sort()
        self.align_words_test = np.array(self.align_words_test)
        self.aligned_w1_test = np.array(self.aligned_w1_test)
        self.aligned_w2_test = np.array(self.aligned_w2_test)
        print("Loaded aligned words test from", DICT_DIR_test, ". #pairs:", len(self.align_words_test))


    def num_align(self):
        return self.n_align

    # negative samples
    def corrupt_align_pos(self, align, pos):
        assert (pos in [0, 1])
        hit = True
        res = None
        while hit:
            res = np.copy(align)
            if pos == 0:
                samp = np.random.choice(self.KG1.ents)
                if samp not in self.ent21[align[1]]:
                    hit = False
                    res = np.array([samp, align[1]])
            else:
                samp = np.random.choice(self.KG2.ents)
                if samp not in self.ent12[align[0]]:
                    hit = False
                    res = np.array([align[0], samp])
        return res

    def corrupt_align(self, align, tar=None):
        pos = tar
        if pos == None:
            pos = np.random.randint(2)
        return self.corrupt_align_pos(align, pos)

    def corrupt_align_batch(self, batch_size, tar = None):
        np.random.seed(int(time.time()))
        return np.array([self.corrupt_align(a, tar) for a in batch_size])

    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)

    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)

if __name__ == '__main__':
    id2lang = ["fr", "en"]
    DIR = "/scratch/swj0419/joint_emb"
    KG_DIR = f"{DIR}/reference/JAPE/data/dbp15k/{id2lang[0]}_{id2lang[1]}/0_3"
    KG1 = KG()
    KG2 = KG()
    KG1.load_data(dir = KG_DIR, id = 0, emb_path = f"/scratch/swj0419/joint_emb/data/wiki2vec/{id2lang[0]}wiki_300d_pro.txt")
    KG2.load_data(dir = KG_DIR, id = 1, emb_path = f"/scratch/swj0419/joint_emb/data/wiki2vec/{id2lang[1]}wiki_300d_pro.txt")
    multiG1 = multiG(KG1, KG2, id2lang[0], id2lang[1])
    multiG1.load_align(KG_DIR = KG_DIR)
    multiG1.save(f"{DIR}/data/KG/MultiG_{id2lang[0]}_{id2lang[1]}")
    # 1/0
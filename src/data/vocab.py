import numpy as np
from collections import defaultdict
import pickle
import sys
sys.path.append("./")
import os
import random

from utils.utils import *
from data.negative_sampler import *
from tqdm import tqdm


class Dictionary(object):
    def __init__(
            self,
            word2id = None,
            replace_lower_freq_word=False,
            replace_word='<unk>'
    ):
        if word2id is not None:
            self.word2id = word2id
            self.id2word = {v:k for k, v in self.word2id.items()}
            self.word2freq = {k:0 for k in self.word2id.keys()}
        self.id2freq = None
        self.replace_lower_freq_word = replace_lower_freq_word
        self.replace_word = replace_word

    def count_freq(self, word):
        if word in self.word2id:
            self.word2freq[word] += 1
        elif word.lower() in self.word2id:
            self.word2freq[word.lower()] += 1

    def rebuild(self, min_count=5):
        self.id2freq = {self.word2id[w]: f for w, f in self.word2freq.items()}
        self.id2freq_table = np.array([self.id2freq[str(i)] for i in range(len(self.id2freq))])

    def __len__(self):
        return len(self.id2word)


class Corpus(object):
    def __init__(
            self,
            min_count=5,
            word2id = None,
            replace_lower_freq_word=False,
            replace_word='<unk>',
            bos_word='<bos>',
            eos_word='<eos>',
            save_dir=None,
            args = None
    ):
        self.dictionary = Dictionary(word2id, replace_lower_freq_word, replace_word)
        self.min_count = min_count
        self.num_words = 0
        self.num_vocab = 0
        self.num_docs = 0
        self.discard_table = None
        self.replace_lower_freq_word = replace_lower_freq_word
        self.replace_word = replace_word
        self.bos_word = bos_word
        self.eos_word = eos_word
        self.is_neg_loss = None
        self.args = args
        self.negative_sampler = None
        self.save_dir = save_dir

    def tokenize_from_file(self, path, lang, subsample):
        def _add_special_word(sentence):
            return self.bos_word + ' ' + sentence + ' ' + self.eos_word

        self.num_words = 0
        self.num_docs = 0

        # write
        fout = open(f"{self.save_dir}/{lang}_id.txt", "w")
        # fout_w = open(f"{self.save_dir}/{lang}.txt", "w")

        with open(path, "r", encoding="utf-8") as f:
            docs = []
            for l in tqdm(f):
                if subsample and ("dbpedia" not in l):
                    continue
                    # ratio = 0.7 if lang == "en" else 0.5
                    # if random.uniform(0,1) < ratio:
                    #     continue
                doc = []
                doc_w = []
                for word in _add_special_word(l.strip()).split():
                    self.dictionary.count_freq(word=word)
                    if word in self.dictionary.word2id:
                        doc.append(self.dictionary.word2id.get(word))
                        # doc_w.append(word)
                    elif word.lower() in self.dictionary.word2id:
                        doc.append(self.dictionary.word2id.get(word.lower()))
                        # doc_w.append(word.lower())

                if len(doc) > 1:
                    # docs.append(np.array(doc))
                    fout.write(" ".join(np.array(doc)))
                    fout.write("\n")
                    # fout_w.write(" ".join(np.array(doc_w)))
                    # fout_w.write("\n")
                    self.num_words += len(doc)
                    self.num_docs += 1
        self.dictionary.rebuild(min_count=self.min_count)
        self.num_vocab = len(self.dictionary)
        # self.docs= np.array(docs)
        fout.close()

        # build negative_sampler
        self.is_neg_loss = self.args["loss"] == 'neg'
        self.negative_sampler = NegativeSampler(
            frequency=self.dictionary.id2freq_table,
            negative_alpha=0.75,
            is_neg_loss=self.is_neg_loss,
            table_length=int(1e8),
        )
        self.log_k_prob = np.log(self.args["negative"] * self.negative_sampler.noise_dist) if not self.is_neg_loss else None
        return docs

    def build_discard_table(self, t=1e-4):
        # https://github.com/facebookresearch/fastText/blob/53dd4c5cefec39f4cc9f988f9f39ab55eec6a02f/src/dictionary.cc#L277
        tf = t / (self.dictionary.id2freq_table / self.num_words)
        self.discard_table = np.sqrt(tf) + tf

    def discard(self, word_id, rnd):
        return rnd.rand() > self.discard_table[word_id]

    def save(self, lang):
        f = open(f"{self.save_dir}/{lang}.pk",'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", f"{self.save_dir}/{lang}.pk")

    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)


if __name__ == '__main__':
    args_SG = {"min_count": 5,
               "negative": 5,
               "window": 5,
               "seed": 5,
               "batch_size": 128,
               "samples": 1e-3,
               "loss": "neg",
               }

    # Corpus
    id2lang = ["fr","en"]
    overwrite = True
    folder_name = "subsample_20k"
    DIR = "/scratch/swj0419/joint_emb"
    save_folder = f"{DIR}/data/SG_corpus/{folder_name}"
    subsample = True
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        print(f"made directory: {save_folder}")
    SG_corpus = []

    if overwrite is True:
        for i in range(len(id2lang)):
            corpus_file = f"{DIR}/data/wiki_db/{id2lang[i]}.txt"
            word2id = f"{DIR}/data/wiki2vec/{id2lang[i]}/{id2lang[i]}word2id.pk"
            corpus = Corpus(min_count=args_SG["min_count"], word2id=load_pk(word2id), save_dir=save_folder, args=args_SG)
            docs = corpus.tokenize_from_file(corpus_file, id2lang[i], subsample)
            corpus.build_discard_table(t=args_SG["samples"])
            corpus.save(id2lang[i])
            SG_corpus.append(corpus)
            # check dbents
            dump_path = f"{DIR}/reference/JAPE/data/dbp15k/fr_en/0_3_pro/id_url_ent_name_{i+1}"
            dbents = set([])
            for line in open(dump_path):
                line = line.rstrip('\n').split('\t')
                dbents.add(line[2])
            word_not_found = set([k for k in corpus.dictionary.word2id.keys() if corpus.dictionary.word2freq[k] == 0])
            remain_ents = dbents & word_not_found
            print("size of remain_ents: ", len(remain_ents))
    else:
        for i in range(len(id2lang)):
            corpus = Corpus()
            corpus.load(f"{save_folder}/{id2lang[i]}.pk")
            # check dbents
            dump_path = f"{DIR}/reference/JAPE/data/dbp15k/fr_en/0_3_pro/id_url_ent_name_{i+1}"
            dbents = set([])
            for line in open(dump_path):
                line = line.rstrip('\n').split('\t')
                dbents.add(line[2])
            word_not_found = set([k for k in corpus.dictionary.word2id.keys() if corpus.dictionary.word2freq[k] == 0])
            remain_ents = dbents & word_not_found
            SG_corpus.append(corpus)







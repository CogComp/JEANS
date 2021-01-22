import torch
import torch.nn as nn
import io
import numpy as np
import pickle

def create_emb_layer(weights_matrix, max_norm = 5, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, max_norm = max_norm)
    emb_layer.weight = nn.Parameter(torch.from_numpy(weights_matrix).float())
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


def load_vec(emb_path, nmax=-1):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def __add_dict_num(dic, k):
    if dic.get(k) is None:
        dic[k] = 1
    else:
        dic[k] += 1

def __add_dict_kv(dic, k, v):
    vs = dic.get(k, set())
    vs.add(v)
    dic[k] = vs

def read_entid(fpath):
    ents2id = {}
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            id = line[0]
            ent = "dbpedia/" + line[1].split("/", maxsplit=4)[-1]
            if ent in ents2id:
                print(ent)
            ents2id[ent] = id
    id2ents = {v:k for k, v in ents2id.items()}
    return id2ents, ents2id


def read_triple_ids(triples_file_path):
    if triples_file_path is None:
        return set()
    file = open(triples_file_path, 'r', encoding='utf8')
    triples = set()
    for line in file.readlines():
        ent_h, prop, ent_t = line.strip('\n').split('\t')
        triples.add((int(ent_h), int(prop), int(ent_t)))
    file.close()
    return triples

def read_triple_ids_h(triples_file_path):
    if triples_file_path is None:
        return set()
    file = open(triples_file_path, 'r', encoding='utf8')
    triples = set()
    for line in file.readlines():
        ent_h, prop, ent_t = line.strip('\n').split('\t')
        if int(ent_h) >= 10500:
            triples.add((int(ent_h), int(prop), int(ent_t)))
    file.close()
    return triples

def read_triple_ids_t(triples_file_path):
    if triples_file_path is None:
        return set()
    file = open(triples_file_path, 'r', encoding='utf8')
    triples = set()
    for line in file.readlines():
        ent_h, prop, ent_t = line.strip('\n').split('\t')
        if int(ent_t) >= 10500:
            triples.add((int(ent_h), int(prop), int(ent_t)))
    file.close()
    return triples

def load_pk(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d

def generate_words_from_doc(doc, num_processed_words):
    """
    this generator separates a long document into shorter documents
    :param doc: np.array(np.int), word_id list
    :param num_processed_words:
    :return: shorter words list: np.array(np.int), incremented num processed words: num_processed_words
    """
    new_doc = []
    for word_id in doc.strip().split():
        num_processed_words += 1
        # if corpus.discard(word_id=word_id, rnd=rnd):
        #     continue
        new_doc.append(word_id)
        if len(new_doc) >= 1000:
            yield np.array(new_doc), num_processed_words
            new_doc = []
    yield np.array(new_doc), num_processed_words


def read_ref(file_path):
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split()
        assert len(params) == 2
        e1 = int(params[0])
        e2 = int(params[1])
        refs.append(e1)
        reft.append(e2)
    assert len(refs) == len(reft)
    return refs, reft

def read_ref_word(file_path):
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split()
        assert len(params) == 2
        e1 = params[0]
        e2 = params[1]
        refs.append(e1)
        reft.append(e2)
    assert len(refs) == len(reft)
    return refs, reft


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def pair2newpairs(pairs):
    left_wids, right_wids = pairs[:, 0], pairs[:, 1]
    left_index, right_index = left_wids.argsort(), right_wids.argsort()
    return torch.stack((left_index, right_index), axis=1)
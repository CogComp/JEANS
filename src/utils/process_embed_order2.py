import random
import numpy as np
import pickle as pk
import re
import os
from utils import *


def process_embed_order(langid):
    id2lang = ["ja", "en"]
    lang1  = id2lang[0]
    lang2 = id2lang[1]
    DIR = "/home/swj0419/joint_emb"
    DATASET = "JAPE"
    out_dir = f"../data/wiki2vec/{id2lang[0]}_{id2lang[1]}_{DATASET}"
    if DATASET == "JAPE":
        dir = f"{DIR}/reference/JAPE/data/dbp15k/{lang1}_{lang2}/0_3_pro"
    elif DATASET == "wk3l_60k":
        dir = f"{DIR}/data/wk3l_60k/{lang1}_{lang2}"

    word4notfound = True
    remain_process = False
    BASELINE = False
    dim = 300
    overwrite = False


    if overwrite is True:
        for lang in ["1","2"]:
            dump_path = f"{dir}/id_url_ent_name_{langid+1}"
            # write dbents we need
            fout = open(dump_path, "w")
            for line in open(os.path.join(dir,f"ent_ids_{langid+1}"), "r", encoding="utf-8"):
                line = line.strip().split("\t")
                eid, url = line[0], line[1]
                ent =  line[1].split("/")[-1]
                ent_name = ent.split("(")[0].replace("_"," ").strip()
                ent = "dbpedia/" + ent
                fout.write(f"{eid}\t{url}\t{ent}\t{ent_name}\n")
            fout.close()

    # read dbents we need
    dump_path = f"{dir}/id_url_ent_name_{langid+1}"
    dbents2id = {}
    db_ents = set([])
    for line in open(dump_path, "r", encoding="utf-8"):
        # lower
        line = line.lower()
        line = line.rstrip('\n').split('\t')
        db_ents.add(line[2])
        dbents2id[line[2]] = line[0]

    # MUSE vocab
    MUSE_words = set()
    if os.path.exists(f"../data/MUSE/wiki.multi.{id2lang[langid]}.vec"):
        for line in open(f"../data/MUSE/wiki.multi.{id2lang[langid]}.vec", "r", encoding="utf-8"):
            line = line.rstrip('\n').split()
            MUSE_words.add(line[0].lower())
    else:
        for line in open(f"../data/MUSE/{lang2}-{lang1}.txt", "r", encoding="utf-8"):
            line = line.rstrip('\n').split()
            MUSE_words.add(line[1].lower())

    # load whole embedding if BASELINE is True
    if BASELINE is True:
        embeddings_f, id2word_f, word2id_f = load_vec(f"../data/wiki2vec/{id2lang[langid]}/{id2lang[langid]}wiki_20180420_300d.txt")

    # extract embedding
    ents = set([])
    words = set([])
    word2id = {}
    vectors = []
    found = 0
    word2id_wiki2vec = {}
    embeddings_wiki2vec = []
    with open(f"../data/wiki2vec/{id2lang[langid]}/{id2lang[langid]}wiki_20200701_{dim}d_ent1.txt", "r", encoding="utf-8") as f:
        #20200701, 20180420
        for index, line in enumerate(f):
            if len(word2id) != len(vectors):
                print(f"{len(word2id)}: not in word2id")
            if index == 0:
                continue
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            word = word.replace("ENTITY/", "dbpedia/")
            # lower
            word = word.lower()
            word2id_wiki2vec[word] = len(word2id_wiki2vec)
            if remain_process is True or word4notfound:
                embeddings_wiki2vec.append(vect)
            if word in word2id:
                continue # make sure not assign dublicate id
            elif word in db_ents:
                found += 1
                ents.add(word)
                if BASELINE:
                    entsegs = word.split("dbpedia/")[1]
                    entsegs = re.split('·|-|,|_|\+', entsegs)
                    ent_embed = []
                    for entseg in entsegs:
                        if entseg == "":
                            continue
                        if entseg in word2id_f:
                            embed = embeddings_f[word2id_f[entseg]]
                            ent_embed.append(embed)
                    if ent_embed == []:
                        print(word)
                    ent_embed = np.mean(np.array(ent_embed), axis=0)
                else:
                    ent_embed = vect
                word2id[word] = len(word2id)
                vectors.append(ent_embed)
            elif word in MUSE_words: # or random.uniform(0, 1) < 0.2
                words.add(word)
                word2id[word] = len(word2id)
                vectors.append(vect)

    # print(word)
    print("ents: ", len(ents), "words: ", len(words), "dbents2id", len(dbents2id))
    print("found: ", found, "db_ents: ", len(db_ents))

    # for remain ents
    remain_ents = db_ents.difference(ents)
    ent_still_notfound = len(remain_ents)
    for e in remain_ents:
        word2id[e] = len(word2id)
        # check again if the e in word2id_wiki2vec to save remain_ents2file
        e_pro = e.split("_(")[0]
        if remain_process is True and e_pro in word2id_wiki2vec:
            ent_still_notfound -= 1
            # print("check again: " , e, e_pro)
            vect = embeddings_wiki2vec[word2id_wiki2vec[e_pro]]
        # e_pro2 = e.split("_")[0]
        elif word4notfound is True:
            word = e.split("dbpedia/")[1]
            word = re.split('·|-|,|_|\+', word)
            ent_embed = []
            for entseg in word:
                if entseg == "":
                    continue
                if entseg in word2id_wiki2vec:
                    embed = embeddings_wiki2vec[word2id_wiki2vec[entseg]]
                    ent_embed.append(embed)
            if ent_embed == []:
                vect = np.random.normal(0, 1, dim)
            else:
                ent_still_notfound -= 1
                vect = np.mean(np.array(ent_embed), axis=0)
        else:
            vect = np.random.normal(0, 1, dim)
        vectors.append(vect)
    print(f"# ent_initial_notfound: {len(remain_ents)}, # ent_still_notfound: {ent_still_notfound}")
    id2word = {v: k for k, v in word2id.items()}
    vectors = np.vstack(vectors)

    # rearrange the index for db
    new_vectors = np.zeros((len(vectors),dim), dtype = float)
    new_word2id = {}
    for ent, id in dbents2id.items():
        entname = ent.split("/")[1]
        new_word2id[ent] = id

    dbents = list(dbents2id)
    dbentsids = set(dbents2id.values())
    idsremain = set(map(str,range(len(new_vectors)))).difference(set(map(str,dbentsids)))
    for i, v in enumerate(vectors):
        # for not ents
        if id2word[i] not in dbents:
            line_id = idsremain.pop()
            new_vectors[int(line_id)] = v
            new_word2id[id2word[i]] = line_id
        # for ents
        else:
            w = id2word[i]
            new_vectors[int(new_word2id[w])] = v

    new_id2word = {v: k for k, v in new_word2id.items()}
    print("check empty: ", len(idsremain))

    # write to output
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if remain_process is True and word4notfound is True:
        fout_name = f"{out_dir}/{id2lang[langid]}wiki_{dim}d_pro_4notfound/remain.txt"
    elif remain_process is True and word4notfound is False:
        fout_name = f"{out_dir}/{id2lang[langid]}wiki_{dim}d_pro_remain.txt"
    elif remain_process is False and word4notfound is False:
        fout_name = f"{out_dir}/{id2lang[langid]}wiki_{dim}d_pro_ent1.txt" # d_pro_ent1
    elif remain_process is False and word4notfound is True:
        fout_name = f"{out_dir}/{id2lang[langid]}wiki_{dim}d_pro_4notfound_ent1.txt" #d_pro_4notfound_ent1
    fout = open(fout_name, "w")
    fout.write(u"%i %i\n" % new_vectors.shape)
    for i in range(len(new_word2id)):
        try:
            fout.write(u"%s %s\n" % (new_id2word[str(i)], " ".join('%.5f' % x for x in new_vectors[i])))
        except:
            print(new_id2word[str(i)])
    fout.close()
    # dump word2id
    with open(f"{out_dir}/{id2lang[langid]}word2id.pk","wb") as f:
        pk.dump(new_word2id, f)

if __name__ == '__main__':
    # build dbpedia entity name
    langid = 0
    process_embed_order(langid)
    # langid = 1
    # process_embed_order(langid)
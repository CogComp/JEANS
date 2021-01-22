import random
import numpy as np
import pickle as pk

# build dbpedia entity name
langid = "1"
lang1 = "fr"
lang2 = "en"
id2lang = {"1":lang1, "2": lang2}
dir = f"../reference/JAPE/data/dbp15k/{lang1}_{lang2}/0_3_pro"

dim = 300
# dbents we need
dump_path = f"{dir}/id_url_ent_name_{langid}"
dbents2id = {}
db_ents = set([])
for line in open(dump_path):
    line = line.rstrip('\n').split('\t')
    db_ents.add(line[2])
    dbents2id[line[2]] = line[0]

# MUSE vocab
MUSE_words = set()
for line in open(f"../data/MUSE/wiki.multi.{id2lang[langid]}.vec"):
    line = line.rstrip('\n').split()
    MUSE_words.add(line[0].lower())

# extract embedding
ents = set([])
words = set([])
word2id = {}
vectors = []
found = 0
with open(f"../data/wiki2vec/{id2lang[langid]}/{id2lang[langid]}wiki_20180420_win10_300d.txt") as f: #{id2lang[langid]}wiki_20180420_300d.txt" # {id2lang[langid]}-fasttext-subsample.vec
    for index, line in enumerate(f):
        if index == 0:
            continue
        word, vect = line.rstrip().split(' ', 1)
        vect = np.fromstring(vect, sep=' ')
        word = word.replace("ENTITY/", "dbpedia/")

        if word in db_ents:
            found += 1
            ents.add(word)
            word2id[word] = len(word2id)
            vectors.append(vect)
        elif word in MUSE_words: # or random.uniform(0, 1) < 0.2
        # else:
            words.add(word)
            word2id[word] = len(word2id)
            vectors.append(vect)

# print(word)
print("ents: ", len(ents), "words: ", len(words), "dbents2id", len(dbents2id))
print("found: ", found, "db_ents: ", len(db_ents))

# for remain ents
remain_ents = db_ents.difference(ents)
for e in remain_ents:
    word2id[e] = len(word2id)
    vectors.append(np.random.normal(0, 1, dim))
id2word = {v: k for k, v in word2id.items()}
vectors = np.vstack(vectors)

# rearrange the index
new_vectors = np.zeros((len(vectors),dim), dtype = float)
new_word2id = {}
for ent, id in dbents2id.items():
    new_vectors[int(id)] = vectors[word2id[ent]]
    new_word2id[ent] = id

dbents = list(dbents2id)
dbentsids = set(dbents2id.values())
idsremain = set(map(str,range(len(new_vectors)))).difference(set(map(str,dbentsids)))
for i, v in enumerate(vectors):
    if id2word[i] not in dbents:
        line_id = idsremain.pop()
        new_vectors[int(line_id)] = v
        new_word2id[id2word[i]] = line_id
new_id2word = {v: k for k, v in new_word2id.items()}
print("check empty: ", len(idsremain))


# write to output
fout = open(f"../data/wiki2vec/{id2lang[langid]}/{id2lang[langid]}wiki_300d_pro.txt","w")
# fout = open(f"../data/wiki2vec/{id2lang[langid]}-fasttext-100d.txt","w")
fout.write(u"%i %i\n" % new_vectors.shape)
for i in range(len(new_word2id)):
    fout.write(u"%s %s\n" % (new_id2word[str(i)], " ".join('%.5f' % x for x in new_vectors[i])))
fout.close()
# dump word2id
with open(f"../data/wiki2vec/{id2lang[langid]}/{id2lang[langid]}word2id.pk","wb") as f:
    pk.dump(new_word2id, f)
import logging
import os
import torch
import shutil

from data.multiG import multiG
from data.KG import KG
from data.vocab import Corpus,Dictionary
from utils.utils import *
from trainer.Trainer import Trainer

DIR = "/scratch/swj0419/joint_emb"
id2lang = ["fr", "en"]

args_SG = {"min_count": 5,
        "negative": 5,
        "window": 5,
        "seed": 5,
        "batch_size": 1500,
        "samples": 1e-3,
        "loss": "neg",
        }

save_name = "test"
args = {"batch_size":256,
        "epochs": 50,
        "cuda": 1,
        "gpuid": 0,
        "lr": 0.001,
        "save_every_epoch": 10,
        "save_path": f"{DIR}/saved_model/{save_name}",
        "emb_ew1": f"{DIR}/data/wiki2vec/{id2lang[0]}wiki_300d_pro.txt",
        "emb_ew2": f"{DIR}/data/wiki2vec/{id2lang[1]}wiki_300d_pro.txt",
        "m1": 0.5,
        "norm": 2,
        "eval_per_epoch":1,
        "restore": False,
        "load_path":f"{DIR}/saved_model/{save_name}",
        "extra_dict": True
}

if not os.path.exists(args["save_path"]):
    os.mkdir(args["save_path"])
save_path = args["save_path"]
shutil.copyfile("train.py", f"{save_path}/train.py")
shutil.copyfile("trainer/Trainer.py", f"{save_path}/Trainer.py")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(f"{save_path}", "log"),'w+'),
        logging.StreamHandler()
    ])

# multiG
overwrite = False
if overwrite is True:
    KG_DIR = f"{DIR}/reference/JAPE/data/dbp15k/{id2lang[0]}_{id2lang[1]}/0_3"
    KG1 = KG()
    KG2 = KG()
    KG1.load_data(dir=KG_DIR, id=0, emb_path=f"{DIR}/data/wiki2vec/{id2lang[0]}wiki_300d_pro.txt")
    KG2.load_data(dir=KG_DIR, id=1, emb_path=f"{DIR}/data/wiki2vec/{id2lang[1]}wiki_300d_pro.txt")
    multiG1 = multiG(KG1, KG2, id2lang[0], id2lang[1])
    multiG1.load_align(KG_DIR=KG_DIR)
    multiG1.save(f"{DIR}/data/KG/MultiG_{id2lang[0]}_{id2lang[1]}")
else:
    multiG1 = multiG()
    multiG1.load(f"{DIR}/data/KG/MultiG_{id2lang[0]}_{id2lang[1]}")

# Corpus
subsample = True
folder_name = "subsample_20k"
save_folder = f"{DIR}/data/SG_corpus/{folder_name}"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
    print(f"made directory: {save_folder}")
SG_corpus = []

overwrite = False
if overwrite is True:
    for i in range(len(id2lang)):
        corpus_file = f"{DIR}/data/wiki_db/{id2lang[i]}.txt"
        word2id = f"{DIR}/data/wiki2vec/{id2lang[i]}word2id.pk"
        corpus = Corpus(min_count=args_SG["min_count"], word2id=load_pk(word2id), save_dir=save_folder, args=args_SG)
        docs = corpus.tokenize_from_file(corpus_file, id2lang[i], subsample)
        corpus.build_discard_table(t=args_SG["samples"])
        corpus.save(id2lang[i])
        SG_corpus.append(corpus)
        # check dbents
        dump_path = f"{DIR}/reference/JAPE/data/dbp15k/fr_en/0_3/id_url_ent_name_{i + 1}"
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
        SG_corpus.append(corpus)

if args["restore"] == False:
    this_trainer = Trainer(multiG1, SG_corpus[0], SG_corpus[1], args, args_SG)
else:
    this_model = torch.load(args["load_path"])
    this_trainer = Trainer(multiG1, SG_corpus[0], SG_corpus[1], args, args_SG, this_model, restore = True)

this_trainer.train_all_MTransE()

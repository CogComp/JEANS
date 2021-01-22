import logging
import os
import torch
import shutil
import argparse

from data.multiG import multiG
from data.KG import KG
from data.vocab import Corpus,Dictionary
from utils.utils import *
from trainer.Trainer import Trainer
# from utils_EmbedOrder import *


# DIR = "/scratch/swj0419/joint_emb"
# DIR = "/mnt/macniece/experiment_tmp/swj0419/joint_emb"
DIR = "/home/swj0419/joint_emb"

id2lang = ["fr", "en"]
dim = 300
DATASET = "JAPE"
if DATASET == "JAPE":
    KG_DIR = f"{DIR}/reference/JAPE/data/dbp15k/{id2lang[0]}_{id2lang[1]}/0_3_pro"
elif DATASET == "wk3l_60k":
    KG_DIR = f"{DIR}/data/wk3l_60k/{id2lang[0]}_{id2lang[1]}"

parser = argparse.ArgumentParser(description = 'args')
parser.add_argument('--langs', default="_".join(id2lang), type=str)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--norm', default=2, type=int)
parser.add_argument('--save_every_epoch', default=2, type=int)
parser.add_argument('--MUSE', default=True, type=bool)
parser.add_argument('--MUSE_per_epoch', default=1, type=int)
parser.add_argument('--eval_per_epoch', default=1, type=int)
parser.add_argument('--restore', default=False, type=bool)
parser.add_argument('--load_path', default=f"{DIR}/saved_model/test/model", type=str)
parser.add_argument('--save_path', default=f"{DIR}/saved_model/fr_en", type=str)
parser.add_argument('--max_norm', default=6, type=float, help = "norm_clip")

# process KG data index, output 0_3_pro
# parser.add_argument('--pro_kg_index', default=True, type=bool)
# parser.add_argument('--pretrained_overwrite', default=True, type=bool)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--dim', default=dim, type=int)
parser.add_argument('--random_initialize', default=False, type=bool)
parser.add_argument('--emb_ew1', default=f"{DIR}/data/wiki2vec/{id2lang[0]}_{id2lang[1]}_{DATASET}/{id2lang[0]}wiki_{dim}d_pro_4notfound.txt", type=str) # zh: d_pro_4notfound_ent1
parser.add_argument('--emb_ew2', default=f"{DIR}/data/wiki2vec/{id2lang[0]}_{id2lang[1]}_{DATASET}/{id2lang[1]}wiki_{dim}d_pro_4notfound.txt", type=str)
parser.add_argument('--model', default="TransE", type=str, help = "DistMult, TransE")
parser.add_argument('--GCN', default=False, type=bool)
parser.add_argument('--Skipgram', default=False, type=bool)
parser.add_argument('--dropout', type=float, default=0.5, help='GCN Dropout rate (1 - keep probability).')

parser.add_argument('--loss', default="marginal", type=str, help = "marginal, limited, MarginRankingLoss")
parser.add_argument('--m1', default=2.4, type=float, help = "pos > neg by m1 in marginal loss") #best:2.4
parser.add_argument('--neg_sample', default="neighbour", type=str, help = "random, neighbour")
parser.add_argument('--num_negs', default=5, type=int, help = "1 if random")
parser.add_argument('--epsilon', default=0.997, type=float, help = "trunc_ent_num") #number of ents where to select neg
parser.add_argument('--lambda', default=0.01, type=float, help = "dismult regularizer")
parser.add_argument('--lambda_1', default=0, type=float, help = "for pos score")
parser.add_argument('--lambda_2', default=4, type=float, help = "for neg score")
parser.add_argument('--extra_dict', default=False, type=bool)

parser.add_argument('--mu_1', default=20, type=float, help = "for weight of neg loss in limited")
parser.add_argument('--bootmethod', default="refinement", type=str, help = "refinement or bootea or ")
parser.add_argument('--bootstrap_epoch', default=20, type=int, help = "begin filter bootstrap seeds")
parser.add_argument('--threshold', default=0.6, type=float, help = "bootstrapping threshold, >th")
parser.add_argument('--topk', default=10, type=int, help = "bootea")

parser.add_argument('--bootriple_h', default=False, type=bool)
parser.add_argument('--bootriple_t', default=False, type=bool)
parser.add_argument('--set', default=False, type=bool)
parser.add_argument('--train_edge', default=True, type=bool)

parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_rank", type=int, default=0, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_method", type=str, default='invsm_beta_30', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--eval_method", type=str, default='invsm_beta_30', help="csls_knn_10, nn, invsm_beta_30")

args = parser.parse_args()
args = vars(args)

args_SG = {"min_count": 5,
        "negative": 5,
        "window": 5,
        "seed": 5,
        "batch_size": 1500,
        "samples": 1e-3,
        "loss": "neg",
        }

'''
save data
'''
if not os.path.exists(args["save_path"]):
    os.mkdir(args["save_path"])
save_path = args["save_path"]
shutil.copyfile("train_MUSE.py", f"{save_path}/train_MUSE.py")
shutil.copyfile("trainer/Trainer.py", f"{save_path}/Trainer.py")

'''
logging
'''
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(f"{save_path}", "log"),'w+'),
        logging.StreamHandler()
    ])

# '''
# process KG index to output pro
# '''
# if args["pro_kg_index"]:
#     newdir = KG_DIR + "_pro"
#     process_kgindex(KG_DIR, newdir)
#
# if args["pretrained_overwrite"]:
#     build_index4vocab(newdir, id2lang)
#



# Corpus
overwrite = False
subsample = True
folder_name = "smallest" # only sample sentences that contain dbpedia entity
save_folder = f"{DIR}/data/SG_corpus/{folder_name}"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
    print(f"made directory: {save_folder}")
SG_corpus = []

if args["Skipgram"] is True:
    if overwrite is True:
        for i in range(len(id2lang)):
            corpus_file = f"{DIR}/data/wiki_db/{id2lang[i]}.txt"
            word2id = f"{DIR}/data/wiki2vec/{id2lang[0]}_{id2lang[1]}_{DATASET}/{id2lang[i]}word2id.pk"
            corpus = Corpus(min_count=args_SG["min_count"], word2id=load_pk(word2id), save_dir=save_folder, args=args_SG)
            docs = corpus.tokenize_from_file(corpus_file, id2lang[i], subsample)
            corpus.build_discard_table(t=args_SG["samples"])
            corpus.save(id2lang[i])
            SG_corpus.append(corpus)
            # check dbents
            dump_path = f"{DIR}/reference/JAPE/data/dbp15k/fr_en/0_3_pro/id_url_ent_name_{i + 1}"
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
else:
    SG_corpus = [0,0]


'''
load multiG
'''
overwrite = True
if overwrite is True:
    DICT_DIR_train = f"{DIR}/data/dict/{id2lang[0]}/{id2lang[0]}-{id2lang[1]}.0-5000.txt"
    DICT_DIR_test = f"{DIR}/data/dict/{id2lang[0]}/{id2lang[0]}-{id2lang[1]}.5000-6500.txt"
    KG1 = KG()
    KG2 = KG()
    KG1.load_data(dir=KG_DIR, id=0, emb_path=args["emb_ew1"])
    KG2.load_data(dir=KG_DIR, id=1, emb_path=args["emb_ew2"])
    multiG1 = multiG(KG1, KG2, id2lang[0], id2lang[1])
    multiG1.load_align(KG_DIR=KG_DIR)
    multiG1.load_bidict(DICT_DIR_train = DICT_DIR_train, DICT_DIR_test = DICT_DIR_test)
    # multiG1.save(f"{DIR}/data/KG/{DATASET}/{id2lang[0]}_{id2lang[1]}")
else:
    multiG1 = multiG()
    multiG1.load(f"{DIR}/data/KG/{DATASET}/{id2lang[0]}_{id2lang[1]}")

if args["restore"] == False:
    this_trainer = Trainer(multiG1, SG_corpus[0], SG_corpus[1], args, args_SG)
else:
    this_model = torch.load(args["load_path"])
    this_trainer = Trainer(multiG1, SG_corpus[0], SG_corpus[1], args, args_SG, this_model, restore = True)

this_trainer.train_all()


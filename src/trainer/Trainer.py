import pickle
import time
# from tqdm import tqdm
import sys
import torch
import torch.utils.data as Data
import numpy as np
import logging
from torch.optim.lr_scheduler import StepLR

from data.vocab import Dictionary, Corpus
from data.negative_sampler import NegativeSampler
from utils.utils import *
from utils.test_funcs import *
from utils.train_bp import bootstrapping, generate_newly_triples, check_alignment
from utils.dico_builder import build_dictionary, normalize_embed
from model.mymodel2 import mymodel




logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, multiG_data, SG_corpus1, SG_corpus2, args, args_SG, model=None, restore=False):
        self.batch_size = args["batch_size"]
        self.epochs = args["epochs"]
        self.cuda = args["cuda"]
        self.device = args["gpuid"]
        self.optimizer = None
        self.lr = args["lr"]
        self.save_every_epoch = args["save_every_epoch"]
        self.save_path = args["save_path"]
        self.precision_at_1 = 0
        self.eval_method = args["eval_method"].split(",")

        # data
        self.multiG = multiG_data
        self.SG_corpus1 = SG_corpus1
        self.SG_corpus2 = SG_corpus2
        args["num_rels1"], args["num_rels2"] = self.multiG.KG1.rel_num, self.multiG.KG2.rel_num
        args["num_ew1"], args["num_ew2"] = len(self.multiG.KG1.we2id), len(self.multiG.KG2.we2id)
        args["num_ent1"], args["num_ent2"] = self.multiG.KG1.ent_num, self.multiG.KG2.ent_num

        # args
        self.args, self.args_SG = args, args_SG
        self.trunc_ent_num = int(len(self.multiG.KG1.ents) * (1 - self.args["epsilon"]))
        logger.info(f"trunc_ent_num:  {self.trunc_ent_num}")

        # model
        self.model = mymodel(args, args_SG, self.multiG) if restore is False else model

        if self.cuda:
            torch.cuda.set_device(self.device)
            self.model.cuda()

        # load adj matrix if GCN
        if args["GCN"] is True:
            if self.cuda:
                self.multiG.KG1.attr = self.multiG.KG1.attr.cuda()
                self.multiG.KG2.attr = self.multiG.KG2.attr.cuda()
                self.multiG.KG1.adj = self.multiG.KG1.adj.cuda()
                self.multiG.KG2.adj = self.multiG.KG2.adj.cuda()

        # loader
        self.loader_KG1 = Data.DataLoader(dataset = np.array(self.multiG.KG1.train_triples), shuffle = True, batch_size = self.batch_size)
        self.loader_KG2 = Data.DataLoader(dataset = np.array(self.multiG.KG2.train_triples), shuffle = True, batch_size = self.batch_size)
        self.loader_KG1_h = Data.DataLoader(dataset = np.array(self.multiG.KG1.train_triples_h), shuffle = True, batch_size = self.batch_size)
        self.loader_KG1_t = Data.DataLoader(dataset = np.array(self.multiG.KG1.train_triples_t), shuffle = True, batch_size = self.batch_size)
        self.loader_KG2_h = Data.DataLoader(dataset = np.array(self.multiG.KG2.train_triples_h), shuffle = True, batch_size = self.batch_size)
        self.loader_KG2_t = Data.DataLoader(dataset = np.array(self.multiG.KG2.train_triples_t), shuffle = True, batch_size = self.batch_size)

        self.multiG.align_words = np.vstack((self.multiG.align_words, self.multiG.align_words_test))
        self.multiG.align_words = np.vstack((self.multiG.align_words, self.multiG.align_words_test))
        align_seeds = np.vstack((self.multiG.align_ents, self.multiG.align_words)) if self.args["extra_dict"] is True else np.array(self.multiG.align_ents)
        self.loader_multiG_ents = Data.DataLoader(dataset=align_seeds, shuffle=True, batch_size=self.batch_size)
        self.loader_multiG_rels = Data.DataLoader(dataset = np.array(self.multiG.align_rels), shuffle = True, batch_size = self.batch_size)
        self.loader_SG1 = self.generate_SG_batch(1)
        self.loader_SG2 = self.generate_SG_batch(2)

        # bootstrapping
        self.dico = {}
        self.labeled_align = set()
        self.aligned_e1 = set()
        self.aligned_e2 = set()


    '''
    Generate Batch
    '''
    # KG
    def gen_TransE_batch(self, loader, KG, nbours):
        for step, pos_triples in enumerate(loader):
            # t0 = time.time()
            neg_triples = torch.LongTensor(KG.corrupt_batch(pos_triples, self.args["neg_sample"], nbours, multi = self.args["num_negs"]))
            pos_triples_new = []
            for i in pos_triples:
                pos_triples_new += [i]*self.args["num_negs"]
            pos_triples_new = torch.stack(pos_triples_new)
            # logger.info(f"generate neg batch, Time use: {time.time()-t0}")
            yield pos_triples_new, neg_triples

    def gen_DistMult_batch(self, loader, KG, forever=False, shuffle=True):
        for step, batch_t in enumerate(loader):
            t0 = time.time()
            neg_batch_t = torch.LongTensor(KG.corrupt_batch(batch_t.numpy()))
            h_batch, r_batch, t_batch = batch_t[:, 0], batch_t[:, 1], batch_t[:, 2]
            y = torch.ones(len(h_batch))
            neg_h_batch, neg_t_batch = neg_batch_t[:, 0], neg_batch_t[:, 2]
            y = torch.cat((y, -torch.ones(len(neg_h_batch))))
            # logger.info(f"generate 1 KM batch: {time.time() - t0}s")
            yield torch.cat((h_batch, neg_h_batch)), torch.cat((r_batch, r_batch)), torch.cat((t_batch, neg_t_batch)), y

    def gen_AM_batch(self, loader, multiG, forever=False, shuffle=True):
        for step, batch_t in enumerate(loader):
            neg_batch_t = torch.LongTensor(multiG.corrupt_align_batch(batch_t.numpy()))
            e1_batch, e2_batch, e1_nbatch, e2_nbatch = batch_t[:, 0], batch_t[:, 1], neg_batch_t[:, 0], neg_batch_t[:, 1]
            yield e1_batch, e2_batch, e1_nbatch, e2_nbatch

    # skip gram
    def generate_SG_batch(self, lang_index):
        rnd = np.random.RandomState(self.args_SG["seed"])
        inputs, contexts = [], []
        SG_corpus = self.SG_corpus1 if lang_index == 1 else self.SG_corpus2
        # read file:
        doc_fpath = f"{self.multiG.lang1}_id.txt" if lang_index == 1 else f"{self.multiG.lang2}_id.txt"
        # print(f"{SG_corpus.save_dir}/{doc_fpath}")
        fin = open(f"{SG_corpus.save_dir}/{doc_fpath}","r")
        num_processed_words = 0
        for sentence in fin:
            # if "dbpedia" not in sentence and
            for doc, num_processed_words in generate_words_from_doc(doc=sentence, num_processed_words=num_processed_words):
                doclen = len(doc)
                dynamic_window_sizes = rnd.randint(low=1, high=self.args_SG["window"] + 1, size=doclen)
                for (position, (word_id, dynamic_window_size)) in enumerate(zip(doc, dynamic_window_sizes)):
                    begin_pos = max(0, position - dynamic_window_size)
                    end_pos = min(position + dynamic_window_size, doclen - 1) + 1
                    for context_position in range(begin_pos, end_pos):
                        if context_position == position:
                            continue
                        contexts.append(doc[context_position])
                        inputs.append(word_id)
                        if len(inputs) >= self.args_SG["batch_size"]:
                            negatives = SG_corpus.negative_sampler.sample(k=self.args_SG["negative"], rnd=rnd, exclude_words=contexts)
                            yield inputs, contexts, negatives
                            inputs.clear()
                            contexts.clear()
        fin.close()
    '''
    trainer
    '''
    def train_all(self):
        # model
        params_to_opt = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(params=params_to_opt, lr=self.lr)
        scheduler = StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.model.train()

        for epoch in range(self.epochs):
            t0 = time.time()
            if epoch == 0:
                self.train_ALIGN(epoch)  # ALIGN
                self.test_entityAlign(epoch)  # evaluation
            else:
                self.train_KG(epoch) # KG
                if self.args["train_edge"]:
                    self.train1epoch_edge(epoch)
                if self.args["Skipgram"]:
                    self.train_SG(epoch) # SG
                self.train_ALIGN(epoch) # ALIGN
                self.test_entityAlign(epoch) # evaluation

            logger.info("Time use for %i epoch: %s" % (epoch, time.time() - t0))
            if (epoch + 1) % self.save_every_epoch == 0:
                self.save(self.save_path)
                logger.info("saved in file: %s" % self.save_path)

        self.save(self.save_path)
        logger.info("saved in file: %s" % self.save_path)
        logger.info("Done")


    def train_KG(self, epoch):
        # negative samples
        if self.args["neg_sample"] == "neighbour":
            t0 = time.time()
            self.nbours1 = generate_neighbours_multi_embed(self.args, self.model.emb_ew1, self.multiG.KG1.ents, self.trunc_ent_num)
            self.nbours2 = generate_neighbours_multi_embed(self.args, self.model.emb_ew2, self.multiG.KG2.ents, self.trunc_ent_num)
            logger.info("generate neighbours, Time_use: %s" % (time.time() - t0))
            # model
        else:
            self.nbours1, self.nbours2 = None, None
        if self.args["model"] == "DistMult":
            self.train1epoch_DistMult(1, self.loader_KG1, self.multiG.KG1, epoch)
            self.train1epoch_DistMult(2, self.loader_KG2, self.multiG.KG2, epoch)
        elif self.args["model"] == "TransE":
            self.train1epoch_KM(1, self.loader_KG1, self.multiG.KG1, epoch)
            self.train1epoch_KM(2, self.loader_KG2, self.multiG.KG2, epoch)

        if self.args["langs"] != "fr":
            self.train1epoch_KM_bootriple_h(2, self.loader_KG2_h, self.multiG.KG2, epoch)

    def train_SG(self, epoch):
        self.train_batch_SG(self.loader_SG1, 1)
        self.train_batch_SG(self.loader_SG2, 2)

    def train_ALIGN(self, epoch):
        if self.args["MUSE"] is True:
            if epoch % self.args["MUSE_per_epoch"] == 0:
                self.train1epoch_MUSE(epoch, ht=True)
        else:
            self.train1epoch_AM(self.loader_multiG_ents, self.multiG, epoch, ht=True)
            self.train1epoch_AM(self.loader_multiG_rels, self.multiG, epoch, ht=False)

    def train1epoch_MUSE(self, epoch, ht):
        t0 = time.time()
        seeds = np.vstack((self.multiG.align_ents, self.multiG.align_words)) if self.args["extra_dict"] is True else self.multiG.align_ents
        rel_seeds = self.multiG.align_rels
        batch = torch.LongTensor(seeds)
        batch_rels= torch.LongTensor(rel_seeds)
        e1_batch, e2_batch = batch[:,0], batch[:,1]
        r1_batch, r2_batch = batch_rels[:,0], batch_rels[:,1]
        batch = e1_batch, e2_batch
        batch_rels = r1_batch, r2_batch
        if self.args["GCN"]:
            self.model.GCN4ent_embed(self.multiG)
        e1_batch_emb, e2_batch_emb = self.model.forward_AM(batch, ht = True)
        r1_batch_emb, r2_batch_emb = self.model.forward_AM(batch_rels, ht = False)
        er1_batch_emb, er2_batch_emb = torch.cat((e1_batch_emb, r1_batch_emb), dim = 0), torch.cat((e2_batch_emb, r2_batch_emb), dim = 0)
        # del e1_batch_emb, e2_batch_emb, r1_batch_emb, r2_batch_emb
        if epoch <= 2:
            self.model.mapping = self.model.MUSE.procrustes(er1_batch_emb, er2_batch_emb, self.model.mapping)
            # self.model.mapping_rel = self.model.MUSE.procrustes(r1_batch_emb, r2_batch_emb, self.model.mapping_rel)
        if self.args["bootmethod"] == "bootea" or self.args["bootmethod"] == "refinement":
            if self.args["bootmethod"] == "bootea":
                self.labeled_align = bootstrapping(self.args, self.model.emb_ew1, self.model.emb_ew2, self.multiG.aligned_KG1_ents_test, self.multiG.aligned_KG2_ents_test, self.model.mapping, self.labeled_align)
                seeds = np.vstack((seeds, np.array([list(pair) for pair in self.labeled_align])))
            else:
                src_emb, tgt_emb = normalize_embed(self.args, self.model.mapping, self.model.emb_ew1,
                                                   self.model.emb_ew2, self.multiG.aligned_KG1_ents_test,
                                                   self.multiG.aligned_KG2_ents_test)
                if epoch < self.args["bootstrap_epoch"]:
                    self.aligned_e1, self.aligned_e2 = set(), set()
                dico = build_dictionary(src_emb, tgt_emb, self.args, self.aligned_e1, self.aligned_e2).detach().cpu().numpy()
                self.aligned_e1, self.aligned_e2 = self.aligned_e1 | set(dico[:,0].tolist()), self.aligned_e2 | set(dico[:,1].tolist())
                new_dico =  np.array(list(set(map(tuple,dico)).difference(set(map(tuple,self.dico)))))
                check_alignment(new_dico, len(self.multiG.aligned_KG1_ents_test))
                check_alignment(dico, len(self.multiG.aligned_KG1_ents_test))
                self.dico = dico
                seeds = np.vstack((seeds, self.dico))
            batch = torch.LongTensor(seeds)
            e1_batch, e2_batch = batch[:, 0], batch[:, 1]
            batch = e1_batch, e2_batch
            e1_batch_emb, e2_batch_emb = self.model.forward_AM(batch, ht=True)
            self.model.mapping = self.model.MUSE.procrustes(e1_batch_emb, e2_batch_emb, self.model.mapping)

        if self.args["set"]:
            batch = torch.LongTensor(seeds)
            e1_batch, e2_batch = batch[:, 0], batch[:, 1]
            with torch.no_grad():
                self.model.emb_ew2.weight[e2_batch.cuda()] = self.model.mapping(self.model.emb_ew1(e1_batch.cuda()))
        logger.info("MUSE of %d epoch, Time_use: %s" % (epoch, time.time() - t0))


    def train_alignKM(self, seeds):
        ents1, ents2 = seeds[:,0], seeds[:, 1]
        new_211_tris, new_112_tris, new_221_tris, new_122_tris = [], [], [], []
        for i in range(len(ents1)):
            new_211, new_112, new_221, new_122 = generate_newly_triples(ents1[i], ents2[i], self.multiG.KG1, self.multiG.KG2)
            new_211_tris, new_112_tris, new_221_tris, new_122_tris = new_211_tris+new_211, new_112_tris+new_112, new_221_tris+new_221, new_122_tris+new_122

    def train1epoch_edge(self, epoch):
        t0 = time.time()
        rel_seeds = self.multiG.align_rels
        batch_rels = torch.LongTensor(rel_seeds)
        r1_batch, r2_batch = batch_rels[:, 0], batch_rels[:, 1]
        batch_rels = r1_batch, r2_batch
        r1_batch_emb, r2_batch_emb = self.model.forward_AM(batch_rels, ht=False)
        loss = self.model.MtransE.loss_edge((r1_batch_emb, r2_batch_emb))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.info("Edge Loss of %d epoch, loss: %s, Time_use: %s" % (epoch, id, time.time() - t0))


    def train1epoch_KM(self, KG_index, loader, KG, epoch):
        this_loss = []
        this_neg, this_pos = [], []
        t0 = time.time()
        nbours = self.nbours1 if KG_index == 1 else self.nbours2

        for id, batch in enumerate(self.gen_TransE_batch(loader, KG, nbours)):
            if self.args["GCN"]:
                self.model.GCN4ent_embed(self.multiG)
            batch_emb = self.model.forward_KM(KG_index, batch)
            if self.args["loss"] == "marginal":
                loss, pos, neg = self.model.MtransE.forward_KM_marginal(batch_emb)
            elif self.args["loss"] == "limited":
                loss, pos, neg = self.model.MtransE.forward_KM_limited(batch_emb)
            elif self.args["loss"] == "MarginRankingLoss":
                loss, pos, neg = self.model.MtransE.forward_KM_MarginRankingLoss(batch_emb)
            this_loss.append(loss.cpu().data)
            this_pos.append(pos.cpu().data)
            this_neg.append(neg.cpu().data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        l = np.mean(this_loss)
        logger.info("KM Loss of %d epoch, # batch: %s, loss: %s, Time_use: %s" % (epoch, id, l, time.time() - t0))
        logger.info("pos score: %s, neg score: %s" % (np.mean(this_pos), np.mean(this_neg)))

    def train1epoch_KM_bootriple_h(self, KG_index, loader, KG, epoch):
        this_loss = []
        this_neg, this_pos = [], []
        t0 = time.time()
        nbours = self.nbours1 if KG_index == 1 else self.nbours2

        for id, batch in enumerate(self.gen_TransE_batch(loader, KG, nbours)):
            # set directly
            if self.args["GCN"]:
                self.model.GCN4ent_embed(self.multiG)
            batch_emb = self.model.forward_KM_bootriple_h(KG_index, batch)
            if self.args["loss"] == "marginal":
                loss, pos, neg = self.model.MtransE.forward_KM_marginal(batch_emb)
            elif self.args["loss"] == "limited":
                loss, pos, neg = self.model.MtransE.forward_KM_limited(batch_emb)
            elif self.args["loss"] == "MarginRankingLoss":
                loss, pos, neg = self.model.MtransE.forward_KM_MarginRankingLoss(batch_emb)
            this_loss.append(loss.cpu().data)
            this_pos.append(pos.cpu().data)
            this_neg.append(neg.cpu().data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        l = np.mean(this_loss)
        logger.info("KM Loss of %d epoch bootriple, # batch: %s, loss: %s, Time_use: %s" % (epoch, id, l, time.time() - t0))
        logger.info("pos score: %s, neg score: %s" % (np.mean(this_pos), np.mean(this_neg)))


    def train1epoch_DistMult(self, KG_index, loader, KG, epoch):
        this_loss = []
        t0 = time.time()
        for id, batch in enumerate(self.gen_DistMult_batch(loader, KG)):
            self.optimizer.zero_grad()
            batch_emb  = self.model.forward_DistMult(KG_index, batch)
            loss = self.model.DistMult.forward(batch_emb)
            this_loss.append(loss.cpu().data)
            loss.backward()
            self.optimizer.step()
        l = np.mean(this_loss)
        logger.info("DistMult Loss of %d epoch, # batch: %s, loss: %s, Time_use: %s" % (epoch, id, l, time.time() - t0))



    def train1epoch_AM(self, loader, MultiG, epoch, ht):
        this_loss = []
        t0 = time.time()
        for id, batch in enumerate(self.gen_AM_batch(loader, MultiG)):
            self.optimizer.zero_grad()
            e1_batch, e2_batch, e1_nbatch, e2_nbatch = batch
            batch = e1_batch, e2_batch
            AM_ent1_embs, AM_ent2_embs = self.model.forward_AM(batch, ht)
            AM_ent1_embs_transformed, AM_ent2_embs = self.model.MtransE.forward_AM(AM_ent1_embs, AM_ent2_embs, self.model.mapping)
            loss = self.model.MtransE.loss_AM(AM_ent1_embs_transformed, AM_ent2_embs)
            this_loss.append(loss.cpu().data)
            loss.backward()
            self.optimizer.step()
        l = np.mean(this_loss)
        logger.info("AM Loss of %d epoch, #batch: %s, loss: %s, Time_use: %s" % (epoch, id, l, time.time() - t0))

        # SG

    def train_batch_SG(self, loader_SG, lang_index):
        this_loss = []
        t0 = time.time()
        loader = loader_SG
        for bid in range(10000):
            batch = next(loader)
            self.optimizer.zero_grad()
            batch = self.model.forward_SG(lang_index, batch)
            pos, neg = self.model.SkipGram.forward(batch)
            loss = self.model.SkipGram.negative_sampling_loss(pos, neg)
            loss_cpu = loss.cpu().data.numpy()
            this_loss.append(loss_cpu)
            loss.backward()
            self.optimizer.step()
            l = np.mean(this_loss)
        logger.info("SkipGram Loss of %d bid, loss: %s, Time_use: %s" % (bid, l, time.time() - t0))

    def test_entityAlign(self, epoch):
        if epoch % self.args["eval_per_epoch"] == 0:
            logger.info("=============eval_acc_mrr begin==================")
            t0 = time.time()
            self.model.eval()

            with torch.no_grad():
                KG1_ents = torch.LongTensor(self.multiG.KG1.ents)
                KG2_ents = torch.LongTensor(self.multiG.KG2.ents)
                tmp_precision = self.model.test(KG1_ents, KG2_ents[0:len(self.multiG.align_ents_test)], self.multiG.align_ents_test, self.eval_method, multiG = self.multiG, test_train = False)
                if tmp_precision > self.precision_at_1 - 1:
                    self.precision_at_1 = tmp_precision
                else:
                    sys.exit("accuracy declined")
                logging.info("Test word translation")
                # w1, w2 = torch.LongTensor(self.multiG.aligned_w1_test).cuda(), torch.LongTensor(self.multiG.aligned_w2_test).cuda()
                # w1, w2 = torch.LongTensor(np.array(range(len(self.multiG.KG1.id2we)))).cuda(), torch.LongTensor(np.array(range(len(self.multiG.KG2.id2we)))).cuda()
                # self.model.test_word(self.multiG.align_words_test, self.eval_method)

            logger.info("=============eval_acc_mrr finished, Time use: %s==================" % (time.time()-t0))


    def save(self,PATH):
        torch.save(self.model, f"{PATH}/model")

    def load(self, PATH):
        the_model = torch.load(PATH)
        return the_model


# if __name__ == '__main__':

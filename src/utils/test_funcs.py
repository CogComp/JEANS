import multiprocessing
import torch
import numpy as np
import time
import logging
import torch.nn.functional as F
import sys
from utils.utils import *
from difflib import SequenceMatcher
import pinyin

logger = logging.getLogger(__name__)

# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False
FAISS_AVAILABLE = False


def eval_acc_mrr(embed1, embed2, selected_pairs, method, KG1_id2we, KG2_id2we, langs):
    # compute similarityinverted softmax
    embed1 = embed1 / embed1.norm(2, 1, keepdim=True).expand_as(embed1)
    embed2 = embed2 / embed2.norm(2, 1, keepdim=True).expand_as(embed2)

    # nearest neighbors
    if method == 'nn':
        query = embed1[selected_pairs[:, 0]]
        scores = query.mm(embed2.transpose(0, 1))
        scores_index = scores.argsort(dim=-1, descending=True)

    #
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, embed2.size(0), bs):
            scores = embed1.mm(embed2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            word_scores.append(scores.index_select(0, selected_pairs[:, 0]))
        scores = torch.cat(word_scores, 1)
        scores_index = scores.argsort(dim=-1, descending=True)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(embed2, embed1, knn)
        average_dist2 = get_nn_avg_dist(embed1, embed2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(embed1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(embed2)
        # average_dist1 = average_dist1.cuda()
        # average_dist2 = average_dist2.cuda()

        query = embed1[selected_pairs[:, 0]]
        scores = query.mm(embed2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[selected_pairs[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])
        scores_index = scores.argsort(dim=-1, descending=True)

    '''
    MRR
    '''
    pair_dic_newid_right = selected_pairs[:,1].unsqueeze(1)
    mrr_input = (scores_index == pair_dic_newid_right).nonzero()[:,1]
    mrr_input = mrr_input.cpu().data.numpy()
    mrr_score = np.mean([1. / (r + 1) for r in mrr_input])
    logger.info("MRR: %.5f" % mrr_score)

    '''
    precision
    '''
    top_matches = scores.topk(10, 1, True)[1]
    '''
    string match
    '''
    # top_matches_cpu = top_matches.cpu().numpy()
    # top_matches_new = []
    # for i, src_id in enumerate(selected_pairs[:, 0].cpu().numpy()):
    #     # print(KG1_id2we[src_id],src_id)
    #     src_word = KG1_id2we[src_id].split("dbpedia/")[1]
    #     if langs == "zh_en":
    #         src_word = pinyin.get('src_word', format="strip", delimiter=" ")
    #         src_word = src_word.replace(" ", "")
    #     origin_id_list = top_matches_cpu[i].tolist()
    #     sim_scores = []
    #     for cand_word_id in origin_id_list:
    #         # print(cand_word_id,  KG2_id2we[cand_word_id])
    #         cand_word = KG2_id2we[cand_word_id].split("dbpedia/")[1]
    #         sim_scores.append(similar(src_word, cand_word))
    #     new_id_list = [x for _, x in sorted(zip(sim_scores, origin_id_list), reverse=True)]
    #     top_matches_new.append(new_id_list)
    # del top_matches
    # top_matches_new = torch.LongTensor(np.array(top_matches_new)).cuda()
    # top_matches = top_matches_new

    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == selected_pairs[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(selected_pairs[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
            '''
            error analysis
            '''
            # if(k == 1):
            #     if matching[src_id] == 0:
            #         wrong = ",".join([str(x) for x in top_matches_cpu[i].tolist()])
            #         logger.info(f"incorrect pair - {src_id}: right: {src_id}, wrong: {wrong}")
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        if k == 1:
            precision_at_1 = precision_at_k
        logger.info("Hits@%i = %.5f" %(k, precision_at_k))

    return precision_at_1


def eval_acc_word(embed1, embed2, selected_pairs, method):
    pair_dic_newid = selected_pairs
    # compute similarity
    embed1 = embed1 / embed1.norm(2, 1, keepdim=True).expand_as(embed1)
    embed2 = embed2 / embed2.norm(2, 1, keepdim=True).expand_as(embed2)

    # nearest neighbors
    if method == 'nn':
        query = embed1[pair_dic_newid[:, 0]]
        scores = query.mm(embed2.transpose(0, 1))
    # contextual dissimilarity measure
    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, embed2.size(0), bs):
            scores = embed1.mm(embed2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            word_scores.append(scores.index_select(0, selected_pairs[:, 0]))
        scores = torch.cat(word_scores, 1)
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(embed2, embed1, knn)
        average_dist2 = get_nn_avg_dist(embed1, embed2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(embed1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(embed2)
        query = embed1[selected_pairs[:, 0]]
        scores = query.mm(embed2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[selected_pairs[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

    # results = []
    top_matches = scores.topk(10, 1, True)[1]
    for k in [1, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == pair_dic_newid[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(pair_dic_newid[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        logger.info("Hits@%i = %.5f" %(k, precision_at_k))



def generate_neighbours_multi_embed(args, ent_embedding, ent_list, k):
    '''
    ent_list: [100, 101, 102 .... ] - offset
    '''
    ent_list = torch.LongTensor(ent_list)
    if args["cuda"]:
        ent_list = ent_list.cuda()
    ent_list_embedding = F.normalize(ent_embedding(ent_list), 2, 1)
    scores = ent_list_embedding.mm(ent_list_embedding.transpose(0, 1))
    top_matches = scores.topk(k, 1, True)[1].cpu()
    dic = dict()
    # print("entlist: ", ent_list)
    for i in ent_list:
        dic[i.tolist()] = top_matches[i]
    del ent_list_embedding, ent_list, scores
    return dic


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()

def filter_entity(aligned_e1, not_aligned_e1):
    aligned_e1 = set(aligned_e1)
    not_aligned_e1 = np.array(list(set(not_aligned_e1) - aligned_e1))
    not_aligned_e1.sort(axis=0)
    not_aligned_e1_index2id = {index: id for index, id in enumerate(not_aligned_e1)}
    not_aligned_e1_id2index = {id: index for index, id in not_aligned_e1_index2id.items()}
    return not_aligned_e1, not_aligned_e1_index2id, not_aligned_e1_id2index


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()




import torch
import torch.nn.functional as F
import time
import logging
# import igraph as ig
import networkx as nx
import itertools
import gc
import numpy as np


logger = logging.getLogger(__name__)

def bootstrapping(args, ent_embedding1, ent_embedding2, ent_list1, ent_list2, mapping, labeled_alignment, related_mat = None):
    is_sigmoid = False
    ref_sim_mat = cal_similarity(args, ent_embedding1, ent_embedding2, ent_list1, ent_list2, mapping)
    if related_mat is not None:
        ref_sim_mat = ref_sim_mat + (related_mat * 0.05)
    n = ref_sim_mat.shape[0]
    curr_labeled_alignment = find_potential_alignment(args, ref_sim_mat, n)
    if curr_labeled_alignment is not None:
        labeled_alignment = update_labeled_alignment(labeled_alignment, curr_labeled_alignment, ref_sim_mat, n)
        labeled_alignment = update_labeled_alignment_label(labeled_alignment, ref_sim_mat, n)
        del curr_labeled_alignment
    # labeled_alignment = curr_labeled_alignment
    gc.collect()
    return labeled_alignment


def cal_similarity(args, ent_embedding1, ent_embedding2, ent_list1, ent_list2, mapping):
    ent_list1 = torch.LongTensor(ent_list1)
    ent_list2 = torch.LongTensor(ent_list2)
    if args["cuda"]:
        ent_list1 = ent_list1.cuda()
        ent_list2 = ent_list2.cuda()
    ent_list_embedding1 = F.normalize(mapping(ent_embedding1(ent_list1)), 2, 1)
    ent_list_embedding2 = F.normalize(ent_embedding2(ent_list2), 2, 1)
    scores = ent_list_embedding1.mm(ent_list_embedding2.transpose(0, 1))
    return scores

def find_potential_alignment(args, ref_sim_mat, n):
    t = time.time()
    potential_aligned_pairs = generate_alignment(args, ref_sim_mat, n)
    if potential_aligned_pairs is None or len(potential_aligned_pairs) == 0:
        return None
    t1 = time.time()
    selected_pairs = mwgm(potential_aligned_pairs, ref_sim_mat, mwgm_igraph)
    check_alignment(selected_pairs, n, context="selected_pairs")
    del potential_aligned_pairs
    logger.info("mwgm costs time: {:.3f} s".format(time.time() - t1))
    logger.info("selecting potential alignment costs time: {:.3f} s".format(time.time() - t))
    return selected_pairs


def generate_alignment(args, ref_sim_mat, n):
    potential_aligned_pairs = filter_mat(ref_sim_mat, args["threshold"])
    if len(potential_aligned_pairs) == 0:
        return None
    check_alignment(potential_aligned_pairs, n, context="after sim filtered")
    neighbors = search_nearest_k(ref_sim_mat, args["topk"])
    if neighbors is not None:
        potential_aligned_pairs &= neighbors
        if len(potential_aligned_pairs) == 0:
            return None, None
        check_alignment(potential_aligned_pairs, n, context="after sim and neighbours filtered")
    del neighbors
    return potential_aligned_pairs

def filter_mat(mat, threshold, greater=True, equal=False):
    mat = mat.detach().cpu().numpy()
    if greater and equal:
        x, y = np.where(mat >= threshold)
    elif greater and not equal:
        x, y = np.where(mat > threshold)
    elif not greater and equal:
        x, y = np.where(mat <= threshold)
    else:
        x, y = np.where(mat < threshold)
    return set(zip(x, y))

def check_alignment(aligned_pairs, all_n, context="", is_cal=True):
    if aligned_pairs is None or len(aligned_pairs) == 0:
        logger.info("{}, Empty aligned pairs".format(context))
        return
    num = 0
    for x, y in aligned_pairs:
        if x == y:
            num += 1
    logger.info("{}, right alignment: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))
    if is_cal:
        precision = round(num / len(aligned_pairs), 6)
        recall = round(num / all_n, 6)
        if recall > 1.0:
            recall = round(num / all_n, 6)
        f1 = round(2 * precision * recall / (precision + recall+1), 6)
        logger.info("precision={}, recall={}, f1={}".format(precision, recall, f1))


def search_nearest_k(sim_mat, k):
    if k == 0:
        return None
    neighbors = set()
    ref_num = sim_mat.shape[0]
    top_matches = sim_mat.topk(k, 1, True)[1].cpu().numpy()
    for i in range(ref_num):
        pairs = [j for j in itertools.product([i], top_matches[i, :])]
        neighbors |= set(pairs)
        # del rank
    assert len(neighbors) == ref_num * k
    return neighbors


def mwgm(pairs, sim_mat, func):
    return func(pairs, sim_mat)

def mwgm_igraph(pairs, sim_mat):
    if not isinstance(pairs, list):
        pairs = list(pairs)
    index_id_dic1, index_id_dic2 = dict(), dict()
    index1 = set([pair[0] for pair in pairs])
    index2 = set([pair[1] for pair in pairs])
    for index in index1:
        index_id_dic1[index] = len(index_id_dic1)
    off = len(index_id_dic1)
    for index in index2:
        index_id_dic2[index] = len(index_id_dic2) + off
    assert len(index1) == len(index_id_dic1)
    assert len(index2) == len(index_id_dic2)
    edge_list = [(index_id_dic1[x], index_id_dic2[y]) for (x, y) in pairs]
    weight_list = [sim_mat[x, y] for (x, y) in pairs]
    leda_graph = ig.Graph(edge_list)
    leda_graph.vs["type"] = [0] * len(index1) + [1] * len(index2)
    leda_graph.es['weight'] = weight_list
    res = leda_graph.maximum_bipartite_matching(weights=leda_graph.es['weight'])
    print(res)
    selected_index = [e.index for e in res.edges()]
    matched_pairs = set()
    for index in selected_index:
        matched_pairs.add(pairs[index])
    return matched_pairs


def update_labeled_alignment(labeled_alignment, curr_labeled_alignment, sim_mat, all_n):
    # all_alignment = labeled_alignment | curr_labeled_alignment
    # check_alignment(labeled_alignment, all_n, context="before updating labeled alignment")
    labeled_alignment_dict = dict(labeled_alignment)
    n, n1 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n1 += 1
        if i in labeled_alignment_dict.keys():
            jj = labeled_alignment_dict.get(i)
            old_sim = sim_mat[i, jj]
            new_sim = sim_mat[i, j]
            if new_sim >= old_sim:
                if jj == i and j != i:
                    n += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n, "greedy update wrongly: ", n1)
    labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_alignment(labeled_alignment, all_n, context="after editing labeled alignment (<-)")
    # selected_pairs = mwgm(all_alignment, sim_mat, mwgm_igraph)
    # check_alignment(selected_pairs, context="after updating labeled alignment with mwgm")
    return labeled_alignment

def update_labeled_alignment_label(labeled_alignment, sim_mat, all_n):
    # check_alignment(labeled_alignment, all_n, context="before updating labeled alignment label")
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        ents_j = labeled_alignment_dict.get(j, set())
        ents_j.add(i)
        labeled_alignment_dict[j] = ents_j
    for j, ents_j in labeled_alignment_dict.items():
        if len(ents_j) == 1:
            for i in ents_j:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in ents_j:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_alignment(updated_alignment, all_n, context="after editing labeled alignment (->)")
    return updated_alignment


def generate_newly_triples(ent1, ent2, KG1, KG2):
    new_211, new_112, new_221, new_122 = list(), list(), list(), list()
    for r, t in KG1.rt_dict.get(ent1, set()):
        new_211.append((ent2, r, t))
    for h, r in KG1.hr_dict.get(ent1, set()):
        new_112.append((h, r, ent2))

    for r, t in KG2.rt_dict.get(ent2, set()):
        new_122.append((ent1, r, t))
    for h, r in KG2.hr_dict.get(ent2, set()):
        new_221.append((h, r, ent1))
    return new_211, new_112, new_221, new_122


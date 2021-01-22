# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
import torch.nn.functional as F
import numpy as np



logger = getLogger()

def normalize_embed(args, mapping, ent_embedding1, ent_embedding2, ent_list1, ent_list2):
    ent_list1 = torch.LongTensor(ent_list1)
    ent_list2 = torch.LongTensor(ent_list2)
    if args["cuda"]:
        ent_list1 = ent_list1.cuda()
        ent_list2 = ent_list2.cuda()
    # swj
    # ent_list_embedding1 = F.normalize(mapping(ent_embedding1[ent_list1]), 2, 1)
    # ent_list_embedding2 = F.normalize(ent_embedding2[ent_list2], 2, 1)
    ent_list_embedding1 = F.normalize(mapping(ent_embedding1(ent_list1)), 2, 1)
    ent_list_embedding2 = F.normalize(ent_embedding2(ent_list2), 2, 1)
    return ent_list_embedding1.data, ent_list_embedding2.data


def get_candidates(emb1, emb2, params, aligned_e1, aligned_e2, mode):
    """
    Get best translation pairs candidates.
    """
    if mode == "s2t":
        aligned_left = aligned_e1
        aligned_right = aligned_e2
    elif mode == "t2s":
        aligned_left = aligned_e2
        aligned_right = aligned_e1
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if params["dico_max_rank"] > 0 and not params["dico_method"].startswith('invsm_beta_'):
        n_src = params["dico_max_rank"]

    # nearest neighbors
    if params["dico_method"] == 'nn':

        # for every source word
        for i in range(0, n_src, bs):
            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(8, dim=1, largest=True, sorted=True)
            best_scores, best_targets = filter_entity(best_scores, best_targets, aligned_left, aligned_right, left_index= i)


            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    # inverted softmax
    elif params["dico_method"].startswith('invsm_beta_'):

        beta = float(params["dico_method"][len('invsm_beta_'):])

        # for every target word
        for i in range(0, emb2.size(0), bs):
            # compute source words scores
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            try:
                best_scores, best_targets = scores.topk(5, dim=1, largest=True, sorted=True)
            except:
                best_scores, best_targets = scores.topk(4, dim=1, largest=True, sorted=True)
            best_scores, best_targets = filter_entity(best_scores, best_targets, aligned_left, aligned_right, left_index= i)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append((best_targets + i).cpu())

        all_scores = torch.cat(all_scores, 1)
        all_targets = torch.cat(all_targets, 1)

        all_scores, best_targets = all_scores.topk(2, dim=1, largest=True, sorted=True)
        all_targets = all_targets.gather(1, best_targets)

    # contextual dissimilarity measure
    elif params["dico_method"].startswith('csls_knn_'):

        knn = params["dico_method"][len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(30, dim=1, largest=True, sorted=True)
            best_scores, best_targets = filter_entity(best_scores, best_targets, aligned_left, aligned_right, left_index= i)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if params["dico_max_rank"] > 0:
        selected = all_pairs.max(1)[0] <= params["dico_max_rank"]
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if params["dico_max_size"] > 0:
        all_scores = all_scores[:params["dico_max_size"]]
        all_pairs = all_pairs[:params["dico_max_size"]]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if params["dico_min_size"] > 0:
        diff[:params["dico_min_size"]] = 1e9

    # confidence threshold
    if params["dico_threshold"] > 0:
        mask = diff > params["dico_threshold"]
        logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs


def build_dictionary(src_emb, tgt_emb, params, aligned_e1, aligned_e2, s2t_candidates=None, t2s_candidates=None):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")
    s2t = 'S2T' in params["dico_build"]
    t2s = 'T2S' in params["dico_build"]
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, params,  aligned_e1, aligned_e2, "s2t")
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, params, aligned_e1, aligned_e2, "t2s")
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    logger.info("Got candidates for s2t and t2s")
    if params["dico_build"] == 'S2T':
        dico = s2t_candidates
    elif params["dico_build"] == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])
        if params["dico_build"] == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert params["dico_build"] == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                logger.warning("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    logger.info('New train dictionary of %i pairs.' % dico.size(0))
    return dico.cuda() if params["cuda"] else dico

def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances.numpy()


def filter_entity(best_scores, best_targets, aligned_left, aligned_right, left_index):
    best_scores, best_targets = best_scores.cpu().numpy(), best_targets.cpu().numpy()
    best_scores_new, best_targets_new = [], []
    for score, target in zip(best_scores, best_targets):
        score_new, target_new = [], []
        count = 0
        if left_index in aligned_left:
            score_new = score[:2].tolist()
            target_new = target[:2].tolist()
        else:
            for s, t in zip(score, target):
                if t in aligned_right:
                    continue
                else:
                    if count == 2:
                        break
                    score_new.append(s)
                    target_new.append(t)
                    count += 1
        if len(target_new) != 2:
            print(left_index)
            exit()
        best_scores_new.append(score_new)
        best_targets_new.append(target_new)
        left_index += 1

    best_scores_new, best_targets_new = torch.FloatTensor(np.array(best_scores_new)).cuda(), torch.LongTensor(np.array(best_targets_new)).cuda()
    del best_scores, best_targets
    return best_scores_new, best_targets_new
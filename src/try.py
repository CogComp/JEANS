from data.vocab import *
from data.negative_sampler import *
from utils.utils import *
import logging
import numpy as np
import torch
from torch.nn.functional import logsigmoid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.terminator = ''
logger.addHandler(stream_handler)

SGargs = {"min_count": 5,
        "corpus": "/scratch/swj0419/joint_emb/data/wiki/en_test.txt",
        "negative": 5,
        "window": 5,
        "seed": 5,
        "batch_szie": 256,
        "word2id": "/scratch/swj0419/joint_emb/data/wiki2vec/frword2id.pk",
        "samples": 1e-3,
        "loss": "neg"
        }



def generate_words_from_doc(doc, num_processed_words):
    """
    this generator separates a long document into shorter documents
    :param doc: np.array(np.int), word_id list
    :param num_processed_words:
    :return: shorter words list: np.array(np.int), incremented num processed words: num_processed_words
    """
    new_doc = []
    for word_id in doc:
        num_processed_words += 1
        # if corpus.discard(word_id=word_id, rnd=rnd):
        #     continue
        new_doc.append(word_id)
        if len(new_doc) >= 1000:
            yield np.array(new_doc), num_processed_words
            new_doc = []
    yield np.array(new_doc), num_processed_words


def nce_loss(pos_dot, neg_dot, pos_log_k_negative_prob, neg_log_k_negative_prob, size_average=True, reduce=True):
    """
    https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf
    :param pos_dot:
    :param neg_dot:
    :param pos_log_k_negative_prob:
    :param neg_log_k_negative_prob:
    :param size_average:
    :param reduce:
    :return:
    """
    s_pos = pos_dot - pos_log_k_negative_prob
    s_neg = neg_dot - neg_log_k_negative_prob
    loss = - (torch.mean(logsigmoid(s_pos) + torch.sum(logsigmoid(-s_neg), dim=1)))

    if not reduce:
        return loss
    if size_average:
        return torch.mean(loss)
    return torch.sum(loss)


def negative_sampling_loss(pos_dot, neg_dot, size_average=True, reduce=True):
    """
    :param pos_dot: The first tensor of SKipGram's output: (#mini_batches)
    :param neg_dot: The second tensor of SKipGram's output: (#mini_batches, #negatives)
    :param size_average:
    :param reduce:
    :return: a tensor has a negative sampling loss
    """
    loss = - (
            logsigmoid(pos_dot) + torch.sum(logsigmoid(-neg_dot), dim=1)
    )

    if not reduce:
        return loss
    if size_average:
        return torch.mean(loss)

    return torch.sum(loss)


def train_on_minibatches(model, optimizer, use_cuda, inputs, contexts, negatives):
    num_minibatches = len(contexts)
    inputs = torch.LongTensor(inputs).view(num_minibatches, 1)
    if use_cuda:
        inputs = inputs.cuda()

    optimizer.zero_grad()

    if is_neg_loss:
        contexts = torch.LongTensor(contexts).view(num_minibatches, 1)
        negatives = torch.LongTensor(negatives)
        if use_cuda:
            contexts = contexts.cuda()
            negatives = negatives.cuda()

        pos, neg = model.forward(inputs, contexts, negatives)
        loss = negative_sampling_loss(pos, neg)
    else:
        pos_log_k_negative_prob = torch.FloatTensor(log_k_prob[contexts]).view(num_minibatches, 1)
        neg_log_k_negative_prob = torch.FloatTensor(log_k_prob[negatives])

        contexts = torch.LongTensor(contexts).view(num_minibatches, 1)
        negatives = torch.LongTensor(negatives)
        if use_cuda:
            pos_log_k_negative_prob = pos_log_k_negative_prob.cuda()
            neg_log_k_negative_prob = neg_log_k_negative_prob.cuda()
            contexts = contexts.cuda()
            negatives = negatives.cuda()

        pos, neg = model.forward(inputs, contexts, negatives)
        loss = nce_loss(pos, neg, pos_log_k_negative_prob, neg_log_k_negative_prob)

    loss.backward()
    optimizer.step()
    return loss.item()


corpus = Corpus(min_count=args["min_count"], word2id = load_pk(args["word2id"]))
docs = corpus.tokenize_from_file(args["corpus"])
corpus.build_discard_table(t=args["samples"])
logger.info('V:{}, #words:{}\n'.format(corpus.num_vocab, corpus.num_words))
rnd = np.random.RandomState(args["seed"])

is_neg_loss = (args["loss"] == 'neg')
negative_sampler = NegativeSampler(
    frequency=corpus.dictionary.id2freq,
    negative_alpha=0.75,
    is_neg_loss=is_neg_loss,
    table_length=int(1e8),
)

if is_neg_loss:
    logger.info('loss function: Negative Sampling\n')
else:
    log_k_prob = np.log(args["negative"] * negative_sampler.noise_dist)
    logger.info('loss function: NCE\n')

inputs = []
contexts = []
num_processed_words = last_check = 0

for sentence in docs:
    for doc, num_processed_words in generate_words_from_doc(doc=sentence,
                                                            num_processed_words=num_processed_words):
        doclen = len(doc)
        dynamic_window_sizes = rnd.randint(low=1, high=args["window"] + 1, size=doclen)
        for (position, (word_id, dynamic_window_size)) in enumerate(zip(doc, dynamic_window_sizes)):
            begin_pos = max(0, position - dynamic_window_size)
            end_pos = min(position + dynamic_window_size, doclen - 1) + 1
            for context_position in range(begin_pos, end_pos):
                if context_position == position:
                    continue
                contexts.append(doc[context_position])
                inputs.append(word_id)
                if len(inputs) >= args["batch_szie"]:
                    negatives = negative_sampler.sample(k=args["negative"], rnd=rnd, exclude_words=contexts)
                    inputs.clear()
                    contexts.clear()


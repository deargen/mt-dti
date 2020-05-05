from collections import OrderedDict
import tensorflow as tf
import time
import numpy as np

__author__ = 'Bonggun Shin'


def prepare_interaction_pairs(XD, XDm, XT, XTm, Y, rows, cols):
    drugs = []
    drug_mask = []
    targets = []
    target_mask = []

    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)
        dmask = XDm[rows[pair_ind]]
        drug_mask.append(dmask)

        target = XT[cols[pair_ind]]
        targets.append(target)
        tmask = XTm[cols[pair_ind]]
        target_mask.append(tmask)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    drug_mask_data = np.stack(drug_mask)
    target_data = np.stack(targets)
    target_mask_data = np.stack(target_mask)

    return drug_data, drug_mask_data, target_data, target_mask_data, affinity


def get_trn_dev_tst(dataset, fold=0):
    (XD, XDm, XT, XTm, Y, trn_sets, dev_sets, tst_set, row_idx, col_idx) = dataset
    trn_set = trn_sets[fold]
    dev_set = dev_sets[fold]

    drug_idx_trn = row_idx[trn_set]
    protein_idx_trn = col_idx[trn_set]
    drug_idx_dev = row_idx[dev_set]
    protein_idx_dev = col_idx[dev_set]
    drug_idx_tst = row_idx[tst_set]
    protein_idx_tst = col_idx[tst_set]

    (xd_trn, xdm_trn, xt_trn, xtm_trn, y_trn) = prepare_interaction_pairs(XD, XDm, XT, XTm, Y, drug_idx_trn, protein_idx_trn)
    (xd_dev, xdm_dev, xt_dev, xtm_dev, y_dev) = prepare_interaction_pairs(XD, XDm, XT, XTm, Y, drug_idx_dev, protein_idx_dev)
    (xd_tst, xdm_tst, xt_tst, xtm_tst, y_tst) = prepare_interaction_pairs(XD, XDm, XT, XTm, Y, drug_idx_tst, protein_idx_tst)
    trndev = xd_trn, xdm_trn, xt_trn, xtm_trn, y_trn, \
             xd_dev, xdm_dev, xt_dev, xtm_dev, y_dev, \
             xd_tst, xdm_tst, xt_tst, xtm_tst, y_tst

    return trndev


def get_trn_dev(dataset, fold=0):
    (XD, XT, Y, trn_sets, dev_sets, tst_set, row_idx, col_idx) = dataset
    trn_set = trn_sets[fold]
    dev_set = dev_sets[fold]

    drug_idx_trn = row_idx[trn_set]
    protein_idx_trn = col_idx[trn_set]
    drug_idx_dev = row_idx[dev_set]
    protein_idx_dev = col_idx[dev_set]

    (xd_trn, xt_trn, y_trn) = prepare_interaction_pairs(XD, XT, Y, drug_idx_trn, protein_idx_trn)
    (xd_dev, xt_dev, y_dev) = prepare_interaction_pairs(XD, XT, Y, drug_idx_dev, protein_idx_dev)
    trndev = xd_trn, xt_trn, y_trn, xd_dev, xt_dev, y_dev

    return trndev


class DTITokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        tokens = []
        tokens.append("[BEGIN]")
        for c in text:
            tokens.append(c)
        tokens.append("[END]")

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)

    def convert_by_vocab(self, vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = OrderedDict()
        index = 0
        with tf.gfile.GFile(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab


class SmilesTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        tokens = []
        tokens.append("[BEGIN]")
        for c in text:
            tokens.append(c)
        tokens.append("[END]")

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)

    def convert_by_vocab(self, vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = OrderedDict()
        index = 0
        with tf.gfile.GFile(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))
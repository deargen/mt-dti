import os
import csv
import json
import pickle
import argparse
import numpy as np
import random as rn
import _pickle as cPickle
from copy import deepcopy
from collections import OrderedDict

__author__ = 'Bonggun Shin'

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)


def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)))  # +1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch]-1)] = 1

    return X  # tolist()


def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch])-1] = 1

    return X  # tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X  # tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # tolist()

# TODO: get rid of old voca
class DataSet(object):
    def __init__(self, fpath, setting_no, seqlen, smilen, need_shuffle = False):
        self.seqlen = seqlen
        self.smilen = smilen
        self.charseqset = {
            "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
            "U": 19, "T": 20, "W": 21,
            "V": 22, "Y": 23, "X": 24,
            "Z": 25
        }
        self.charseqset_size = len(self.charseqset)

        self.inv_charseqset = {v: k for k, v in self.charseqset.items()}

        self.charsmiset = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                           "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                           "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                           "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                           "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                           "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                           "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                           "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
                           }
        self.inv_charsmiset = {v: k for k, v in self.charsmiset.items()}

        self.charsmiset_size = len(self.charsmiset)

        # read raw file
        self._raw = self.read_sets(fpath, setting_no)

        # iteration flags
        self._num_data = len(self._raw)

    def read_sets(self, fpath, setting_no):
        print("Reading %s start" % fpath)

        test_fold = json.load(open(fpath + "folds/test_fold_setting" + str(setting_no)+".txt"))
        train_folds = json.load(open(fpath + "folds/train_fold_setting" + str(setting_no)+".txt"))

        return test_fold, train_folds

    def parse_data(self, fpath,  with_label=True):

        print("Read %s start" % fpath)

        ligands = json.load(open(fpath+"ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath+"proteins.txt"), object_pairs_hook=OrderedDict)

        Y = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')  # TODO: read from raw

        XD = []
        XT = []

        chembl_id_list = []
        smiles_list = []

        protein_id_list = []
        fasta_list = []

        if with_label:
            for d in ligands.keys():
                chembl_id_list.append(d)
                smiles_list.append(ligands[d])
                XD.append(label_smiles(ligands[d], self.smilen, self.charsmiset))

            for t in proteins.keys():
                protein_id_list.append(t)
                fasta_list.append(proteins[t])
                XT.append(label_sequence(proteins[t], self.seqlen, self.charseqset))
        else:
            for d in ligands.keys():
                XD.append(one_hot_smiles(ligands[d], self.smilen, self.charsmiset))

            for t in proteins.keys():
                XT.append(one_hot_sequence(proteins[t], self.seqlen, self.charseqset))

        return chembl_id_list, smiles_list, XD, protein_id_list, fasta_list, XT, Y

    def array_to_smiles(self, indices):
        smiles = []
        for index in indices:
            if index==0:
                continue
            item = self.inv_charsmiset[index]
            smiles.append(item)
        return smiles

    def array_to_fasta(self, indices):
        fasta = []
        for index in indices:
            if index == 0:
                continue
            item = self.inv_charseqset[index]
            fasta.append(item)
        return fasta


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def load_cid_chembl_lookup(base_path):
    chembl_cid = {}
    with open("%s/CID_CHEMBL.tsv" % (base_path), "rt") as f: # 1,816,499
        csvr = csv.reader(f, delimiter='\t')
        for row in csvr:
            chembl_cid[row[1]]=row[0]

    with open("%s/chembl_to_cids.txt" %(base_path), "rt") as f: # 1,816,535
        csvr = csv.reader(f, delimiter='\t')
        for row in csvr:
            chembl_cid[row[0]] = row[1]

    return chembl_cid



def save_interactions_to_csv(label_row_inds, label_col_inds, chembl_id_list, smiles_list, XD,
                             protein_id_list, fasta_list, XT, Y, base_path):
    chembl_to_cid = load_cid_chembl_lookup(base_path)

    reference_dic = {}

    with open("%s/kiba/kiba_kd.csv" % base_path, "w") as f:
        column_names = "id,cid,smlies,pid,fasta,kd\n"
        f.write(column_names)
        for i, row in enumerate(label_row_inds):
            col = label_col_inds[i]

            chembl_id = chembl_id_list[row]
            cid = chembl_to_cid[chembl_id]
            smiles = smiles_list[row]
            smiles_idx = XD[row]

            protein_id = protein_id_list[col]
            fasta = fasta_list[col]
            fasta_idx = XT[col]

            y = Y[row, col]
            one_line = "%d,%s,%s,%s,%s,%f\n" % (i, cid, smiles, protein_id, fasta, y)
            reference_dic[(cid, protein_id)] = [i, row, col, y, smiles, fasta]
            f.write(one_line)

    with open("%s/kiba/reference_dic.cpkl" % base_path, "wb") as handle:
        cPickle.dump(reference_dic, handle)
    print('done')
    return reference_dic


def get_split(base_path, problem_type, max_seq_len, max_smi_len):
    data_path = base_path + '/kiba/'
    dataset = DataSet(fpath=data_path,
                      setting_no=problem_type,
                      seqlen=max_seq_len,
                      smilen=max_smi_len,
                      need_shuffle=False)

    chembl_id_list, smiles_list, XD, protein_id_list, fasta_list, XT, Y = dataset.parse_data(fpath=data_path)

    smiles_dic = {}
    for idx, s in enumerate(smiles_list):
        smiles_dic[s] = (idx, chembl_id_list[idx])

    fasta_dic = {}
    for idx, f in enumerate(fasta_list):
        fasta_dic[s] = (idx, protein_id_list[idx])

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    row_idx, col_idx = np.where(np.isnan(Y) == False)  # basically finds the point address of affinity [x,y]

    ref_dic = save_interactions_to_csv(row_idx, col_idx, chembl_id_list, smiles_list, XD, protein_id_list,
                             fasta_list, XT, Y, base_path)

    tst_set, outer_train_sets = dataset.read_sets(data_path, problem_type)
    foldinds = len(outer_train_sets)
    dev_sets = []
    trn_sets = []

    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        dev_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        trn_sets.append(otherfoldsinds)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

    smiles = "".join(dataset.array_to_smiles(XD[0]))
    fasta = "".join(dataset.array_to_fasta(XT[0]))
    # cPickle.dump((XD, XT, Y, trn_sets, dev_sets, tst_set, row_idx, col_idx), open('%s/kiba/kiba.cpkl' % base_path, 'wb'))
    cPickle.dump((XD, XT, Y, trn_sets, dev_sets, tst_set, row_idx, col_idx, chembl_id_list, protein_id_list),
                 open('%s/kiba/kiba_b.cpkl' % base_path, 'wb'))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='../../data', help='Directory for input data.')
    parser.add_argument('--problem_type', type=int, default=1, help='Type of the prediction problem (1-4)')
    parser.add_argument('--max_seq_len', default=1000, type=int, help='Length of input sequences.')
    parser.add_argument('--max_smi_len', default=100, type=int, help='Length of input sequences.')
    args, unparsed = parser.parse_known_args()

    get_split(args.base_path, args.problem_type, args.max_seq_len, args.max_smi_len)
import os
import json
import math
import random
import numpy as np
import tensorflow as tf
import _pickle as cPickle
import collections
from src.utils.utils import DTITokenizer, get_trn_dev_tst
from collections import OrderedDict


__author__ = 'Bonggun Shin'


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("fold", 0, "fold 0 to 4")
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
flags.DEFINE_integer("max_molecule_seq_length", 100, "Maximum molecule sequence length.")
flags.DEFINE_integer("max_protein_seq_length", 1000, "Maximum protein sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 15,
                     "Maximum number of masked LM predictions per sequence.")
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
flags.DEFINE_string("data_path", "../../data", "base path for dataset")
flags.DEFINE_string("dataset_name", "kiba", "dataset name (kiba, davis)")
# flags.DEFINE_string("dataset_name", "davis", "dataset name (kiba, davis)")
flags.DEFINE_integer("version", 2, "version 1: training, version2: analysis")

molecule_input_file = "%s/%s/ligands_can.txt" % (FLAGS.data_path, FLAGS.dataset_name)
protein_input_file = "%s/%s/proteins.txt" % (FLAGS.data_path, FLAGS.dataset_name)


def truncate_seq_pair(tokens, max_num_tokens, rng):
    while True:
        total_length = len(tokens)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def read_data(fold):
    rng = random.Random(FLAGS.random_seed)
    molecule_tokenizer = DTITokenizer("%s/pretrain/vocab_smiles.txt" % (FLAGS.data_path))
    max_seq_length = FLAGS.max_molecule_seq_length
    molecule_list, molecule_mask, cid_list = read_inputs(molecule_input_file, molecule_tokenizer, max_seq_length, rng)
    tf.logging.info("reading molecule done")

    protein_tokenizer = DTITokenizer("%s/pretrain/vocab_fasta.txt" % (FLAGS.data_path))
    max_seq_length = FLAGS.max_protein_seq_length
    protein_list, protein_mask, pid_list = read_inputs(protein_input_file, protein_tokenizer, max_seq_length, rng)
    tf.logging.info("reading protein done")

    (XD, XT, Y, trn_sets, dev_sets, tst_set, row_idx, col_idx, chembl_id_list, protein_id_list) = \
        cPickle.load(open('%s/%s/%s_b.cpkl' % (FLAGS.data_path, FLAGS.dataset_name, FLAGS.dataset_name), 'rb'))

    dataset = (molecule_list, molecule_mask, protein_list, protein_mask, Y, trn_sets, dev_sets, tst_set, row_idx, col_idx)
    trndevtst = get_trn_dev_tst(dataset, fold)

    return trndevtst, cid_list, pid_list


def read_inputs(input_file, tokenizer, max_seq_length, rng):
    max_num_tokens = max_seq_length - 1

    tf.logging.info("Current File: %s" % (input_file))

    seq_to_id = {}
    seq_list = []
    mask_list = []
    id_list = []
    seq_dic = json.load(open(input_file), object_pairs_hook=OrderedDict)

    tf.logging.info("Loaded: %s" % (input_file))

    for idx, k in enumerate(seq_dic.keys()):
        seq = seq_dic[k]
        id_list.append(k)
        tokens = tokenizer.tokenize(seq)
        truncate_seq_pair(tokens, max_num_tokens, rng)
        tokens.insert(0, "[CLS]")

        input_mask = [1] * len(tokens)
        for i in range(len(tokens), max_num_tokens+1):
            tokens.append("[PAD]")
            input_mask.append(0)

        tids = tokenizer.convert_tokens_to_ids(tokens)

        seq_list.append(tids)
        mask_list.append(input_mask)
        tids_srt = ','.join(map(str, tids))
        seq_to_id[tids_srt] = (seq, k)
        tf.logging.info("Processing...[%d/%d]" % (idx, len(seq_dic.keys())))

    return np.array(seq_list), np.array(mask_list), id_list


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def write_instance_to_example_files(xd_list, xdm_list, xt_list, xtm_list, y_list, output_files):
    our_dir = '/'.join(output_files[0].split('/')[:-1])
    if not os.path.exists(our_dir):
        os.makedirs(our_dir)

    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, xd) in enumerate(xd_list):
        xdm = xdm_list[inst_index]
        xt = xt_list[inst_index]
        xtm = xtm_list[inst_index]
        y = y_list[inst_index]

        if FLAGS.dataset_name=="davis":
            y = -math.log(y/1e9,10)

        features = collections.OrderedDict()
        features["xd"] = create_int_feature(xd)
        features["xdm"] = create_int_feature(xdm)
        features["xt"] = create_int_feature(xt)
        features["xtm"] = create_int_feature(xtm)
        features["y"] = create_float_feature([y])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 2:
            tf.logging.info("*** Example ***")
            tf.logging.info("xd: %s" % " ".join(map(str, xd)))
            tf.logging.info("xdm: %s" % " ".join(map(str, xdm)))
            tf.logging.info("xt: %s" % " ".join(map(str, xt)))
            tf.logging.info("xtm: %s" % " ".join(map(str, xtm)))
            tf.logging.info("y: %.2f" % y )

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)



if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("*** CREATING TFRECORD ***")

    for fold in range(5):
        tf.logging.info("*** FOLD %d ***" % fold)
        trndevtst, cid_list, pid_list = read_data(fold)
        xd_trn, xdm_trn, xt_trn, xtm_trn, y_trn, \
        xd_dev, xdm_dev, xt_dev, xtm_dev, y_dev, \
        xd_tst, xdm_tst, xt_tst, xtm_tst, y_tst = trndevtst

        lookup_file_name =  "%s/%s/tst_ids.cpkl" % (FLAGS.data_path, FLAGS.dataset_name)
        if not os.path.isfile(lookup_file_name):
            with open(lookup_file_name, 'wb') as handle:
                cPickle.dump((cid_list, pid_list), handle)

        # exit()

        output_files = ["%s/%s/tfrecord/fold%d.trn.tfrecord" % (FLAGS.data_path, FLAGS.dataset_name, fold)]
        write_instance_to_example_files(xd_trn, xdm_trn, xt_trn, xtm_trn, y_trn, output_files)

        output_files = ["%s/%s/tfrecord/fold%d.dev.tfrecord" % (FLAGS.data_path, FLAGS.dataset_name, fold)]
        write_instance_to_example_files(xd_dev, xdm_dev, xt_dev, xtm_dev, y_dev, output_files)

        output_files = ["%s/%s/tfrecord/fold%d.tst.tfrecord" % (FLAGS.data_path, FLAGS.dataset_name, fold)]
        write_instance_to_example_files(xd_tst, xdm_tst, xt_tst, xtm_tst, y_tst, output_files)


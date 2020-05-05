import os
import csv
import json
import pickle
import random
import argparse
import numpy as np
import tensorflow as tf
import _pickle as cPickle
from copy import deepcopy
import collections
from multiprocessing import Process, Manager
from src.pretrain.smiles_util import SmilesTokenizer, Timer


__author__ = 'Bonggun Shin'


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
flags.DEFINE_integer("max_seq_length", 100, "Maximum sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 15,
                     "Maximum number of masked LM predictions per sequence.")
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
flags.DEFINE_string("base_path", "../../data/pretrain", "base path for dataset")
# flags.DEFINE_string("input_file", ",".join(["%s/smiles%02d.txt" % (FLAGS.base_path, n) for n in range(79)]),
#                     "Input raw text file (or comma-separated list of files).")
flags.DEFINE_string("input_file", ",".join(["%s/molecule/smiles%02d.txt" % (FLAGS.base_path, n) for n in range(50)]),
                    "Input raw text file (or comma-separated list of files).")
flags.DEFINE_string(
    "output_dir", "gs://bdti/mbert/tfr",
    "Output TF example file (or comma-separated list of files).")


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    our_dir = '/'.join(output_files[0].split('/')[:-1])
    if not os.path.exists(our_dir):
        os.makedirs(our_dir)

    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 2:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(instance.tokens))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    # masked_lm_prob 0.15
    # max_seq_length", 170
    # max_predictions_per_seq", 26 (170*.15)
    # vocab_words = list(tokenizer.vocab.keys())
    # rng = random.Random(FLAGS.random_seed)
    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[BEGIN]" or token == "[END]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(4, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)




class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(self.tokens))
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(self.masked_lm_labels))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


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

# def read_smiles(input_files):
#     smiles_list = []
#     len_list = []
#     for i, input_file in enumerate(input_files):
#         tf.logging.info("[%d/%d]Current File: %s" % (i, len(input_files), input_file))
#         # with tf.gfile.GFile("%s/smiles_sample.csv" % (FLAGS.base_path), "r") as reader:
#         with tf.gfile.GFile(input_file, "r") as reader:  # 97,092,853, 97M
#             while True:
#                 line = reader.readline()
#                 if not line:
#                     break
#                 smiles = line.strip().split(',')[1]
#                 smiles_list.append(smiles)
#                 len_list.append(len(smiles))
#
#     tokenizer = SmilesTokenizer("%s/vocab_smiles.txt" % (FLAGS.base_path))
#     vocab_words = list(tokenizer.vocab.keys())
#     rng = random.Random(FLAGS.random_seed)
#     max_num_tokens = FLAGS.max_seq_length - 1
#
#
#     for s in smiles_list:
#         tokens = tokenizer.tokenize(s)
#         truncate_seq_pair(tokens, max_num_tokens, rng)
#         tokens.insert(0, "[CLS]")
#         (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens, FLAGS.masked_lm_prob,
#                                                                                        FLAGS.max_predictions_per_seq,
#                                                                                        vocab_words, rng)
#         instance = TrainingInstance(
#             tokens=tokens,
#             masked_lm_positions=masked_lm_positions,
#             masked_lm_labels=masked_lm_labels)
#         instances.append(instance)
#
#     return instances, tokenizer

#
# def worker_speed_test(input_file):
#     tokenizer = SmilesTokenizer("%s/vocab_smiles.txt" % (FLAGS.base_path))
#     vocab_words = list(tokenizer.vocab.keys())
#     rng = random.Random(FLAGS.random_seed)
#     with Timer("read_smiles_worker"):
#         read_smiles_worker(2, input_file, tokenizer, vocab_words, rng)
#

def tfrecord_smiles_multiprocess(input_files):
    tokenizer = SmilesTokenizer("%s/vocab_smiles.txt" % (FLAGS.base_path))
    vocab_words = list(tokenizer.vocab.keys())
    rng = random.Random(FLAGS.random_seed)
    # manager = Manager()
    jobs = []
    # instances = manager.list()

    for wid, input_file in enumerate(input_files):
        p = Process(target=read_smiles_worker, args=(wid, input_file, tokenizer, vocab_words, rng))
        jobs.append(p)

    for proc in jobs:
        proc.start()

    for proc in jobs:
        proc.join()


def read_smiles_worker(wid, input_file, tokenizer, vocab_words, rng):
    len_list = []
    max_num_tokens = FLAGS.max_seq_length - 1

    instances = []
    tf.logging.info("[worker %d] Current File: %s" % (wid, input_file))

    with tf.gfile.GFile(input_file, "r") as reader:  # 97,092,853, 97M
        while True:
            line = reader.readline()
            if not line:
                break
            smiles = line.strip().split(',')[1]
            len_list.append(len(smiles))
            tokens = tokenizer.tokenize(smiles)
            truncate_seq_pair(tokens, max_num_tokens, rng)
            tokens.insert(0, "[CLS]")
            (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens,
                                                                                           FLAGS.masked_lm_prob,
                                                                                           FLAGS.max_predictions_per_seq,
                                                                                           vocab_words, rng)
            instance = TrainingInstance(
                tokens=tokens,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)

            if len(instances) % 100000 == 0:
                print(len(instances))

            instances.append(instance)

    # 0: 0-9
    # 1: 10-19
    # ...
    # 49: 490-499
    output_files = ["%s/smiles.%03d" % (FLAGS.output_dir, n) for n in range(wid*10, wid*10+10)]
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("*** CREATING TFRECORD ***")

    # # Speed test
    # input_file = "%s/moelecule/10k.txt" % FLAGS.base_path
    # worker_speed_test(input_file)
    # exit()

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tfrecord_smiles_multiprocess(input_files)


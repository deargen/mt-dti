import tensorflow as tf
from src.finetune.dti_model import MbertPcnnModel
from src.finetune.dti_model import load_global_step_from_checkpoint_dir
import argparse
import numpy as np
import os
import time
import shutil
import glob
import re

__author__ = 'Bonggun Shin'


tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Training parser')
parser.add_argument('--gpu_num', default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"], type=str)
parser.add_argument('--model_version', default="11", choices=["1", "2", "3", "4", "11", "14"], type=str)
parser.add_argument('--batch_size', default=512, choices=[256, 512], type=int)
parser.add_argument('--fold', default=0, choices=[0,1,2,3,4], type=int)
parser.add_argument('--data_path', type=str, default="../../data")
parser.add_argument('--dataset_name', type=str, default="kiba", choices=["davis", "kiba"],
                    help='dataset_name')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='learning_rate')
parser.add_argument('--tpu_name', type=str, default="btpu",
                    help='tpu_name')
parser.add_argument('--use_tpu', type=bool, default=False,
                    help='use_tpu')
parser.add_argument('--tpu_zone', type=str, default="us-central1-b",
                    help='tpu_zone')
parser.add_argument('--num_tpu_cores', type=int, default=8,
                    help='num_tpu_cores')
parser.add_argument('--bert_config_file', type=str, default="../../config/m_bert_base_config.json",
                    help='bert_config_file')
parser.add_argument('--init_checkpoint', type=str, default="/pretrain/mbert_6500k/model.ckpt-6500000",
                    help='init_checkpoint')
parser.add_argument('--k1', type=int, default=12, help='kernel_size1')
parser.add_argument('--k2', type=int, default=12, help='kernel_size2')
parser.add_argument('--k3', type=int, default=12, help='kernel_size3')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

i_trn = "%s/%s/tfrecord/fold%d.trn.tfrecord" % (args.data_path, args.dataset_name, args.fold)
i_dev= "%s/%s/tfrecord/fold%d.dev.tfrecord" % (args.data_path, args.dataset_name, args.fold)
i_tst= "%s/%s/tfrecord/fold%d.tst.tfrecord" % (args.data_path, args.dataset_name, args.fold)
output_dir = "%s/%s/mbert_cnn_v%s_lr%.4f_k%d_k%d_k%d_fold%d/" % (args.data_path, args.dataset_name, args.model_version, args.learning_rate, args.k1, args.k2, args.k3, args.fold)
best_model_dir_mse = "%s/%s/mbert_cnn_v%s_lr%.4f_k%d_k%d_k%d_fold%d/best_mse" % (args.data_path, args.dataset_name, args.model_version, args.learning_rate, args.k1, args.k2, args.k3, args.fold)
best_model_dir_ci = "%s/%s/mbert_cnn_v%s_lr%.4f_k%d_k%d_k%d_fold%d/best_ci" % (args.data_path, args.dataset_name, args.model_version, args.learning_rate, args.k1, args.k2, args.k3, args.fold)

if args.dataset_name=="kiba":
    num_trn_example = 78835
    batch_size = args.batch_size
    # num_train_steps = 154000  # (78835/512)*1000 = 153974, 1000 epoch
    num_train_steps = int(num_trn_example*1.0/batch_size*1000)  # (78835/512)*1000 = 153974, 1000 epoch
    num_warmup_steps = num_train_steps//10
    dev_batch_size = 512
    dev_steps = 38*100 # 512*38*10 ~2000k, 100 times
    save_checkpoints_steps = 150 # 78835/512 = 157.67


elif args.dataset_name=="davis":
    num_trn_example = 20035
    batch_size = args.batch_size
    # num_train_steps = 40000  # (20035/512)*1000 = 39130, 1000 epoch
    num_train_steps = int(num_trn_example * 1.0 / batch_size * 1000)  # (78835/512)*1000 = 153974, 1000 epoch
    num_warmup_steps = num_train_steps // 10
    dev_batch_size = 512 # 5009/512
    dev_steps = 10*100 # 100times
    save_checkpoints_steps = 40  # 20035/512 = 40.07

elif args.dataset_name=="metz":
    num_trn_example = 20035
    batch_size = args.batch_size
    num_train_steps = 12021  # (20035/500)*300 = 12, 300 epoch
    num_warmup_steps = num_train_steps // 10
    dev_batch_size = 5009
    dev_steps = 1 # 5009/1=5009
    save_checkpoints_steps = 158  # 20035/500 = 40.07
else:
    batch_size = None
    num_train_steps = None
    num_warmup_steps = None
    dev_batch_size = None
    dev_steps = None
    save_checkpoints_steps = None


def info_scores(current_step, min_mse_step, min_mse_dev, max_ci_dev, mse_tst, ci_tst, prefix='', checkpoint_time=0):
    line1a = '************************** [%s-V%s-lr(%.4f)-f(%d,%d,%d)step(%d/%d)] ***************************' % \
            (args.dataset_name, args.model_version, args.learning_rate, args.k1, args.k2, args.k3, current_step, num_train_steps)
    line1b = '**************************  %s Best @ [%d]  ***************************' % \
             (prefix, min_mse_step)
    line2 = '********** [dev]\tmse:\t%f\tci\t%f **********' % (min_mse_dev, max_ci_dev)
    line3 = '********** [tst]\tmse:\t%f\tci\t%f **********' % (mse_tst, ci_tst)
    line4 = '********** [time]\t%ds **********' % checkpoint_time
    line5 = '********************************************************************'
    tf.logging.info(line1a)
    tf.logging.info(line1b)
    tf.logging.info(line2)
    tf.logging.info(line3)
    if checkpoint_time>0:
        tf.logging.info(line4)
    tf.logging.info(line5)

    with open(output_dir+'/%s_status.txt' % (prefix), 'wt') as handle:
        handle.write(line1a+'\n')
        handle.write(line1b + '\n')
        handle.write(line2+'\n')
        handle.write(line3+'\n')
        if checkpoint_time > 0:
            handle.write(line4+'\n')
        handle.write(line5+'\n')


def restore_best_scores(current_step, best_model_dir_mse, best_model_dir_ci, estimator, input_fn_dev, input_fn_tst):
    minmse_step=0
    minmse_dev=10000
    minmse_ci_dev=0
    minmse_mse_tst=10000
    minmse_ci_tst=0
    maxci_step=0
    maxci_dev=0
    maxci_mse_dev=10000
    maxci_mse_tst=10000
    maxci_ci_tst=0

    checkpoint_file = "%s/checkpoint" % best_model_dir_mse
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'rt') as handle:
            line = handle.readline()
            best_model_prefix = re.findall(r'"(.*?)"', line)[0]

        checkpoint_path = "%s/%s" % (best_model_dir_mse, best_model_prefix)
        eval_results = estimator.evaluate(
            input_fn=input_fn_dev,
            checkpoint_path=checkpoint_path,
            steps=dev_steps)

        minmse_dev = eval_results['mse']
        minmse_ci_dev = eval_results['cindex']
        minmse_step = eval_results['global_step']

        eval_results = estimator.evaluate(
            input_fn=input_fn_tst,
            checkpoint_path=checkpoint_path,
            steps=dev_steps)

        minmse_mse_tst = eval_results['mse']
        minmse_ci_tst = eval_results['cindex']

        info_scores(current_step, minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst,
                    prefix='Restored (sel_mse)')

    checkpoint_file = "%s/checkpoint" % best_model_dir_ci
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'rt') as handle:
            line = handle.readline()
            best_model_prefix = re.findall(r'"(.*?)"', line)[0]

        checkpoint_path = "%s/%s" % (best_model_dir_ci, best_model_prefix)
        eval_results = estimator.evaluate(
            input_fn=input_fn_dev,
            checkpoint_path=checkpoint_path,
            steps=dev_steps)

        maxci_dev = eval_results['cindex']
        maxci_mse_dev = eval_results['mse']
        maxci_step = eval_results['global_step']

        eval_results = estimator.evaluate(
            input_fn=input_fn_tst,
            checkpoint_path=checkpoint_path,
            steps=dev_steps)

        maxci_mse_tst = eval_results['mse']
        maxci_ci_tst = eval_results['cindex']

        info_scores(current_step, maxci_step, maxci_mse_dev, maxci_dev, maxci_mse_tst, maxci_ci_tst,
                    prefix='Restored (sel_ci)')

    return minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst, \
           maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst


def check_improvement_mse(minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst,
                          best_model_dir, eval_results, current_step, estimator, input_fn_tst):
    if minmse_dev > eval_results['mse']:
        tf.logging.info('mse improved!! from %f to "%f"', minmse_dev, eval_results['mse'])
        minmse_dev = eval_results['mse']
        minmse_ci_dev = eval_results['cindex']
        minmse_step = current_step

        eval_results = estimator.evaluate(
            input_fn=input_fn_tst,
            steps=dev_steps)

        minmse_mse_tst = eval_results['mse']
        minmse_ci_tst = eval_results['cindex']

        dest_path = best_model_dir
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for file in glob.glob(r'%s/model.ckpt-%d.*' % (output_dir, current_step)):
            print(file)
            shutil.copy(file, dest_path)

        step_list = []
        for file in glob.glob(r'%s/model.ckpt-*.meta' % (best_model_dir)):
            step_list.append(int(re.findall(r'\d+', file)[-1]))

        n_saved_models = len(step_list)
        n_keep = 5
        n_last = min(n_saved_models, n_keep)
        for del_index in np.sort(step_list)[:(n_saved_models - n_last)]:
            for f in glob.glob(r'%s/model.ckpt-%d.*' % (best_model_dir, del_index)):
                os.remove(f)

        with open('%s/checkpoint' % best_model_dir, 'wt') as handle:
            handle.write('model_checkpoint_path: "model.ckpt-%d"\n' % np.sort(step_list)[-1])

            for i in range(min(n_saved_models, n_keep)):
                handle.write('all_model_checkpoint_paths: "model.ckpt-%d"\n' % (np.sort(step_list)[i - n_last]))

    return minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst


def check_improvement_ci(maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst,
                          best_model_dir, eval_results, current_step, estimator, input_fn_tst):
    if maxci_dev < eval_results['cindex']:
        tf.logging.info('ci improved!! from %f to "%f"', maxci_dev, eval_results['cindex'])
        maxci_dev = eval_results['cindex']
        maxci_mse_dev= eval_results['mse']
        maxci_step = current_step

        eval_results = estimator.evaluate(
            input_fn=input_fn_tst,
            steps=dev_steps)

        maxci_mse_tst = eval_results['mse']
        maxci_ci_tst = eval_results['cindex']

        dest_path = best_model_dir
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for file in glob.glob(r'%s/model.ckpt-%d.*' % (output_dir, current_step)):
            print(file)
            shutil.copy(file, dest_path)

        step_list = []
        for file in glob.glob(r'%s/model.ckpt-*.meta' % (best_model_dir)):
            step_list.append(int(re.findall(r'\d+', file)[-1]))

        n_saved_models = len(step_list)
        n_keep = 5
        n_last = min(n_saved_models, n_keep)
        for del_index in np.sort(step_list)[:(n_saved_models - n_last)]:
            for f in glob.glob(r'%s/model.ckpt-%d.*' % (best_model_dir, del_index)):
                os.remove(f)

        with open('%s/checkpoint' % best_model_dir, 'wt') as handle:
            handle.write('model_checkpoint_path: "model.ckpt-%d"\n' % np.sort(step_list)[-1])

            for i in range(min(n_saved_models, n_keep)):
                handle.write('all_model_checkpoint_paths: "model.ckpt-%d"\n' % (np.sort(step_list)[i - n_last]))

    return maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst



def main(argv):
    del argv

    # init model class
    model = MbertPcnnModel(batch_size, dev_batch_size, 100, 1000,
                           args.bert_config_file, args.data_path+args.init_checkpoint,
                           args.learning_rate, num_train_steps, num_warmup_steps, args.use_tpu,
                           args.k1, args.k2, args.k3)

    tpu_cluster_resolver = None
    if args.use_tpu and args.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            args.tpu_name, zone=args.tpu_zone, project=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9


    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        session_config=config,
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=save_checkpoints_steps,
            num_shards=args.num_tpu_cores,
            per_host_input_for_training=is_per_host))


    model_fn = eval("model.model_fn_v%s" % args.model_version)
    # create classifier
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=dev_batch_size)

    input_fn_trn = model.input_fn_builder([i_trn], is_training=True)
    input_fn_dev = model.input_fn_builder([i_dev], is_training=False)
    input_fn_tst = model.input_fn_builder([i_tst], is_training=False)

    current_step = load_global_step_from_checkpoint_dir(output_dir)
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.',
                    num_train_steps,
                    num_train_steps / (num_trn_example / batch_size),
                    current_step)

    start_timestamp = time.time()  # This time will include compilation time

    minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst, \
    maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst = \
        restore_best_scores(0, best_model_dir_mse, best_model_dir_ci, estimator, input_fn_dev, input_fn_tst)

    last_time = start_timestamp
    while current_step < num_train_steps:
        next_checkpoint = min(current_step + save_checkpoints_steps, num_train_steps)
        estimator.train(input_fn=input_fn_trn, max_steps=next_checkpoint)

        checkpoint_time = int(time.time() - last_time)
        last_time = time.time()
        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))
        tf.logging.info('Starting to evaluate at step %d', next_checkpoint)
        eval_results = estimator.evaluate(input_fn=input_fn_dev, steps=dev_steps)
        tf.logging.info('Eval results at step %d: %s', next_checkpoint, eval_results)

        minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst = \
            check_improvement_mse(minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst,
                                  best_model_dir_mse, eval_results, current_step, estimator, input_fn_tst)

        maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst = \
            check_improvement_ci(maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst,
                                 best_model_dir_ci, eval_results, current_step, estimator, input_fn_tst)

        info_scores(current_step, minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst,
                    prefix='Current (sel_mse)',
                    checkpoint_time=checkpoint_time)
        info_scores(current_step, maxci_step, maxci_mse_dev, maxci_dev, maxci_mse_tst, maxci_ci_tst,
                    prefix='Current (sel_ci)',
                    checkpoint_time=checkpoint_time)
        current_step = next_checkpoint


    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    num_train_steps, elapsed_time)

    info_scores(current_step, minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst, prefix='Final (sel_mse)')
    info_scores(current_step, maxci_step, maxci_mse_dev, maxci_dev, maxci_mse_tst, maxci_ci_tst, prefix='Final(sel_ci)')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

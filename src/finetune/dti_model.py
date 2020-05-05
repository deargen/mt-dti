import tensorflow as tf
import six
from src.bert.modeling import DTIBertModel, BertConfig, get_assignment_map_from_checkpoint
from src.bert.optimization import create_optimizer, create_optimizer_v10

__author__ = 'Bonggun Shin'


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):

    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


class DeepConvolutionModelConfig(object):
    def __init__(self,
                 name,
                 vocab_size,
                 hidden_size=768,
                 hidden_act="gelu",
                 initializer_range=0.02,
                 filters=32,
                 kernel_size1=8,
                 kernel_size2=8,
                 kernel_size3=8):
        self.name = name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.filters = filters
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3


class DeepConvolutionModel(object):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 use_one_hot_embeddings=False,
                 scope=None):

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        with tf.variable_scope(scope, default_name=config.name):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

            with tf.variable_scope("cnn1"):
                xd_z = tf.layers.conv1d(
                    inputs=self.embedding_output,
                    filters=config.filters,
                    kernel_size=config.kernel_size1,
                    padding="valid",
                    activation=tf.nn.relu)

            with tf.variable_scope("cnn2"):
                xd_z = tf.layers.conv1d(
                    inputs=xd_z,
                    filters=config.filters * 2,
                    kernel_size=config.kernel_size2,
                    padding="valid",
                    activation=tf.nn.relu)

            with tf.variable_scope("cnn3"):
                xd_z = tf.layers.conv1d(
                    inputs=xd_z,
                    filters=config.filters * 3,
                    kernel_size=config.kernel_size3,
                    padding="valid",
                    activation=tf.nn.relu)

            with tf.variable_scope("pool"):
                xd_z = tf.layers.max_pooling1d(xd_z,
                                               pool_size=seq_length - (config.kernel_size1+config.kernel_size2+config.kernel_size3 - 3),
                                               strides=1,
                                               padding='VALID')
                xd_z = tf.squeeze(xd_z, 1)
        self.conv_z = xd_z


class DeepConvolutionModelWithoutEmbedding(object):
    def __init__(self,
                 config,
                 is_training,
                 vecs,
                 use_one_hot_embeddings=False,
                 scope=None):

        input_shape = get_shape_list(vecs, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        emb_dim = input_shape[2]

        with tf.variable_scope(scope, default_name=config.name):
            with tf.variable_scope("cnn1"):
                xd_z = tf.layers.conv1d(
                    inputs=vecs,
                    filters=config.filters,
                    kernel_size=config.kernel_size,
                    padding="valid",
                    activation=tf.nn.relu)

            with tf.variable_scope("cnn2"):
                xd_z = tf.layers.conv1d(
                    inputs=xd_z,
                    filters=config.filters * 2,
                    kernel_size=config.kernel_size,
                    padding="valid",
                    activation=tf.nn.relu)

            with tf.variable_scope("cnn3"):
                xd_z = tf.layers.conv1d(
                    inputs=xd_z,
                    filters=config.filters * 3,
                    kernel_size=config.kernel_size,
                    padding="valid",
                    activation=tf.nn.relu)

            with tf.variable_scope("pool"):
                xd_z = tf.layers.max_pooling1d(xd_z,
                                               pool_size=seq_length - (config.kernel_size * 3 - 3),
                                               strides=1,
                                               padding='VALID')
                xd_z = tf.squeeze(xd_z, 1)
        self.conv_z = xd_z


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    # g = tf.reduce_sum(tf.multiply(g, f))
    # f = tf.reduce_sum(f)

    g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
    f, update_op2 = tf.metrics.mean(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f), tf.group(update_op1, update_op2)



class DeepDTAModel(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def input_fn_builder(self, input_files,
                         max_molecule_length,
                         max_protein_length,
                         is_training,
                         num_cpu_threads=4
                         ):

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def _input_fn(params):
            batch_size = params["batch_size"]
            name_to_features = {
                "xd":
                    tf.FixedLenFeature([max_molecule_length], tf.int64),
                "xt":
                    tf.FixedLenFeature([max_protein_length], tf.int64),
                "y":
                    tf.FixedLenFeature([1], tf.float32),
            }

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            if is_training:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
                d = d.repeat()
                d = d.shuffle(buffer_size=len(input_files))

                # `cycle_length` is the number of parallel files that get read.
                cycle_length = min(num_cpu_threads, len(input_files))

                # `sloppy` mode means that the interleaving is not exact. This adds
                # even more randomness to the training pipeline.
                d = d.apply(
                    tf.contrib.data.parallel_interleave(
                        tf.data.TFRecordDataset,
                        sloppy=is_training,
                        cycle_length=cycle_length))
                d = d.shuffle(buffer_size=100)
            else:
                d = tf.data.TFRecordDataset(input_files)
                # Since we evaluate for a fixed number of steps we don't want to encounter
                # out-of-range exceptions.
                d = d.repeat(1)

            # We must `drop_remainder` on training because the TPU requires fixed
            # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
            # and we *don't* want to drop the remainder, otherwise we wont cover
            # every sample.
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=self.batch_size,
                    num_parallel_batches=num_cpu_threads,
                    drop_remainder=True))
            return d

        return _input_fn

    def model_fn(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """

        xd = features['xd']
        xt = features['xt']
        y = features['y']

        training = mode == tf.estimator.ModeKeys.TRAIN

        config_molecule = DeepConvolutionModelConfig("molecule", 73, 128)
        cnn_molecule = DeepConvolutionModel(config_molecule, training, xd)

        config_protein = DeepConvolutionModelConfig("protein", 30, 128, kernel_size1=12, kernel_size2=12, kernel_size3=12)
        cnn_protein = DeepConvolutionModel(config_protein, training, xt)

        concat_z = tf.concat([cnn_molecule.conv_z, cnn_protein.conv_z], 1)
        z = tf.layers.dense(concat_z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 512, activation='relu')

        predictions = tf.layers.dense(z, 1, kernel_initializer='normal')
        loss = tf.losses.mean_squared_error(y, predictions)

        self.y = y
        self.predictions = predictions

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        metrics_dict = {
            'cindex': cindex_score(y, predictions),
            'mse': tf.metrics.mean_squared_error(y, predictions)
        }

        metrics_tuple = (tf.metrics.mean_squared_error(y, predictions))

        def metric_fn(loss, y_true, y_pred):
            mean_loss = tf.metrics.mean(values=loss)

            g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
            g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

            f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
            f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

            # g = tf.reduce_sum(tf.multiply(g, f))
            # f = tf.reduce_sum(f)

            g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
            f, update_op2 = tf.metrics.mean(f)

            cindex = tf.where(tf.equal(g, 0), 0.0, g / f), tf.group(update_op1, update_op2)

            return {
                "mse": mean_loss,
                "cindex": cindex,
            }

        if mode == tf.estimator.ModeKeys.TRAIN:
            adam = tf.train.AdamOptimizer()
            grads = tf.gradients(loss, tvars)
            global_step = tf.train.get_or_create_global_step()
            train_op = adam.apply_gradients(zip(grads, tvars), global_step=global_step)

            eval_metrics = (metric_fn, [loss, y, predictions])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metrics=eval_metrics,
                # training_hooks=[logging_hook],
                training_hooks=[],
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [loss, y, predictions])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec



class MbertPcnnModel(object):
    def __init__(self, batch_size, dev_batch_size, max_molecule_length, max_protein_length,
                 bert_config_file, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps,
                 use_tpu, kernel_size1, kernel_size2, kernel_size3):
        self.batch_size = batch_size
        self.dev_batch_size = dev_batch_size
        self.bert_config_file = bert_config_file
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_tpu = use_tpu
        self.max_molecule_length = max_molecule_length
        self.max_protein_length = max_protein_length
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3

    def input_fn_builder(self, input_files,
                         is_training,
                         num_cpu_threads=4
                         ):

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def _input_fn(params):
            # batch_size = params["batch_size"]
            name_to_features = {
                "xd":
                    tf.FixedLenFeature([self.max_molecule_length], tf.int64),
                "xdm":
                    tf.FixedLenFeature([self.max_molecule_length], tf.int64),
                "xt":
                    tf.FixedLenFeature([self.max_protein_length], tf.int64),
                "xtm":
                    tf.FixedLenFeature([self.max_protein_length], tf.int64),
                "y":
                    tf.FixedLenFeature([1], tf.float32),
            }

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            if is_training:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
                d = d.repeat()
                d = d.shuffle(buffer_size=len(input_files))

                # `cycle_length` is the number of parallel files that get read.
                cycle_length = min(num_cpu_threads, len(input_files))

                # `sloppy` mode means that the interleaving is not exact. This adds
                # even more randomness to the training pipeline.
                d = d.apply(
                    tf.contrib.data.parallel_interleave(
                        tf.data.TFRecordDataset,
                        sloppy=is_training,
                        cycle_length=cycle_length))
                d = d.shuffle(buffer_size=100)
                batch_size = self.batch_size
            else:
                d = tf.data.TFRecordDataset(input_files)
                # Since we evaluate for a fixed number of steps we don't want to encounter
                # out-of-range exceptions.
                d = d.repeat(1)
                batch_size = self.dev_batch_size

            # We must `drop_remainder` on training because the TPU requires fixed
            # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
            # and we *don't* want to drop the remainder, otherwise we wont cover
            # every sample.
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    num_parallel_batches=num_cpu_threads,
                    drop_remainder=True))
            return d

        return _input_fn

    def model_fn_v1(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """
        tf.logging.info('*********************************** MbertPcnnModel V1 ***********************************')
        xd = features['xd']
        xd_mask = features['xdm']
        xt = features['xt']
        xt_mask = features['xtm']
        y = features['y']

        training = mode == tf.estimator.ModeKeys.TRAIN

        bert_config = BertConfig.from_json_file(self.bert_config_file)

        molecule_bert = DTIBertModel(
            config=bert_config,
            is_training=training,
            input_ids=xd,
            input_mask=xd_mask,
            use_one_hot_embeddings=False)

        molecule_representation = molecule_bert.get_pooled_output()

        config_protein = DeepConvolutionModelConfig("protein", 30, 128, kernel_size1=self.kernel_size1, kernel_size2=self.kernel_size2, kernel_size3=self.kernel_size3)
        cnn_protein = DeepConvolutionModel(config_protein, training, xt)

        concat_z = tf.concat([molecule_representation, cnn_protein.conv_z], 1)
        z = tf.layers.dense(concat_z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 512, activation='relu')

        predictions = tf.layers.dense(z, 1, kernel_initializer='normal')

        scaffold_fn = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            loss = tf.losses.mean_squared_error(y, predictions)

            # self.y = y
            # self.predictions = predictions

            tvars = tf.trainable_variables()

            initialized_variable_names = {}

            if self.init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
                if self.use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            # tf.logging.info("**** Trainable Variables ****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                     init_string)

            def metric_fn(loss, y_true, y_pred):
                mean_loss = tf.metrics.mean(values=loss)

                g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
                g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

                f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
                f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

                g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
                f, update_op2 = tf.metrics.mean(f)

                cindex = tf.where(tf.equal(g, 0), 0.0, g / f), tf.group(update_op1, update_op2)

                return {
                    "mse": mean_loss,
                    "cindex": cindex,
                }

            eval_metrics = (metric_fn, [loss, y, predictions])

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = create_optimizer(
                    loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, self.use_tpu)

            else:
                train_op = None

        else:
            loss = None
            train_op = None
            eval_metrics = None

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={
                "predictions": predictions,
                "gold": y,
                "xd": xd,
                "xt": xt,
            },
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn,
            export_outputs={'out': tf.estimator.export.PredictOutput(
                {"predictions": predictions})})

        return output_spec


    def model_fn_v11(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """
        tf.logging.info('*********************************** MbertPcnnModel V11 ***********************************')
        xd = features['xd']
        xd_mask = features['xdm']
        xt = features['xt']
        xt_mask = features['xtm']
        y = features['y']

        training = mode == tf.estimator.ModeKeys.TRAIN

        bert_config = BertConfig.from_json_file(self.bert_config_file)

        molecule_bert = DTIBertModel(
            config=bert_config,
            is_training=training,
            input_ids=xd,
            input_mask=xd_mask,
            use_one_hot_embeddings=False)

        molecule_representation = molecule_bert.get_pooled_output()

        config_protein = DeepConvolutionModelConfig("protein", 30, 128, kernel_size1=self.kernel_size1, kernel_size2=self.kernel_size2, kernel_size3=self.kernel_size3)
        cnn_protein = DeepConvolutionModel(config_protein, training, xt)

        concat_z = tf.concat([molecule_representation, cnn_protein.conv_z], 1)
        z = tf.layers.dense(concat_z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 512, activation='relu')

        predictions = tf.layers.dense(z, 1, kernel_initializer='normal')

        scaffold_fn = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            loss = tf.losses.mean_squared_error(y, predictions)

            # self.y = y
            # self.predictions = predictions

            tvars = tf.trainable_variables()

            initialized_variable_names = {}

            if self.init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
                if self.use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            # tf.logging.info("**** Trainable Variables ****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                     init_string)

            def metric_fn(loss, y_true, y_pred):
                mean_loss = tf.metrics.mean(values=loss)

                g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
                g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

                f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
                f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

                g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
                f, update_op2 = tf.metrics.mean(f)

                cindex = tf.where(tf.equal(g, 0), 0.0, g / f), tf.group(update_op1, update_op2)

                return {
                    "mse": mean_loss,
                    "cindex": cindex,
                }

            eval_metrics = (metric_fn, [loss, y, predictions])



            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = create_optimizer_v10(
                    loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, self.use_tpu)

            else:
                train_op = None

        else:
            loss = None
            train_op = None
            eval_metrics = None

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={
                "predictions": predictions,
                "gold": y,
                "xd": xd,
                "xt": xt,
            },
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn,
            export_outputs={'out': tf.estimator.export.PredictOutput(
                {"predictions": predictions})})

        return output_spec


    def model_fn_v2(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """
        tf.logging.info('*********************************** MbertPcnnModel V2 ***********************************')

        xd = features['xd']
        xd_mask = features['xdm']
        xt = features['xt']
        # xt_mask = features['xtm']
        y = features['y']

        training = mode == tf.estimator.ModeKeys.TRAIN

        bert_config = BertConfig.from_json_file(self.bert_config_file)

        molecule_bert = DTIBertModel(
            config=bert_config,
            is_training=training,
            input_ids=xd,
            input_mask=xd_mask,
            use_one_hot_embeddings=False)

        molecule_tokens = molecule_bert.get_all_encoder_layers()[-1]

        config_molecule = DeepConvolutionModelConfig("molecule", 73, 128)
        cnn_molecule = DeepConvolutionModelWithoutEmbedding(config_molecule, training, molecule_tokens)

        config_protein = DeepConvolutionModelConfig("protein", 30, 128, kernel_size1=self.kernel_size1, kernel_size2=self.kernel_size2, kernel_size3=self.kernel_size3)
        cnn_protein = DeepConvolutionModel(config_protein, training, xt)

        concat_z = tf.concat([cnn_molecule.conv_z, cnn_protein.conv_z], 1)
        z = tf.layers.dense(concat_z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 512, activation='relu')

        predictions = tf.layers.dense(z, 1, kernel_initializer='normal')
        loss = tf.losses.mean_squared_error(y, predictions)

        self.y = y
        self.predictions = predictions

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
            if self.use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)

        def metric_fn(loss, y_true, y_pred):
            mean_loss = tf.metrics.mean(values=loss)

            g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
            g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

            f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
            f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

            g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
            f, update_op2 = tf.metrics.mean(f)

            cindex = tf.where(tf.equal(g, 0), 0.0, g / f), tf.group(update_op1, update_op2)

            return {
                "mse": mean_loss,
                "cindex": cindex,
            }

        if mode == tf.estimator.ModeKeys.TRAIN:
            adam = tf.train.AdamOptimizer()
            grads = tf.gradients(loss, tvars)
            global_step = tf.train.get_or_create_global_step()
            train_op = adam.apply_gradients(zip(grads, tvars), global_step=global_step)

            eval_metrics = (metric_fn, [loss, y, predictions])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metrics=eval_metrics,
                # training_hooks=[logging_hook],
                training_hooks=[],
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [loss, y, predictions])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    def model_fn_v3(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """
        tf.logging.info('*********************************** MbertPcnnModel V3 ***********************************')
        xd = features['xd']
        xd_mask = features['xdm']
        xt = features['xt']
        # xt_mask = features['xtm']
        y = features['y']

        training = mode == tf.estimator.ModeKeys.TRAIN

        bert_config = BertConfig.from_json_file(self.bert_config_file)

        molecule_bert = DTIBertModel(
            config=bert_config,
            is_training=training,
            input_ids=xd,
            input_mask=xd_mask,
            use_one_hot_embeddings=False)

        molecule_tokens = molecule_bert.get_all_encoder_layers()[-1]

        molecule_tokens_shape = get_shape_list(molecule_tokens)

        molecule_tokens = tf.reshape(molecule_tokens, [molecule_tokens_shape[0]]+[molecule_tokens_shape[1]*molecule_tokens_shape[2]])


        config_protein = DeepConvolutionModelConfig("protein", 30, 128, kernel_size1=self.kernel_size1, kernel_size2=self.kernel_size2, kernel_size3=self.kernel_size3)
        cnn_protein = DeepConvolutionModel(config_protein, training, xt)

        concat_z = tf.concat([molecule_tokens, cnn_protein.conv_z], 1)
        z = tf.layers.dense(concat_z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 512, activation='relu')

        predictions = tf.layers.dense(z, 1, kernel_initializer='normal')
        loss = tf.losses.mean_squared_error(y, predictions)

        self.y = y
        self.predictions = predictions

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
            if self.use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)

        def metric_fn(loss, y_true, y_pred):
            mean_loss = tf.metrics.mean(values=loss)

            g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
            g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

            f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
            f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

            g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
            f, update_op2 = tf.metrics.mean(f)

            cindex = tf.where(tf.equal(g, 0), 0.0, g / f), tf.group(update_op1, update_op2)

            return {
                "mse": mean_loss,
                "cindex": cindex,
            }

        if mode == tf.estimator.ModeKeys.TRAIN:
            adam = tf.train.AdamOptimizer()
            grads = tf.gradients(loss, tvars)
            global_step = tf.train.get_or_create_global_step()
            train_op = adam.apply_gradients(zip(grads, tvars), global_step=global_step)

            eval_metrics = (metric_fn, [loss, y, predictions])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metrics=eval_metrics,
                # training_hooks=[logging_hook],
                training_hooks=[],
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [loss, y, predictions])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    def model_fn_v4(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """
        tf.logging.info('*********************************** MbertPcnnModel V4 ***********************************')
        xd = features['xd']
        xd_mask = features['xdm']
        xt = features['xt']
        xt_mask = features['xtm']
        y = features['y']

        training = mode == tf.estimator.ModeKeys.TRAIN

        bert_config = BertConfig.from_json_file(self.bert_config_file)

        molecule_bert = DTIBertModel(
            config=bert_config,
            is_training=training,
            input_ids=xd,
            input_mask=xd_mask,
            use_one_hot_embeddings=False)

        molecule_representation = molecule_bert.get_pooled_output()

        config_protein = DeepConvolutionModelConfig("protein", 30, 128, kernel_size1=self.kernel_size1, kernel_size2=self.kernel_size2, kernel_size3=self.kernel_size3)
        cnn_protein = DeepConvolutionModel(config_protein, training, xt)

        concat_z = tf.concat([molecule_representation, cnn_protein.conv_z], 1)
        z = tf.layers.dense(concat_z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 512, activation='relu')

        predictions = tf.layers.dense(z, 1, kernel_initializer='normal')

        scaffold_fn = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            loss = tf.losses.mean_squared_error(y, predictions)

            # self.y = y
            # self.predictions = predictions

            tvars = tf.trainable_variables()

            initialized_variable_names = {}

            if self.init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
                if self.use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            # tf.logging.info("**** Trainable Variables ****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                     init_string)

            def metric_fn(loss, y_true, y_pred):
                mean_loss = tf.metrics.mean(values=loss)

                g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
                g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

                f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
                f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

                g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
                f, update_op2 = tf.metrics.mean(f)

                cindex = tf.where(tf.equal(g, 0), 0.0, g / f), tf.group(update_op1, update_op2)

                return {
                    "mse": mean_loss,
                    "cindex": cindex,
                }

            eval_metrics = (metric_fn, [loss, y, predictions])

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = create_optimizer(
                    loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, self.use_tpu)

            else:
                train_op = None

        else:
            loss = None
            train_op = None
            eval_metrics = None

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={
                "predictions": predictions,
                "gold": y,
                "xd": xd,
                "xt": xt,
            },
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn,
            export_outputs={'out': tf.estimator.export.PredictOutput(
                {"predictions": predictions})})

        return output_spec


    def model_fn_v14(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """
        tf.logging.info('*********************************** MbertPcnnModel V14 ***********************************')
        xd = features['xd']
        xd_mask = features['xdm']
        xt = features['xt']
        xt_mask = features['xtm']
        y = features['y']

        training = mode == tf.estimator.ModeKeys.TRAIN

        bert_config = BertConfig.from_json_file(self.bert_config_file)

        molecule_bert = DTIBertModel(
            config=bert_config,
            is_training=training,
            input_ids=xd,
            input_mask=xd_mask,
            use_one_hot_embeddings=False)

        molecule_representation = molecule_bert.get_pooled_output()

        config_protein = DeepConvolutionModelConfig("protein", 30, 128, kernel_size1=self.kernel_size1,
                                                    kernel_size2=self.kernel_size2, kernel_size3=self.kernel_size3)
        cnn_protein = DeepConvolutionModel(config_protein, training, xt)

        concat_z = tf.concat([molecule_representation, cnn_protein.conv_z], 1)
        z = tf.layers.dense(concat_z, 1024, activation='relu')
        z = tf.layers.dropout(z, rate=0.1)
        z = tf.layers.dense(z, 512, activation='relu')

        predictions = tf.layers.dense(z, 1, kernel_initializer='normal')

        scaffold_fn = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            loss = tf.losses.mean_squared_error(y, predictions)

            # self.y = y
            # self.predictions = predictions

            tvars = tf.trainable_variables()

            initialized_variable_names = {}

            if self.init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
                if self.use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            # tf.logging.info("**** Trainable Variables ****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                     init_string)

            def metric_fn(loss, y_true, y_pred):
                mean_loss = tf.metrics.mean(values=loss)

                g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
                g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

                f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
                f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

                g, update_op1 = tf.metrics.mean(tf.multiply(g, f))
                f, update_op2 = tf.metrics.mean(f)

                cindex = tf.where(tf.equal(g, 0), 0.0, g / f), tf.group(update_op1, update_op2)

                return {
                    "mse": mean_loss,
                    "cindex": cindex,
                }

            eval_metrics = (metric_fn, [loss, y, predictions])

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = create_optimizer_v10(
                    loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, self.use_tpu)

            else:
                train_op = None

        else:
            loss = None
            train_op = None
            eval_metrics = None

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={
                "predictions": predictions,
                "gold": y,
                "xd": xd,
                "xt": xt,
            },
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn,
            export_outputs={'out': tf.estimator.export.PredictOutput(
                {"predictions": predictions})})

        return output_spec


class BertDTIModel(object):
    def __init__(self, batch_size, bert_config_file, init_checkpoint, use_tpu):
        self.batch_size = batch_size
        self.bert_config_file = bert_config_file
        self.init_checkpoint = init_checkpoint
        self.use_tpu = use_tpu


def load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0
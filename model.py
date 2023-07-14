import absl
from ipdb import set_trace as ipdb
from IPython import embed as ipython
    # ipdb()
import keras_tuner
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from tensorflow.keras import layers
from tensorflow_metadata.proto import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_data_validation as tfdv

from typing import List
from tfx import v1 as tfx
# from tfx.examples.penguin import penguin_utils_base as base
from tfx_bsl.public import tfxio
import logging
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options



_LABEL_KEY='fare_amount'

def make_serving_signatures(model,
                            tf_transform_output: tft.TFTransformOutput):
    """Returns the serving signatures.
  
    Args:
      model: the model function to apply to the transformed features.
      tf_transform_output: The transformation to apply to the serialized
        tf.Example.
  
    Returns:
      The signatures to use for saving the mode. The 'serving_default' signature
      will be a concrete function that takes a batch of unspecified length of
      serialized tf.Example, parses them, transformes the features and
      then applies the model. The 'transform_features' signature will parses the
      example and transforms the features.
    """
  
    # We need to track the layers in the model in order to save it.
    # TODO(b/162357359): Revise once the bug is resolved.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
      """Returns the output to be used in the serving signature."""
      raw_feature_spec = tf_transform_output.raw_feature_spec()
      # Remove label feature since these will not be present at serving time.
      raw_feature_spec.pop(_LABEL_KEY)
      raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
      transformed_features = model.tft_layer(raw_features)
      logging.info('serve_transformed_features = %s', transformed_features)

      outputs = model(transformed_features)
      # TODO(b/154085620): Convert the predicted labels from the model using a
      # reverse-lookup (opposite of transform.py).
      return {'outputs': outputs}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return {
      'serving_default': serve_tf_examples_fn,
      'transform_features': transform_features_fn
    }

NBUCKETS=10
# {{{ def _make_keras_model(model_inputs, dnn_inputs, keras_preprocessing_inputs):
def _make_keras_model(fn_args: tfx.components.FnArgs, model_inputs, dnn_inputs, keras_preprocessing_inputs, hparms=None):
    if hparms:
        hparams = hparms
    """
    Build, compile, and return a model that uses keras preprocessing
    layers in conjunction with the preprocessing already done in tensorflow
    transform.
    """
    bucketed_pickup_longitude_intermediary = keras_preprocessing_inputs[
        'bucketed_pickup_longitude']
    bucketed_pickup_latitude_intermediary = keras_preprocessing_inputs[
        'bucketed_pickup_latitude']
    bucketed_dropoff_longitude_intermediary = keras_preprocessing_inputs[
        'bucketed_dropoff_longitude']
    bucketed_dropoff_latitude_intermediary = keras_preprocessing_inputs[
        'bucketed_dropoff_latitude']
    hash_pickup_crossing_layer_intermediary = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='int', num_bins=NBUCKETS**2, )
    hashed_pickup_intermediary = hash_pickup_crossing_layer_intermediary(
        (bucketed_pickup_longitude_intermediary, bucketed_pickup_latitude_intermediary))
    hash_dropoff_crossing_layer_intermediary = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='int', num_bins=NBUCKETS**2, )
    hashed_dropoff_intermediary = hash_dropoff_crossing_layer_intermediary(
        (bucketed_dropoff_longitude_intermediary, bucketed_dropoff_latitude_intermediary))
    hash_trip_crossing_layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='one_hot', num_bins=NBUCKETS ** 3, name="hash_trip_crossing_layer")
    hashed_trip = hash_trip_crossing_layer(
        (hashed_pickup_intermediary,
         hashed_dropoff_intermediary))
    hashed_trip_and_time = dnn_inputs["hashed_trip_and_time"]

    stacked_inputs = tf.concat(tf.nest.flatten(dnn_inputs), axis=1)
    if hparms:
        units = hparams.get('units')
        number_of_layers = hparams.get('number_of_layers')
        regularization_level = hparams.get('regularization_level')
    else:
        units = 8
        number_of_layers = 2
        regularization_level = None


    if regularization_level:
        if regularization_level == 1:
            regularizer=keras.regularizers.L1(1e-1)
        elif regularization_level == 2:
            regularizer=keras.regularizers.L2(1e-1)
        else:
            regularizer=None

    else:
        regularizer=None

    x = layers.Dense(units=units, activation='relu', kernel_regularizer=regularizer)(stacked_inputs)

    for layer_number in range(number_of_layers - 1):
        x = layers.Dense(units=units, activation='relu', kernel_regularizer=regularizer)(x)

    kp1 = layers.Dense(units=units, activation='relu',
                       name='kp1', kernel_regularizer=regularizer)(hashed_trip)

    trip_locations_embedding_layer = tf.keras.layers.Embedding(
        input_dim=NBUCKETS ** 3,
        output_dim=int(NBUCKETS ** 1.5),
        name="trip_locations_embedding_layer")
    trip_embedding_layer = tf.keras.layers.Embedding(
        input_dim=(NBUCKETS ** 3) * 4,
        output_dim=int(NBUCKETS ** 1.5),
        name="trip_embedding_layer")
    trip_locations_embedding = trip_locations_embedding_layer(hashed_trip)
    trip_embedding = trip_embedding_layer(hashed_trip_and_time)
    flatten_trip_location_embeddings = layers.Flatten()(trip_locations_embedding)
    flatten_trip_embeddings = layers.Flatten()(trip_embedding)
    # et1 = layers.LSTM(32, activation='tanh', name='et1')(
    #     trip_locations_embedding)
    # et2 = layers.LSTM(32, activation='tanh', name='et2')(trip_embedding)
    et1 = layers.Dense(units=64, activation='relu')(flatten_trip_location_embeddings)
    et2 = layers.Dense(units=64, activation='relu')(flatten_trip_embeddings)
    merge_layer = layers.concatenate([x, kp1, et1, et2])
    nht3 = layers.Dense(units=units)(merge_layer)

    # final output is a linear activation because this is regression
    output = layers.Dense(1, activation='linear', name='fare')(nht3)
    new_model = tf.keras.Model(model_inputs, output)
    new_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return new_model



# }}}
def map_features_and_labels(example, LABEL_NAME):
    label = example.pop(LABEL_NAME)
    return example, label

def input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY ),
      tf_transform_output.transformed_metadata.schema).repeat()

def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    # ipdb()
    schema = tfdv.load_schema_text(fn_args.schema_path)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    # <<<< #
    feature_spec = tf_transform_output.transformed_feature_spec()
    #schema = schema_utils.schema_from_feature_spec(feature_spec)
    BATCH_SIZE=256
    _ = feature_spec.pop(_LABEL_KEY)
    inputs = {}
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = tf.keras.layers.Input(
            shape=(1,), name=key, dtype=spec.dtype, sparse=True)
        elif isinstance(spec, tf.io.FixedLenFeature):
            inputs[key] = tf.keras.layers.Input(
            shape=spec.shape, name=key, dtype=spec.dtype)
        else:
            raise ValueError('Spec type is not supported: ', key, spec)


    """
    Splits model inputs into two: one part goes into the dense
    layer, the other part goes to keras preprocessing layers.
    """
    dnn_input_names = [
    'hashed_trip_and_time',
    'day_of_week',
    'euclidean',
    'pickup_longitude',
    'dropoff_location',
    'dropoff_longitude',
    'pickup_location',
    'passenger_count',
    'dropoff_latitude',
    'hour_of_day',
    'pickup_latitude',
    'hashed_trip',
    'hour_of_day_of_week',
    ]

    keras_preprocessing_input_names = [
    'bucketed_dropoff_longitude',
    'bucketed_dropoff_latitude',
    'bucketed_pickup_latitude',
    'bucketed_pickup_longitude',
    ]
    dnn_inputs = {}
    dnn_inputs = {
        input_name: inputs[input_name]
        for input_name in dnn_input_names}
    keras_preprocessing_inputs = {}
    keras_preprocessing_inputs = {
        input_name: inputs[input_name]
        for input_name in keras_preprocessing_input_names}
    # return dnn_inputs, keras_preprocessing_inputs
    model = _make_keras_model(fn_args, inputs, dnn_inputs, keras_preprocessing_inputs)
    train_dataset = input_fn(
          fn_args.train_files,
          fn_args.data_accessor,
          tf_transform_output,
          BATCH_SIZE)

    eval_dataset = input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      BATCH_SIZE)

    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    # else:
        # This is a shown case when hyperparameters is decided and Tuner is removed
        # from the pipeline. User can also inline the hyperparameters directly in
        # _build_keras_model.
        # hparams = _get_hyperparameters()
    # absl.logging.info('HyperParameters for training: %s' % hparams.get_config())

      # mirrored_strategy = tf.distribute.MirroredStrategy()
      # with mirrored_strategy.scope():
      #   model = _make_keras_model(hparams)

      # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
          log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(
          train_dataset,
          steps_per_epoch=fn_args.train_steps,
          validation_data=eval_dataset,
          validation_steps=fn_args.eval_steps,
          callbacks=[tensorboard_callback])

    signatures = make_serving_signatures(model, tf_transform_output)
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

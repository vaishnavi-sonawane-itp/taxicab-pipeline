import absl
from ipdb import set_trace as ipdb
from IPython import embed as ipython
from model import _make_keras_model
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
from functools import partial
#tfx.proto.tuner_pb2.TuneArgs

def _get_hyperparameters():
    hp = keras_tuner.HyperParameters()
    hp.Choice('learning_rate', [1e-1, 1e-2], default=1e-1)
    hp.Choice('units', [32, 64], default=32)
    hp.Choice('number_of_layers', [16, 32], default=16)
    hp.Choice('regularization_level', [0, 1, 2], default=2)
    return hp

_LABEL_KEY='fare_amount'
BATCH_SIZE=256
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
    # ipdb()
    # fn_args.hyperparameters = _get_hyperparameters()
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    feature_spec = tf_transform_output.transformed_feature_spec()
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
    _inject_create_model = partial(_make_keras_model, fn_args, inputs, dnn_inputs, keras_preprocessing_inputs, ) 
    tuner = keras_tuner.RandomSearch(
        _inject_create_model,
        max_trials=8,
        hyperparameters = _get_hyperparameters(),
        allow_new_entries=False,
        objective=keras_tuner.Objective('val_mse', 'min'),
        directory=fn_args.working_dir,
        project_name='synthetic')
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
    return tfx.components.TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
        })

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

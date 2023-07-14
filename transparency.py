# {{{ Imports
import tensorflow as tf
import glob
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tensorflow import keras
LABEL_NAME='fare_amount'
# }}}
# {{{ Create TFRecordDataset
example_gen_glob='./pipeline/CsvExampleGen/examples/1/Split-eval/data_*gz'
filenames=glob.glob(example_gen_glob)
tfrecord_ds = tf.data.TFRecordDataset(filenames=filenames, compression_type='GZIP')
# }}}
# {{{ Map to Dataset using schema and feature spec to decode
original_schema_path='./pipeline/SchemaGen/schema/3/schema.pbtxt'
original_schema = tfdv.load_schema_text(original_schema_path)
original_feature_spec_list = tft.tf_metadata.schema_utils.schema_as_feature_spec(original_schema)
original_feature_spec = original_feature_spec_list[0]
def decode_fn(example, feature_spec):
    return tf.io.parse_single_example(serialized=example, features=feature_spec, )
mapped_ds = tfrecord_ds.map(lambda X: decode_fn(X, original_feature_spec))


def datemap(example):
    original_string = example['pickup_datetime']
    string_tensor = tf.strings.split(input=original_string, sep=' ')
    ymd_string =  string_tensor[0][0][0]
    hms_string = string_tensor[0][0][1]
    y_m_d_string = tf.strings.split(input=ymd_string, sep='-')
    y_string = y_m_d_string[0]
    m_string = y_m_d_string[1]
    d_string = y_m_d_string[2]
    h_min_s_string = tf.strings.split(input=hms_string, sep=':')
    h_string = h_min_s_string[0]
    min_string = h_min_s_string[1]
    s_string = h_min_s_string[2]
    hour = tf.expand_dims(tf.reshape(tf.strings.to_number(h_string), shape=(1,)), axis=1)
    minute = tf.expand_dims(tf.reshape(tf.strings.to_number(min_string), shape=(1,)), axis=1)
    second = tf.expand_dims(tf.reshape(tf.strings.to_number(s_string), shape=(1,)), axis=1)
    year = tf.expand_dims(tf.reshape(tf.strings.to_number(y_string), shape=(1,)), axis=1)
    month = tf.expand_dims(tf.reshape(tf.strings.to_number(m_string), shape=(1,)), axis=1)
    date = tf.expand_dims(tf.reshape(tf.strings.to_number(d_string), shape=(1,)), axis=1)
    four = tf.constant(4, dtype=tf.int32)
    one = tf.constant(1, dtype=tf.int32)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_dict = {}
    days_till_eom = 0
    month_dict[0] = days_till_eom
    for month_number in range(12):
        days_till_eom = days_till_eom + days_in_month[month_number]
        month_dict[month_number + 1] = days_till_eom
    # ipdb()
    keys_tensor = tf.constant(list(range(13)))
    vals_tensor = tf.constant(list(month_dict.values()), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
    default_value=-1)
    month = tf.cast(month, dtype=tf.int32)
    days_since_1970 = 365*(year - 1970) + ((year - 1968)//4) + date + tf.cast(table.lookup(month - 1), tf.float32) - tf.cast(one, tf.float32)
    seconds_since_1970 = days_since_1970*24*3600 + (hour*3600) + minute*60 + second
    return seconds_since_1970
next(iter(mapped_ds.batch(1).map(datemap)))

    # ipdb()
    # seconds_since_1970 = tf.cast(
    #     tfa.text.parse_time(
    #         example["pickup_datetime"],
    #         "%Y-%m-%d %H:%M:%S %Z",
    #         output_unit="SECOND"),
    #     tf.float32)
    # s = month - 1
    # days_since_1970 = tf.constant(month_dict[s], dtype=tf.int32)
    # days_since_1970 = table.lookup(month - 1)
    # seconds_since_1970_return = tf.expand_dims(seconds_since_1970, axis=0)
    # ipdb()






def datemap(example):
    original_string = example['pickup_datetime']
    string_tensor = tf.strings.split(input=original_string, sep=' ')
    ymd_string =  string_tensor[0][0]
    hms_string = string_tensor[0][1]
    y_m_d_string = tf.strings.split(input=ymd_string, sep='-')
    y_string = y_m_d_string[0]
    m_string = y_m_d_string[1]
    d_string = y_m_d_string[2]
    h_min_s_string = tf.strings.split(input=hms_string, sep=':')
    h_string = h_min_s_string[0]
    min_string = h_min_s_string[1]
    s_string = h_min_s_string[2]
    hour =  tf.reshape(tf.strings.to_number(h_string, out_type=tf.int32), shape=(1,))
    minute =  tf.reshape(tf.strings.to_number(min_string, out_type=tf.int32), shape=(1,))
    second =  tf.reshape(tf.strings.to_number(s_string, out_type=tf.int32), shape=(1,))
    year =  tf.reshape(tf.strings.to_number(y_string, out_type=tf.int32), shape=(1,))
    month =  tf.reshape(tf.strings.to_number(m_string, out_type=tf.int32), shape=(1,))
    date =  tf.reshape(tf.strings.to_number(d_string, out_type=tf.int32), shape=(1,))
    four = tf.constant(4, dtype=tf.int32)
    one = tf.constant(1, dtype=tf.int32)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_dict = {}
    days_till_eom = 0
    month_dict[0] = days_till_eom
    for month_number in range(12):
        days_till_eom = days_till_eom + days_in_month[month_number]
        month_dict[month_number + 1] = days_till_eom
    # ipdb()
    keys_tensor = tf.constant(list(range(13)))
    vals_tensor = tf.constant(list(month_dict.values()), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
    default_value=-1)
    days_since_1970 = 365*(year - 1970) + ((year - 1968)//four) + date + table.lookup(month - 1) - one
    # seconds_since_1970 = days_since_1970*24*3600 + (hour*3600) + minute*60 + second
    # seconds_since_1970 = tf.cast(seconds_since_1970, dtype=tf.float32)
    seconds_since_1970 = tf.cast(
        tfa.text.parse_time(
            example["pickup_datetime"],
            "%Y-%m-%d %H:%M:%S %Z",
            output_unit="SECOND"),
        tf.float32)
    # s = month - 1
    # days_since_1970 = tf.constant(month_dict[s], dtype=tf.int32)
    # days_since_1970 = table.lookup(month - 1)
    return seconds_since_1970



output = mapped_ds.map(datemap)
next(iter(output))


from ipdb import set_trace as ipdb
from IPython import embed as ipython
    # ipdb()


# }}}
# {{{ Split Features and Labels
def map_features_and_labels(example, label_name):
    label = example.pop(label_name)
    return example, label
tuple_ds = mapped_ds.map(lambda X: map_features_and_labels(X, LABEL_NAME))
# }}}
# {{{ Transform data to enriched version
transform_output_location='./pipeline/Transform/transform_graph/16/'
tf_transform_output = tft.TFTransformOutput(transform_output_location)
tft_layer = tf_transform_output.transform_features_layer()
tft_layer(next(iter(mapped_ds.batch(1))))
# }}}
# {{{ Create Model Inputs
feature_spec = original_feature_spec.copy()
_ = feature_spec.pop(LABEL_NAME)
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
    # }}}
# {{{ Create Preprocessing Model
preprocessing_model = tf.keras.Model(inputs, tft_layer(inputs))
# preprocessing_model(next(iter(mapped_ds)))
preprocessing_model(next(iter(mapped_ds.batch(1))))
# }}}
# {{{ Load Models that were previously trained
pushed_model  = keras.models.load_model('./pipeline/Pusher/pushed_model/10/')
trained_model = keras.models.load_model('./pipeline/Trainer/model/7/Format-Serving/')
stamped_model = keras.models.load_model('./pipeline/InfraValidator/blessing/9/stamped_model/')
# }}}
# {{{ Predict from raw data
trained_model.predict(preprocessing_model(next(iter(mapped_ds))))
# }}}
# {{{ Create end to end model which includes preprocessing
x = preprocessing_model(inputs)
outputs = trained_model(x)
end_to_end_model = tf.keras.Model(inputs, outputs)
end_to_end_model.predict(next(iter(mapped_ds)))
# }}}


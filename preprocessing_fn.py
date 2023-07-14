import tensorflow as tf
import tensorflow_transform as tft
NBUCKETS = 10
from ipdb import set_trace as ipdb
from IPython import embed as ipython


def new_datemap(date_string):
    original_string = date_string
    y_string = tf.strings.substr(original_string, 0, 4, name="year_string")
    m_string = tf.strings.substr(original_string, 5, 2, name="month_string")
    d_string = tf.strings.substr(original_string, 8, 2, name="year_string")
    h_string = tf.strings.substr(original_string, 11, 2, name="year_string")
    min_string = tf.strings.substr(original_string, 14, 2, name="year_string")
    s_string = tf.strings.substr(original_string, 17, 2, name="year_string")
    def to_number(x):
        return tf.strings.to_number(x)
    year = to_number(y_string)
    month = to_number(m_string)
    date = to_number(d_string)
    hour = to_number(h_string)
    minute = to_number(min_string)
    second = to_number(s_string)
    four = tf.constant(4, dtype=tf.int32)
    one = tf.constant(1, dtype=tf.int32)
    with tf.init_scope():
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_dict = {}
        days_till_eom = 0
        month_dict[0] = days_till_eom
        for month_number in range(12):
            days_till_eom = days_till_eom + days_in_month[month_number]
            month_dict[month_number + 1] = days_till_eom
        keys_tensor = tf.constant(list(range(13)))
        vals_tensor = tf.constant(list(month_dict.values()), dtype=tf.int32)
        table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1)
    month = tf.cast(month, dtype=tf.int32)
    days_since_1970 = 365*(year - 1970) + ((year - 1968)//4) + date + tf.cast(table.lookup(month - 1), tf.float32) - tf.cast(one, tf.float32)
    seconds_since_1970 = days_since_1970*24*3600 + (hour*3600) + minute*60 + second
    return seconds_since_1970

def preprocessing_fn(inputs):
    """
    Preprocess input columns into transformed features. This is what goes
    into tensorflow transform/apache beam.
    """
    # Since we are modifying some features and leaving others unchanged, we
    # start by setting `outputs` to a copy of `inputs.
    transformed = inputs.copy()
    del(transformed["key"])
    transformed['passenger_count'] = tft.scale_to_0_1(
        inputs['passenger_count'])
    # cannot use the below in tft as managing those learned values need
    # to be
    # managed carefully
    # normalizer = tf.keras.layers.Normalization(axis=None,
    # name="passenger_count_normalizer")
    # normalizer.adapt(inputs['passenger_count'])
    # transformed['other_passenger_count'] = normalizer(
    #               inputs['passenger_count'])
    for col in ['dropoff_longitude', 'dropoff_latitude']:
        transformed[col] = tft.sparse_tensor_to_dense_with_shape(inputs[col], default_value=tft.mean(inputs[col]), shape=[None, 1]) #You can make this more robust by using the shape from the feature spec
    for lon_col in ['pickup_longitude', 'dropoff_longitude']:
        # transformed[lon_col] = scale_longitude(inputs[lon_col])
        transformed[lon_col] = (transformed[lon_col] + 78) / 8.
    for lat_col in ['pickup_latitude', 'dropoff_latitude']:
        transformed[lat_col] = (transformed[lat_col] - 37) / 8.
    position_difference = tf.square(
        transformed["dropoff_longitude"] -
        transformed["pickup_longitude"])
    position_difference += tf.square(
        transformed["dropoff_latitude"] -
        transformed["pickup_latitude"])
    transformed['euclidean'] = tf.sqrt(position_difference)
    lat_lon_buckets = [
        bin_edge / NBUCKETS
        for bin_edge in range(0, NBUCKETS)]

    transformed['bucketed_pickup_longitude'] = tft.apply_buckets(
        transformed["pickup_longitude"],
        bucket_boundaries=tf.constant([lat_lon_buckets]))
    transformed["bucketed_pickup_latitude"] = tft.apply_buckets(
        transformed['pickup_latitude'],
        bucket_boundaries=tf.constant([lat_lon_buckets]))

    transformed['bucketed_dropoff_longitude'] = tft.apply_buckets(
        transformed["dropoff_longitude"],
        bucket_boundaries=tf.constant([lat_lon_buckets]))
    transformed['bucketed_dropoff_latitude'] = tft.apply_buckets(
        transformed["dropoff_latitude"],
        bucket_boundaries=tf.constant([lat_lon_buckets]))

    # transformed["pickup_cross"]=tf.sparse.cross(
    # inputs=[transformed['pickup_latitude_apply_buckets'],
    # transformed['pickup_longitude_apply_buckets']])
    hash_pickup_crossing_layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='one_hot', num_bins=NBUCKETS**2, name='hash_pickup_crossing_layer')
    transformed['pickup_location'] = hash_pickup_crossing_layer(
        (transformed['bucketed_pickup_latitude'],
         transformed['bucketed_pickup_longitude']))
    hash_dropoff_crossing_layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='one_hot', num_bins=NBUCKETS**2,
        name='hash_dropoff_crossing_layer')
    transformed['dropoff_location'] = hash_dropoff_crossing_layer(
        (transformed['bucketed_dropoff_latitude'],
         transformed['bucketed_dropoff_longitude']))

    hash_pickup_crossing_layer_intermediary = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='int', num_bins=NBUCKETS**2, )
    hashed_pickup_intermediary = hash_pickup_crossing_layer_intermediary(
        (transformed['bucketed_pickup_longitude'],
         transformed['bucketed_pickup_latitude']))
    hash_dropoff_crossing_layer_intermediary = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='int', num_bins=NBUCKETS**2, )
    hashed_dropoff_intermediary = hash_dropoff_crossing_layer_intermediary(
        (transformed['bucketed_dropoff_longitude'],
         transformed['bucketed_dropoff_latitude']))

    hash_trip_crossing_layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='one_hot', num_bins=NBUCKETS ** 2,
        name="hash_trip_crossing_layer")
    transformed['hashed_trip'] = hash_trip_crossing_layer(
        (hashed_pickup_intermediary,
         hashed_dropoff_intermediary))

    # seconds_since_1970 = tf.cast(
    seconds_since_1970 = new_datemap(inputs['pickup_datetime'])

    # seconds_since_1970 = fn_seconds_since_1970(inputs['pickup_datetime'])
    seconds_since_1970 = tf.cast(seconds_since_1970, tf.float32)
    hours_since_1970 = seconds_since_1970 / 3600.
    hours_since_1970 = tf.floor(hours_since_1970)
    hour_of_day_intermediary = hours_since_1970 % 24
    transformed['hour_of_day'] = hour_of_day_intermediary
    hour_of_day_intermediary = tf.cast(hour_of_day_intermediary, tf.int32)
    days_since_1970 = seconds_since_1970 / (3600 * 24)
    days_since_1970 = tf.floor(days_since_1970)
    # January 1st 1970 was a Thursday
    day_of_week_intermediary = (days_since_1970 + 4) % 7
    transformed['day_of_week'] = day_of_week_intermediary
    day_of_week_intermediary = tf.cast(day_of_week_intermediary, tf.int32)
    hashed_crossing_layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        num_bins=24 * 7, output_mode="one_hot")
    hashed_crossing_layer_intermediary = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        num_bins=24 * 7, output_mode="int", name='hashed_hour_of_day_of_week_layer')
    transformed['hour_of_day_of_week'] = hashed_crossing_layer(
        (hour_of_day_intermediary, day_of_week_intermediary))
    hour_of_day_of_week_intermediary = hashed_crossing_layer_intermediary(
        (hour_of_day_intermediary, day_of_week_intermediary))

    hash_trip_crossing_layer_intermediary = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='int', num_bins=NBUCKETS ** 2)
    hashed_trip_intermediary = hash_trip_crossing_layer_intermediary(
        (hashed_pickup_intermediary, hashed_dropoff_intermediary))

    hash_trip_and_time_layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
        output_mode='one_hot', num_bins=(
            NBUCKETS ** 2) * 4, name='hash_trip_and_time_layer')
    transformed['hashed_trip_and_time'] = hash_trip_and_time_layer(
        (hashed_trip_intermediary, hour_of_day_of_week_intermediary))
    return transformed


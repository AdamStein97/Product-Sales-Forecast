import tensorflow as tf
import numpy as np

class Preprocessor():
    @staticmethod
    @tf.function
    def normalise_series(series):
      return (series - tf.math.reduce_mean(series)) / tf.math.reduce_std(series)

    @staticmethod
    @tf.function
    def _make_window_dataset(ds, window_in=400, window_out=30, window_shift=200):
        window_size = window_in + window_out
        windows = ds.window(window_size, shift=window_shift)

        def sub_to_batch(sub):
            return sub.batch(window_size, drop_remainder=True)

        def split_window(window):
            return window[:window_in], window[window_in:]

        windows = windows.flat_map(sub_to_batch)
        windows = windows.map(split_window)
        return windows

    @tf.function
    def preprocess_series(self, series, **kwargs):
        norm_series = self.normalise_series(series)
        ds = tf.data.Dataset.from_tensor_slices(norm_series)
        ds = self._make_window_dataset(ds, **kwargs)
        return ds

    def preprocess_predict_series(self, series):
        return np.expand_dims(np.expand_dims(series, axis=-1), axis=0)

    def form_datasets(self, df, num_test_series=200, shuffle_buffer_size=2048, batch_size=128, **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices(df.values.astype(float)).shuffle(shuffle_buffer_size)

        test_dataset = dataset.take(num_test_series)
        train_dataset = dataset.skip(num_test_series)
        train_dataset = (train_dataset.interleave(self.preprocess_series,
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
                         .cache()
                         .shuffle(shuffle_buffer_size)
                         .batch(batch_size, drop_remainder=True)
                         .prefetch(tf.data.experimental.AUTOTUNE))

        test_dataset = (test_dataset.interleave(self.preprocess_series,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        .shuffle(shuffle_buffer_size)
                        .batch(batch_size, drop_remainder=True)
                        .prefetch(tf.data.experimental.AUTOTUNE))

        return train_dataset, test_dataset
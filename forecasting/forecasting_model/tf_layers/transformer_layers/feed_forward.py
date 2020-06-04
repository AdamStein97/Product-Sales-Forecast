import tensorflow as tf

class FeedFoward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
      super(FeedFoward, self).__init__()
      self.layer_feed = [tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)]

    def call(self, x):
      for layer in self.layer_feed:
        x = layer(x)
      return x
import tensorflow as tf
class DenseDecoder(tf.keras.Model):
    def __init__(self, window_out=1, layer_sizes=None, activation=tf.nn.relu, **kwargs):
        super(DenseDecoder, self).__init__()

        if layer_sizes is None:
            layer_sizes = [256, 128]

        self.dense_layers = [tf.keras.layers.Dense(dim, activation=activation) for dim in layer_sizes]
        self.final_reshape_layer = tf.keras.layers.Dense(window_out)

    def call(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return self.final_reshape_layer(x)
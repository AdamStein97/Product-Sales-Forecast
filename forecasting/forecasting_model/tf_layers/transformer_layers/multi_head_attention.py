import tensorflow as tf
from forecasting.utils import scaled_dot_product_attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model

        assert dim_model % self.num_heads == 0

        self.depth = dim_model // self.num_heads

        self.query_layer = tf.keras.layers.Dense(dim_model)
        self.key_layer = tf.keras.layers.Dense(dim_model)
        self.value_layer = tf.keras.layers.Dense(dim_model)

        self.dense = tf.keras.layers.Dense(dim_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, value, key, query, mask=None):
        batch_size = tf.shape(query)[0]

        query = self.query_layer(query)  # (batch_size, seq_len, d_model)
        key = self.key_layer(key)  # (batch_size, seq_len, d_model)
        value = self.value_layer(value)  # (batch_size, seq_len, d_model)

        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dim_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output

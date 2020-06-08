import tensorflow as tf
from forecasting.forecasting_model.tf_layers.transformer_layers.stacked_encoder import Encoder
from forecasting.forecasting_model.tf_layers.transformer_layers.stacked_decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self, batch_size, num_transformer_layers=1, d_model=128, num_heads=8, dff=256,
                 max_pe_input=84, window_out=30, dropout_rate=0.1, **kwargs):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_transformer_layers, d_model, num_heads, dff,
                               max_pe_input, dropout_rate)

        self.decoder = Decoder(num_transformer_layers, d_model, num_heads, dff,
                               window_out, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(1)
        self.batch_size = batch_size
        self.max_pe_target = window_out

    def call(self, inp, tar=None, training=True, look_ahead_mask=None, **kwargs):
        # inp = tf.expand_dims(inp, axis=-1)

        if tar is None:
            tar = tf.zeros((self.batch_size, self.max_pe_target))

        tar = tf.expand_dims(tar, axis=-1)

        enc_output = self.encoder(inp, training, mask=None)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, padding_mask=None)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
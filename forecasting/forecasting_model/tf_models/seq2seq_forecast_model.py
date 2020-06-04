import tensorflow as tf
from forecasting.forecasting_model.tf_layers.conv_encoder import Conv1DEncoder
from forecasting.forecasting_model.tf_layers.lstm_decdoder import DecoderLSTM

class Seq2SeqForecastModel(tf.keras.Model):
    def __init__(self, lstm_dim=128, **kwargs):
        super(Seq2SeqForecastModel, self).__init__()
        self.conv_encoder = Conv1DEncoder(**kwargs)

        self.seq_model = tf.keras.layers.LSTM(lstm_dim)

        self.decoder = DecoderLSTM(**kwargs)

    def call(self, x, **kwargs):
        x = self.conv_encoder(x)
        x = self.seq_model(x)
        prediction = self.decoder(x, **kwargs)
        return prediction

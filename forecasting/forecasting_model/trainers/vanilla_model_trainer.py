import tensorflow as tf
import os
import forecasting as f
from forecasting.forecasting_model.tf_models.vanilla_forecast_model import VanillaForecastModel

class VanillaTrainer():
    def __init__(self, loss_func=tf.keras.losses.mse, optimizer=tf.keras.optimizers.Adam, lr=1e-3, model_name='dense_decoder_forecasting', **kwargs):
        self.model = VanillaForecastModel(**kwargs)
        optimizer = optimizer(lr)

        self.model.compile(optimizer=optimizer, loss=loss_func)
        checkpoint_path = os.path.join(f.MODEL_DIR, "{}.ckpt".format(model_name))

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

    def train_loop(self, train_dataset, test_dataset, epochs=90, **kwargs):
        self.model.fit(train_dataset, validation_data=test_dataset,
                  epochs=epochs, callbacks=[self.cp_callback])
        return self.model

    def predict(self, trained_model, model_input):
        return trained_model(model_input)


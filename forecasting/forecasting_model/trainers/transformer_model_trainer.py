import tensorflow as tf
import time
from forecasting.forecasting_model.tf_models.transformer_forecast_model import ForecastTransformer
from forecasting.utils import create_look_ahead_mask
import forecasting as f

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=1000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class TransformerModelTrainer():
    def __init__(self, window_out=30, loss_func=tf.keras.losses.mse, optimizer=None, d_model=128, **kwargs):
        if optimizer is None:
            optimizer = CustomSchedule(d_model)

        self.model = ForecastTransformer(d_model=d_model, **kwargs)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.window_out = window_out

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        self.val_loss_no_correction = tf.keras.metrics.Mean(name='val_loss_no_correction')

    @tf.function
    def train_step(self, x, y):
        look_ahead_mask = create_look_ahead_mask(self.window_out)

        with tf.GradientTape() as tape:
            predictions, _ = self.model(x,
                                         training=True,
                                         look_ahead_mask=look_ahead_mask)
            loss = self.loss_func(y, tf.squeeze(predictions))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    @tf.function
    def eval_step(self, x, y):
        look_ahead_mask = create_look_ahead_mask(self.window_out)

        predictions, _ = self.model(x,
                                    training=True,
                                    look_ahead_mask=look_ahead_mask)
        loss = self.loss_func(y, tf.squeeze(predictions))

        self.val_loss(loss)

    def train_loop(self, train_dataset, test_dataset, epochs=90, **kwargs):
        ckpt = tf.train.Checkpoint(transformer=self.model,
                                   optimizer=self.optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, f.MODEL_DIR, max_to_keep=5)

        for epoch in range(epochs):
            start = time.time()
            for batch, (x, y) in enumerate(train_dataset):
                self.train_step(x, y)

                if batch % 25 == 0:
                    print('Batch {} Loss {:.4f}'.format(
                        batch, self.train_loss.result()))

            print('Epoch {} Loss {:.4f}'.format(
                epoch + 1, self.train_loss.result()))

            for (batch, (x, y)) in enumerate(test_dataset.take(20)):
                self.eval_step(x, y)

            print('Epoch {} Val Loss {:.4f}'.format(
                epoch + 1, self.val_loss.result()))

            # for (batch, (x, y)) in enumerate(test_dataset.take(20)):
            #   eval_step(x, y)

            # print ('Epoch {} Val Loss {:.4f} Val Loss No Correction {:.4f} '.format(
            #       epoch + 1, val_loss.result(), val_loss_no_correction.result()))

            if (epoch + 1) % 3 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            self.train_loss.reset_states()
            self.val_loss.reset_states()
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        return self.model

    def predict(self, trained_model, model_input):
        return trained_model(model_input)



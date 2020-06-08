import forecasting as f
from forecasting.preprocessor import Preprocessor
from forecasting.forecasting_model.trainers.seq2seq_model_trainer import Seq2SeqTrainer
from forecasting.forecasting_model.trainers.vanilla_model_trainer import VanillaTrainer
from forecasting.forecasting_model.trainers.transformer_model_trainer import TransformerModelTrainer

def predict(model_type, trained_model, input_data):
    preprocessor = Preprocessor()
    model_input = preprocessor.preprocess_predict_series(input_data)

    trainer_obj_dict = {
        f.VANILLA_MODEL_NAME: VanillaTrainer,
        f.SEQ2SEQ_MODEL_NAME: Seq2SeqTrainer,
        f.TRANSFORMER_MODEL_NAME: TransformerModelTrainer
    }
    trainer = trainer_obj_dict[model_type]()
    out = trainer.predict(trained_model, model_input).numpy()[0]
    return out
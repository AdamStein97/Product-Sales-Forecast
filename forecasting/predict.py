import forecasting as f
from forecasting.preprocessor import Preprocessor
from forecasting.forecasting_model.trainers.seq2seq_model_trainer import Seq2SeqTrainer
from forecasting.forecasting_model.trainers.vanilla_model_trainer import VanillaTrainer

def predict(model_type, trained_model, model_input):
    preprocessor = Preprocessor()
    model_input_processed = preprocessor.preprocess_predict_series(model_input)

    trainer_obj_dict = {
        f.VANILLA_MODEL_NAME: VanillaTrainer,
        f.SEQ2SEQ_MODEL_NAME: Seq2SeqTrainer
    }
    trainer = trainer_obj_dict[model_type]()
    out = trainer.predict(trained_model, model_input_processed).numpy()[0]
    return out
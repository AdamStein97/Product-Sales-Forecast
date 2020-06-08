# Product Sales Forecasting

Three different machine learning architectures designed to forecast the daily sales of 
individual products a month into the future.

### Architectures:
- __Vanilla__: Convolution, max pooling and LSTM encoder, decoded by a MLP 
- __Seq2Seq__: Convolution, max pooling and LSTM encoder, decoded by a LSTM
- __Transformer__: Convolution, max pooling followed by Transformer encoder and decoder

## 1. Data

Contains 1941 days of unit sales in Wallmart stores for over 3000 unique products. A handful
of products have been removed from `wallmart_item_sales.csv` into `sample_test_data.csv` to
illustrate predictions of trained models. During training, 200 random products are placed into a validation dataset.

The preprocessor creates multiple training examples per product by sliding a window over the history. The default
window size is 400 days of input to produce the 30 day forecast. This start of the window is shifted 200 days into the future
after each training example is generated.

## 2. Implementation Details

### 2.1 Vanilla Model

Multiple layers of convolution and max pooling are used to increase the dimensionality of the input 
by encoding some spatial features as well as reducing the length of the sequence. The shortened sequence
is then fed through a LSTM to produce an encoding of the history. A Multi-Layered Perceptron (MLP) is then used to decode
the sequence and produce the 30 day forecast.

### 2.2 Seq2Seq Model

Multiple layers of convolution and max pooling are used to increase the dimensionality of the input 
by encoding some spatial features as well as reducing the length of the sequence. The shortened sequence
is then fed through a LSTM to produce an encoding of the history. A decoder LSTM a prediction 
one time step at a time by taking in the previous prediction (embedded to a high dimension using a dense layer) as well
as the state of the LSTM from the previous step. During training, teacher forcing is used to improve optimisation;
instead of stepping using the previous prediction, the ground truth at the previous time step is used as input.

### 2.3 Transformer Model

## 3. Install

Create Environment: `conda create --name forecasting python=3.6`

Activate Environment: `conda activate forecasting`

Install Requirements: `pip install -r requirements.txt`

## 4. Run

Train:
- `python -m examples.example_train_vanilla`
- `python -m examples.example_train_seq2seq`
- `python -m examples.example_train_transformer`

Predict:
- `python -m examples.example_predict_vanilla`
- `python -m examples.example_predict_seq2seq`
- `python -m examples.example_predict_transformer`

## 5. Produced Forecasts

A number of example generated forecasts for each architectures can be found in the generated_forecasts directory.

## 6. Project Structure

__config__: Yaml files with configurations for models and input data
__data__: Csv files of product sales data
__examples__: Examples of how to effectively use the library
__forecasting__: Library
__generated_forecasts__: Saved visualisations of generated forecasts by the different architectures
__logs__: Saved training logs
__saved_models__: Saved model weights

## 7. Library Structure

- __forecasting_model__: Implementation of tensorflow models and their trainers
    - __tf_layers__: Reusable Tensorflow layers which are all instances of tf.keras.layers.Layer
        - __transformer_layers__: Layers specifically used for the Transformer architecture
    - __tf_model__: Instances of tf.keras.Model, the architectures used to predict the sales of products 
    - __trainers__: Wrappers around the models to train them and save the resulting weights and logs
- __predict__:  Function to forecast an individual time series using a trained model
- __preprocessor__:  Preprocesses training and test data
- __train__: Function to train models
- __utils__:  Extra utility functions

## 8. References

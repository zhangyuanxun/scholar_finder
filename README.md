# ScholarFinder: Knowledge Embedding based Recommendations using a Deep Generative Model
This implements the algorithms that describe in the paper "ScholarFinder: Knowledge Embedding based Recommendations using a Deep Generative Model"

## Prerequisites
- Python: 2.7
- Numpy: 1.16.5
- TensorFlow: 1.9.0
- XGBoost: 0.82
- Keras: 2.2.5

## Structure of the code
```
collector/
    *.py
dataset/
    processed/
    raw/
model/
    input_fn.py
    model_fn.py
output/
eval/
constants.py
train.py
README.md
```
- collector/ : contains all scripts to collector data from raw dataset (NSF Grant dataset) or from other websites (Google Scholar)
- dataset/ : contains all the data include raw and processed of the project
- model/ : defines the deep learning model pipeline 
    - model/input_fn.py : defines the input data pipeline (which includes data pre-processing before model training.)
    - model/model_fn.py : defines the deep learning model
- output/ : contains all output files including pre-trained model files and experimental results
- eval/ : contains utility functions to evaluate a model and compare with other models
- constants.py : defines constants variables for the project
- utils.py : 
## Citations
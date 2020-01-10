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
data_preprocessor.py
train.py
tsne.py
utils.py
README.md
```
- dataset/ : contains all the data include raw and processed of the project
- model/ : defines the deep learning model pipeline 
    - model/input_fn.py : defines the input data pipeline (which includes data pre-processing before model training.)
    - model/model_fn.py : defines the deep learning model
- output/ : contains all output files including pre-trained model files and experimental results
- eval/ : contains utility functions to evaluate the model and to generate experimental results
- constants.py : defines constants variables for the project
- data_collector: utility to collect data from raw dataset (NSF Grant dataset) or from other websites (Google Scholar)
- data_preprocessor.py : utility functions to preprocess dataset
- train.py : utility functions to train the model
- tsne.py : implements the t-sne algorithm
- utils.py : implements the helper functions for the project

## Getting Started
### Data Collecting
We collect two types of dataset: a) scholars' publications abstract which collects from Google Scholar; b NSF grant abstract from NSF grant dataset). 
(Note, we have already provided processed dataset. it is not necessary to run the data collecting, unless you want to collect new dataset)

- Collect scholars' publications abstract which collects from Google Scholar (This commands needs a few hours to collect)
```
python data_collector.py --type pub
```
- Collect NSF grant abstract from NSF grant dataset
```
python data_collector.py --type pub
```

### Data Processing

### 

## Citations
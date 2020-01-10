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
In the data collecting stage, we collect two types of the dataset: a) scholars' publications abstract which collects from Google Scholar; b NSF grant abstract from NSF grant dataset).  
Finally, the collected dataset will be saved into the folder(/dataset/processed/). (Note, we have already collected the relevant dataset. it is not necessary to run the data collecting script unless you want to collect new dataset). 
- Collect scholars' publications abstract which collects from Google Scholar (This command needs a few hours to collect.)
```
python data_collector.py --type pub
```
- Collect NSF grant abstract from NSF grant dataset
```
python data_collector.py --type pub
```

### Data Pre-Processing
After the data collecting stage, we get scholar's publications' abstract and NSF grant abstract with text format. In the data pre-processing stage, we transform their format from text to Bag-of-Words,
which allows us to train our deep learning model. Finally, the pre-processed dataset will be saved into the folder(/dataset/processed/). 
(Note, we have provided a pre-processed dataset. it is not necessary to run the data pre-processing script unless you want to collect a new dataset. 
If you try to run this script, please make sure you backup the original pre-processed datasets. Otherwise, it will override the original datasets). 

- transform the dataset format from text to Bag-of-Words,
```
python data_preprocessor.py
```

### Train our deep learning model
The goal of our deep learning model is to capture the latent representation of scholars' knowledge and NSF grant abstracts. Because, the dataset used for training is Bag-of-Words format, 
which is high dimensional datasets (more than 10,000 dimension) . It's hard to between two scholars or a scholar and a grant. Hence, we apply [Variational Autoencoder](https://arxiv.org/abs/1312.6114) to capture latent
representation or lower dimensional representation of knowledge (usually from 2~200 dimension that defined by user)



## Citations
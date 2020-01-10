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

### Train the knowledge embedding model
The goal of our knowledge embedding model is to capture the latent representation of scholars' knowledge and NSF grant abstracts. Because, the dataset used for training is Bag-of-Words format, 
which is high dimensional datasets (more than 10,000 dimension) . It's hard to between two scholars or a scholar and a grant. Hence, we apply [Variational Autoencoder (VAE)](https://arxiv.org/abs/1312.6114) to capture latent
representation or lower dimensional representation of knowledge (usually from 2~200 dimension that defined by user, and the 2 or 3 dimension representation is good for visualization). 

We need to two steps to train the knowledge representation model: 
- Train VAE model to capture latent representation of scholar's knowledge by specifying the parameter --type scholar. You also need
 to specify the model_folder to save model file. You may adjust other parameters such as batch_size, num_epoch.
```
python train.py \
  --type scholar \
  --batch_size 128 \
  --num_epoch 100 \
  --latent_dim 50 \
  --test_size 0.0 \
  --load_model no \
  --model_folder model_scholar_embedding
```
- Train VAE model to capture latent representation of NSF's knowledge by specifying the parameter --type nsf. You also need
 to specify the model_folder to save model file and make sure it's different scholar embedding folder. 
 You may adjust other parameters such as batch_size, num_epoch.
```
python train.py \
  --type nsf \
  --batch_size 128 \
  --num_epoch 100 \
  --latent_dim 50 \
  --test_size 0.0 \
  --load_model no \
  --model_folder model_nsf_embedding
```
After finishing the training embeddings, you will get model files (*.h5) under the folder /output.


## Citations
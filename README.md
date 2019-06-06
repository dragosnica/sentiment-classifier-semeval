# semeval-tweets-sentiment-analysis

## Overview

Classifier training pipeline for sentiment analysis. The models available for training are: fully-connected neural network (FCNN), convolutional neural network (CNN), long short term memory neural network (LSTM) and a simple stacking ensemble (all built using Keras with Tensorflow as backend).  

The pipeline was originally built to accommodate the 2017 semeval tweets data. The code is entirely written in Python 3.5.

Author: Dragoş A Nica (@DragosANica and nica.dragos_alexandru@yahoo.com)

Coursework #2 for the Natural Language Processing (CS918) module for the MSc Computer Science at The University of Warwick.

## How to use the pipeline

1. The data has to be in the .txt file format in the "data/original" folder. The tweets inside the data file must have the format: ID(space)Class(space)Tweet;
The preprocessed data in .txt format is stored in "data/processed/txt". The preprocessing step also generates a .csv version of the data in "data/processed/csv"

2. The word embeddings files must be present in "data/word_embeddings". From the tests run, GloVe word embeddings are the best choice for Twitter data. They can be downloaded from: https://nlp.stanford.edu/projects/glove/
![alt text](https://i.imgur.com/BgssdYn.png)
Alternatively, other word embeddings can be used in both binary and text formats. 

3. Ensure all dependencies are installed as in requirements.txt:
```bash
pip3 install -r requirements.txt
```

Additionally, ensure that the twokenize.py module (download from https://github.com/myleott/ark-twokenize-py)  is in "data_preprocessing" or in another Python3 packages path. (Credit to https://github.com/myleott/ - the author of twokenize.py)

4. To start the data preprocessing step, open preprocess_data.sh from the "data_preprocessing" folder and change the names of the data files. Then open a bash terminal and run:

  ```bash
  (cd data_preprocessing && ./preprocess_data.sh)
  ```

5. Modify the names of the testsets in tesetsets.py. At the moment the program requires both the .csv and the .txt versions of the files.
 
6. From the model_building folder, open "build.sh" and uncomment one of the commands you'd like to run. Again, modify the file sources required. The available flags and features in "NN_build.py" are as follows:

```python3
-model (-m): "DNN" for fully connected network; "CNN" for convolutional neural network; "LSTM" for long short term memory network; "CNN_LSTM_Ensemble" for stacking ensemble. The network architectures are defined in the .py files in model_building
-phase (-p): "train"/"load_weights". Make sure the architecture is defined as in the weights file before using "load_weights". 
-model_weights_file (-mwf): the path to the model weights file to be loaded (only required if -p is "load_weights"
-model_weights_file_train_1 (-mwftr1): the path to the convolutional neural network weights file for the ensemble model (required both for training the ensemble model and just loading the weights)
-model_weights_file_train_2 (-mwftr2): same as model_weights_file_train_1, except it's for the LSTM network
-train_data_file (-trd): the path to the training data file (.csv format only)
-dev_data_file (-devd): the path to the development data file (.csv format only)
-word_embeddings_file (-we): the path to the word embeddings file
-train_we (-trwe or -no_trwe): Pass the flag -trwe if the word embeddings are to be trainable by the neural network. Pass the flag -no_trwe if the word embeddings should remain as they are during the training phase
-max_len (-ml): the maximum length for the sentence (required just for the CNN and LSTM models)
-
```

Open classification.py:

  ```bash
  gedit classification.py
  ```
If you choose to train the classifier using neural networks, ensure that the numeric_target variable equals "True" (bool). Else, set it to "False" (bool).

7. Train your classifier:
  ```bash
  python3 classification.py
  ```
  The best neural network models are saved in the "models" folder.
 

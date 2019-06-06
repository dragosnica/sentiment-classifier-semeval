#!/usr/bin/env bash

### DNN
# ./NN_build.py -m "DNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -pt
# ./NN_build.py -m "DNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.100d.txt" -pt
# ./NN_build.py -m "DNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -pt
# ./NN_build.py -m "DNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/GoogleNews-vectors-negative300.bin" -pt
# ./NN_build.py -m "DNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/new_word2vec_50d.model" -pt
# ./NN_build.py -m "DNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/new_word2vec_100d.model" -pt
# ./NN_build.py -m "DNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/new_word2vec_200d.model" -pt

# ./NN_build.py -m "DNN" -p "load_weights" -mwf "../models_weights/Averaged_GloVe_50d_DNN_model.04-0.6195.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -pt
# ./NN_build.py -m "DNN" -p "load_weights" -mwf "../models_weights/Averaged_GloVe_100d_DNN_model.04-0.6235.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.100d.txt" -pt
# ./NN_build.py -m "DNN" -p "load_weights" -mwf "../models_weights/Averaged_GloVe_200d_DNN_model.02-0.6240.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -pt
# ./NN_build.py -m "DNN" -p "load_weights" -mwf "../models_weights/Averaged_GNword2vec_300d_DNN_model.06-0.6595.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/GoogleNews-vectors-negative300.bin" -pt
# ./NN_build.py -m "DNN" -p "load_weights" -mwf "../models_weights/TFIDF_Bigrams_DNN_model.01-0.6600.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/GoogleNews-vectors-negative300.bin" -pt

# ./NN_build.py -m "DNN" -p "load_model" -mpf ".."

#### CNN
# ./NN_build.py -m "CNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -no_trwe -ml 50 -pt
# ./NN_build.py -m "CNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.100d.txt" -trwe -ml 50 -pt
# ./NN_build.py -m "CNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -trwe -ml 50 -pt
# ./NN_build.py -m "CNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/GoogleNews-vectors-negative300.bin" -trwe -ml 50 -pt
# ./NN_build.py -m "CNN" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/new_word2vec_200d.model" -trwe -ml 50 -pt

# ./NN_build.py -m "CNN" -p "load_weights" -mwf "../models_weights/GloVe_50d_NotTrainableWE_Conv_NN_padding50d_model.01-0.6530.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -ml 50 -pt
# ./NN_build.py -m "CNN" -p "load_weights" -mwf "../models_weights/GloVe_50d_TrainableWE_Conv_NN_padding50d_model.02-0.6730.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -ml 50 -pt
# ./NN_build.py -m "CNN" -p "load_weights" -mwf "../models_weights/GloVe_100d_NotTrainableWE_Conv_NN_padding50d_model.03-0.6630.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.100d.txt" -ml 50 -pt
# ./NN_build.py -m "CNN" -p "load_weights" -mwf "../models_weights/GloVe_100d_TrainableWE_Conv_NN_padding50d_model.03-0.6635.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.100d.txt" -ml 50 -pt
# ./NN_build.py -m "CNN" -p "load_weights" -mwf "../models_weights/GloVe_200d_NotTrainableWE_Conv_NN_padding50d_model.01-0.6675.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -ml 50 -pt
# ./NN_build.py -m "CNN" -p "load_weights" -mwf "../models_weights/GloVe_200d_TrainableWE_Conv_NN_padding50d_model.01-0.6830.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -ml 50 -pt
# ./NN_build.py -m "CNN" -p "load_weights" -mwf "../models_weights/GNword2vec_300d_NotTrainableWE_Conv_NN_padding50d_model.09-0.6705.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/GoogleNews-vectors-negative300.bin" -ml 50 -pt

# ./NN_build.py -m "CNN" -p "load_model" -mpf ".."

#### LSTM
# ./NN_build.py -m "LSTM" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -no_trwe -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.100d.txt" -trwe -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -trwe -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/GoogleNews-vectors-negative300.bin" -trwe -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "train" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/new_word2vec_200d.model" -no_trwe -ml 50 -pt

# ./NN_build.py -m "LSTM" -p "load_weights" -mwf "../models_weights/GloVe_50d_NotTrainableWE_LSTM_NN_padding50d_model.32-0.6555.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "load_weights" -mwf "../models_weights/GloVe_50d_TrainableWE_LSTM_NN_padding50d_model.21-0.6850.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "load_weights" -mwf "../models_weights/GloVe_100d_NotTrainableWE_LSTM_NN_padding50d_model.23-0.6690.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.100d.txt" -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "load_weights" -mwf "../models_weights/GloVe_100d_TrainableWE_LSTM_NN_padding50d_model.02-0.6855.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.100d.txt" -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "load_weights" -mwf "../models_weights/GloVe_200d_NotTrainableWE_LSTM_NN_padding50d_model.18-0.6750.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "load_weights" -mwf "../models_weights/GloVe_200d_TrainableWE_LSTM_NN_padding50d_model.04-0.6860.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "load_weights" -mwf "../models_weights/GNword2vec_300d_NotTrainableWE_LSTM_NN_padding50d_model.26-0.6745.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/GoogleNews-vectors-negative300.bin" -ml 50 -pt
# ./NN_build.py -m "LSTM" -p "load_weights" -mwf "../models_weights/GNword2vec_300d_TrainableWE_LSTM_NN_padding50d_model.08-0.6790.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/GoogleNews-vectors-negative300.bin" -ml 50 -pt

# ./NN_build.py -m "LSTM" -p "load_model" -mpf ".."


#### Ensemble
# ./NN_build.py -m "CNN_LSTM_Ensemble" -p "train" -mwftr1 "../models_weights/GloVe_50d_NotTrainableWE_Conv_NN_padding50d_model.20-0.6625.hdf5" -mwftr2 "../models_weights/GloVe_50d_NotTrainableWE_LSTM_NN_padding50d_model.32-0.6555.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.50d.txt" -no_trwe -ml 50 -pt
# ./NN_build.py -m "CNN_LSTM_Ensemble" -p "train" -mwftr1 "../models_weights/GloVe_200d_TrainableWE_Conv_NN_padding50d_model.02-0.6675.hdf5" -mwftr2 "../models_weights/GloVe_200d_TrainableWE_LSTM_NN_padding50d_model.10-0.6815.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -devd "../data/processed/csv/proc-twitter-dev-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -trwe -ml 50 -pt

./NN_build.py -m "CNN_LSTM_Ensemble" -p "load_weights" -mwf "../models_weights/GloVe_200d_CNN_LSTM_Ensemble_padding50d_model.05-0.6905.hdf5" -mwftr1 "../models_weights/GloVe_200d_TrainableWE_Conv_NN_padding50d_model.01-0.6830.hdf5" -mwftr2 "../models_weights/GloVe_200d_TrainableWE_LSTM_NN_padding50d_model.10-0.6815.hdf5" -trd "../data/processed/csv/proc-twitter-training-data.csv" -we "../data/word_embeddings/glove.twitter.27B.200d.txt" -ml 50 -pt


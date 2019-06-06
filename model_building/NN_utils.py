#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# import twokenize
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors

def load_word_embeddings(word_embeddings_file):
	'''Loads the weights from a word embeddings file'''

	if str(word_embeddings_file).endswith(".bin"):
		word_to_vec_mapping = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=True)
	else:
		word_to_vec_mapping = KeyedVectors.load_word2vec_format(word_embeddings_file, binary=False)
		''' NOTE: GloVe vectors can also be loaded using this method. However, the GloVe files  
		must be edited to contain the vocabulary size and the vector size on 
		the first line. This is a requirement of the load_word2vec_format() method

		Exemple: The first line in glove.twitter.27B.50d.txt would have to be: 
		1193514 50 where 1193514 is (the number of rows in the file - 1) and 50 
		is the dimension of each word vector representation '''

	return (word_to_vec_mapping)


def csv_to_np(filename):

	data = pd.read_csv(filename)
	ID = data.iloc[:, 0]
	target_class = data.iloc[:, 1]
	examples = data.iloc[:, 2]

	ID = ID.values.astype(int)
	X = examples.values.astype(str)
	Y = target_class.values.astype(int)

	return(ID, X, Y)


def average_word_embeddings(X, word_to_vec_mapping):
	'''Creates document vectors of dimension equal to the embeddings 
	in word_to_vec_mapping. The vectors are obtained by averaging the 
	embeddings for the words in the document'''

	m = X.shape[0] #number of documents (tweets) in the provided data
	embedding_dim = word_to_vec_mapping.vector_size #dimension of the word embeddings
	extracted_features = np.zeros((m, embedding_dim))

	for i in range(m):
		num_words_found = 0
		for word in X[i].split():
			if word in word_to_vec_mapping:
				extracted_features[i] += word_to_vec_mapping[word]#.reshape((1, embedding_dim))
				num_words_found += 1
		if num_words_found > 0:
			extracted_features[i] = extracted_features[i] / num_words_found

	return (extracted_features)

def integer_to_one_hot(Y):

	num_clases = np.amax(Y) + 1
	Y = np.eye(num_clases)[Y.reshape(-1)]

	return (Y)


def example_to_indices(X, word_to_vec_mapping, max_len):
	'''Converts string training examples into arrays of indices 
	corresponding to the indices of the words in the word embeddings''' 

	m = X.shape[0]
	X_indices = np.zeros((m, max_len))
	
	for i in range(m):
		# example_words = twokenize.tokenizeRawTweetText(X[i])
		example_words = X[i].lower().split()
		j = 0
		for word in example_words:
			if word in word_to_vec_mapping:
				X_indices[i, j] = word_to_vec_mapping.vocab[word].index
			else:
				X_indices[i, j] = 0
			j += 1

	return (X_indices)


def build_embedding_layer(word_to_vec_mapping, trainable=False):
	'''Generates a Keras Embedding() layer loaded with the weights imported
	from a word embeddings model'''

	num_words = len(word_to_vec_mapping.vocab) + 1
	embedding_dim = word_to_vec_mapping.vector_size
	embedding_matrix = np.zeros((num_words, embedding_dim))

	for word in word_to_vec_mapping.vocab:
		idx = word_to_vec_mapping.vocab[word].index
		embedding_matrix[idx] = word_to_vec_mapping[word]

	embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim, trainable=trainable)
	embedding_layer.build((None,))
	embedding_layer.set_weights([embedding_matrix])

	return (embedding_layer)

def build_vocabulary_indices(X):
	'''Constructs the vocabulary based on the training data and indexes 
	the words in the vocabulary''' 

	tokenizer = Tokenizer(num_words=100000)
	tokenizer.fit_on_texts(X)

	return(tokenizer)

def example_to_indices_v2(X, tokenizer, max_len):
	'''Converts string training examples into arrays of indices 
	corresponding to the indices of the words in the word embeddings''' 

	X_train_indices = tokenizer.texts_to_sequences(X)
	X_train_indices = pad_sequences(X_train_indices, maxlen=max_len)

	return (X_train_indices)

def build_embedding_layer_v2(tokenizer, word_to_vec_mapping, trainable=False):
	'''Generates a Keras Embedding() layer loaded with the weights imported
	from a word embeddings model'''

	num_words = len(tokenizer.word_index.items()) + 1
	embedding_dim = word_to_vec_mapping.vector_size
	embedding_matrix = np.zeros((num_words, embedding_dim))

	for word, idx in tokenizer.word_index.items():
		if word in word_to_vec_mapping:
			embedding_matrix[idx] = word_to_vec_mapping[word]

	embedding_layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=trainable)
	# embedding_layer.build((None,))

	return (embedding_layer)


def label_to_sentiment(label):
	if label == 0:
		return("negative")
	elif label == 1:
		return("neutral")
	elif label == 2:
		return("positive")


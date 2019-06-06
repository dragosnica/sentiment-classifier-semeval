#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''LSTM (Long Short-Term Memory) neural network architecture for sentiment analysis with word embeddings'''

import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from NN_utils import *
import sys
sys.path.insert(0, "../")
import evaluation

class LSTM_NN:
	def __init__(self, train_data_file, word_to_vec_mapping, max_len, train_we):
		self.word_to_vec_mapping = word_to_vec_mapping
		self.max_len = max_len
		self.train_we = train_we
		_, self.X_train_text, self.Y_train = csv_to_np(train_data_file)
		self.vocabulary = build_vocabulary_indices(self.X_train_text)


		self.embedding_layer = build_embedding_layer_v2(self.vocabulary, self.word_to_vec_mapping, trainable=self.train_we)
		self.sentence_indices = Input(shape=(self.max_len,), dtype='int32')
		self.embeddings = self.embedding_layer(self.sentence_indices)
		X = Bidirectional(LSTM(units=200, dropout=0.5, return_sequences=True))(self.embeddings)
		X = Bidirectional(LSTM(units=200, dropout=0.5))(X)
		X = Dense(units=30, activation='relu')(X)
		X = Dropout(0.5)(X)
		X = Dense(units=3, activation='softmax')(X)
		self.model = Model(self.sentence_indices, X)

		custom_adam = Adam(lr=0.0001)
		self.model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'])
		print(self.model.summary())

	def train(self, dev_data_file=None):
		X_train = example_to_indices_v2(self.X_train_text, self.vocabulary, self.max_len)
		Y_train = integer_to_one_hot(self.Y_train)

		if self.train_we == True:
			save_filename = "GloVe_"+str(self.word_to_vec_mapping.vector_size)+"d_TrainableWE_"+type(self).__name__+"_padding"+str(self.max_len)+"d_model"
		elif self.train_we == False:
			save_filename = "GloVe_"+str(self.word_to_vec_mapping.vector_size)+"d_NotTrainableWE_"+type(self).__name__+"_padding"+str(self.max_len)+"d_model"

		if dev_data_file == None:
			save_filepath = "../models_weights/"+save_filename+".{epoch:02d}-{acc:.4f}.hdf5"
			checkpoint = ModelCheckpoint(save_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
			history = self.model.fit(X_train, Y_train, epochs = 100, batch_size = 32, shuffle=True, callbacks=[checkpoint])

		elif dev_data_file != None:
			_, X_dev_text, Y_dev = csv_to_np(dev_data_file)
			X_dev = example_to_indices_v2(X_dev_text, self.vocabulary, self.max_len)
			Y_dev = integer_to_one_hot(Y_dev)
			save_filepath = "../models_weights/"+save_filename+".{epoch:02d}-{val_acc:.4f}.hdf5"

			checkpoint = ModelCheckpoint(save_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max') 
			callbacks_list = [checkpoint, early_stop]

			history = self.model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), epochs = 100, batch_size = 32, shuffle=True, callbacks=callbacks_list)

		plt.plot(history.history['acc'], label='train_acc')
		plt.plot(history.history['val_acc'], label='dev_acc')
		plt.xlabel("#epochs")
		plt.ylabel("accuracy")
		plt.legend()
		plt.savefig("../training_plots/"+save_filename+".png", bbox_inches='tight')
		plt.show()

	def evaluate(self, test_data_file):
		ID_test, X_test, Y_test = csv_to_np(test_data_file[0])
		X_test = example_to_indices_v2(X_test, self.vocabulary, self.max_len)
		# X_test = example_to_indices(X_test, self.word_to_vec_mapping, self.max_len)
		Y_test = integer_to_one_hot(Y_test)

		predictions = self.model.predict(X_test)
		pred_dict = dict()
		for i in range(len(X_test)):
			pred_dict[str(ID_test[i])] = label_to_sentiment(np.argmax(predictions[i]))

		loss, accuracy = self.model.evaluate(X_test, Y_test)
		print()
		print("Loss = ", loss)
		print("Test accuracy = " + str(accuracy*100) + "%")

		evaluation.evaluate(pred_dict, test_data_file[1], type(self).__name__)
		evaluation.confusion(pred_dict, test_data_file[1], type(self).__name__)

		return (predictions, Y_test)

	def show_errors(self, data_file):
		_, X, Y = csv_to_np(data_file)
		X_indices = example_to_indices_v2(X, self.vocabulary, self.max_len)
		# X_indices = example_to_indices(X, self.word_to_vec_mapping, self.max_len)
		predictions = self.model.predict(X_indices)

		for i in range(len(X)):
		    num = np.argmax(predictions[i])
		    if(num != Y[i]):
		        print('TWEET: ' + X[i])
		        print('PREDICTION: ' + label_to_sentiment(num))  
		        print('REAL: '+ label_to_sentiment(Y[i])+ '\n')

	def prediction_on_new_example(self, example):
		X_test = np.array([example])
		X_test_indices = example_to_indices_v2(X_test, self.vocabulary, self.max_len)
		# X_test_indices = example_to_indices(X_test, self.word_to_vec_mapping, self.max_len)
		print(X_test[0] + ' -> ' + label_to_sentiment(np.argmax(self.model.predict(X_test_indices))))




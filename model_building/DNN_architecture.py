#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Deep neural network architecture for sentiment analysis with word embeddings'''

import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from NN_utils import *
import sys
sys.path.insert(0, "../")
import evaluation

class DNN:
	def __init__(self, word_to_vec_mapping):
		self.word_to_vec_mapping = word_to_vec_mapping
		self.model = Sequential()
		self.model.add(Dense(256, activation='relu', input_dim=self.word_to_vec_mapping.vector_size))
		# self.model.add(Dropout(0.5))
		self.model.add(Dense(128, activation='relu'))
		# self.model.add(Dropout(0.5))
		self.model.add(Dense(3, activation='softmax'))

		# custom_adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		print(self.model.summary())

	def train(self, train_data_file, dev_data_file=None):
		_, X_train, Y_train = csv_to_np(train_data_file)
		X_train = average_word_embeddings(X_train, self.word_to_vec_mapping)
		Y_train = integer_to_one_hot(Y_train)
		save_filename = "Averaged_GloVe_"+str(self.word_to_vec_mapping.vector_size)+"d_"+type(self).__name__+"model"

		if dev_data_file == None:
			history = self.model.fit(X_train, Y_train, epochs = 100, batch_size = 32, shuffle=True)
		
		elif dev_data_file != None:
			_, X_dev, Y_dev = csv_to_np(dev_data_file)
			X_dev = average_word_embeddings(X_dev, self.word_to_vec_mapping)
			Y_dev = integer_to_one_hot(Y_dev)
			save_filepath = "../models_weights/"+save_filename+".{epoch:02d}-{val_acc:.4f}.hdf5"

			checkpoint = ModelCheckpoint(save_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max') 
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
		X_test = average_word_embeddings(X_test, self.word_to_vec_mapping)
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

		return(predictions, Y_test)

	def show_errors(self, data_file):
		_, X, Y = csv_to_np(data_file)
		X_indices = average_word_embeddings(X, self.word_to_vec_mapping)
		predictions = self.model.predict(X_indices)

		for i in range(len(X)):
		    num = np.argmax(predictions[i])
		    if(num != Y[i]):
		        print('TWEET: ' + X[i])
		        print('PREDICTION: ' + label_to_sentiment(num))  
		        print('REAL: '+ label_to_sentiment(Y[i])+ '\n')

	def prediction_on_new_example(self, example):
		X_test = np.array([example])
		X_test_averaged = average_word_embeddings(X_test, self.word_to_vec_mapping)
		print(X_test[0] + ' -> ' + label_to_sentiment(np.argmax(self.model.predict(X_test_averaged))))





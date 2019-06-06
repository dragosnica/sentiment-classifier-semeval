from CNN_architecture import *
from LSTM_NN_architecture import *
import numpy as np
from keras.models import Sequential
from matplotlib import pyplot as plt
from testsets import *
from math import *
import sys
sys.path.insert(0, "../")

class CNN_LSTM_Ensemble():
	def __init__(self, cnn_model_weights_file, lstm_model_weights_file, train_data_file, word_to_vec_mapping, max_len, train_we):
		self.word_to_vec_mapping = word_to_vec_mapping
		self.max_len = max_len
		self.train_we = train_we

		self.cnn_model = Conv_NN(self.word_to_vec_mapping, self.max_len, self.train_we)
		self.cnn_model.model.load_weights(cnn_model_weights_file)

		self.lstm_model = LSTM_NN(train_data_file, self.word_to_vec_mapping, self.max_len, self.train_we)
		self.lstm_model.model.load_weights(lstm_model_weights_file)

		self.model = Sequential()
		self.model.add(Dense(128, activation='relu', input_dim=6))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(3, activation='softmax'))

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		print(self.model.summary())

	def train(self, train_data_file, dev_data_file=None):
		_, X_train, Y_train = csv_to_np(train_data_file)
		cnn_X = example_to_indices(X_train, self.cnn_model.word_to_vec_mapping, self.cnn_model.max_len)
		lstm_X = example_to_indices_v2(X_train, self.lstm_model.vocabulary, self.lstm_model.max_len)
		Y_train = integer_to_one_hot(Y_train)

		cnn_Y_hat = self.cnn_model.model.predict(cnn_X)
		lstm_Y_hat  = self.lstm_model.model.predict(lstm_X)
		X_train = np.dstack((cnn_Y_hat, lstm_Y_hat))
		X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

		save_filename = "GloVe_"+str(self.word_to_vec_mapping.vector_size)+"d_"+type(self).__name__+"_padding"+str(self.max_len)+"d_model"
		
		if dev_data_file == None:
			checkpoint = ModelCheckpoint("../models_weights/"+save_filename+".{epoch:02d}-{acc:.4f}.hdf5", monitor='acc', verbose=1, save_best_only=True, mode='max')	
			history = self.model.fit(X_train, Y_train, epochs = 100, batch_size = 32, shuffle=True, callbacks=[checkpoint])

		elif dev_data_file != None:
			_, X_dev, Y_dev = csv_to_np(dev_data_file)
			cnn_X_dev = example_to_indices(X_dev, self.cnn_model.word_to_vec_mapping, self.cnn_model.max_len)
			lstm_X_dev = example_to_indices_v2(X_dev, self.lstm_model.vocabulary, self.lstm_model.max_len)

			cnn_Y_hat_dev = self.cnn_model.model.predict(cnn_X_dev)
			lstm_Y_hat_dev  = self.lstm_model.model.predict(lstm_X_dev)
			X_dev = np.dstack((cnn_Y_hat_dev, lstm_Y_hat_dev))
			X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]))
			Y_dev = integer_to_one_hot(Y_dev)

			checkpoint = ModelCheckpoint("../models_weights/"+save_filename+".{epoch:02d}-{val_acc:.4f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')	
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
		ID_test, _, _ = csv_to_np(test_data_file[0])
		cnn_X_test, _ = self.cnn_model.evaluate(test_data_file)
		lstm_X_test, Y_test = self.lstm_model.evaluate(test_data_file)

		X_test = np.dstack((cnn_X_test, lstm_X_test))
		X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

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
		cnn_X, _ = self.cnn_model.evaluate(train_data_file)
		lstm_X, Y  = self.lstm_model.evaluate(train_data_file)
		X = np.dstack((cnn_X, lstm_X))
		X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))

		predictions = self.model.predict(X)

		for i in range(len(X)):
		    num = np.argmax(predictions[i])
		    if(num != Y[i]):
		        print('TWEET: ' + X[i])
		        print('PREDICTION: ' + label_to_sentiment(num))  
		        print('REAL: '+ label_to_sentiment(Y[i])+ '\n')

	# def prediction_on_new_example(self, example):
	# 	X_test = np.array([example])
	# 	X_test_indices = example_to_indices(X_test, self.word_to_vec_mapping, self.max_len)
	# 	print(X_test[0] + ' -> ' + label_to_sentiment(np.argmax(self.model.predict(X_test_indices))))









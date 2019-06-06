#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from DNN_architecture import *
from CNN_architecture import *
from LSTM_NN_architecture import *
from Ensemble_architecture import *
from NN_utils import *
from testsets import *
import sys, argparse, pickle
sys.path.insert(0, "../")

def train_DNN(TRAIN_DATA_FILE, DEV_DATA_FILE, word_to_vec_mapping):
	neural_network = DNN(word_to_vec_mapping)
	neural_network.train(train_data_file=TRAIN_DATA_FILE, dev_data_file=DEV_DATA_FILE)
	# neural_network.prediction_on_new_example("I hate this Collin thing, but I like his brother..")
	return (neural_network)


def load_weights_DNN(DNN_MODEL_WEIGHTS_FILE, word_to_vec_mapping):
	neural_network = DNN(word_to_vec_mapping)
	neural_network.model.load_weights(DNN_MODEL_WEIGHTS_FILE)
	# neural_network.prediction_on_new_example("I hate this Collin thing, but I like his brother..")
	return (neural_network)


def train_CNN(TRAIN_DATA_FILE, DEV_DATA_FILE, word_to_vec_mapping, max_len, train_we):
	neural_network = Conv_NN(word_to_vec_mapping, max_len, train_we)
	neural_network.train(train_data_file=TRAIN_DATA_FILE, dev_data_file=DEV_DATA_FILE)
	# neural_network.prediction_on_new_example("I hate this Collin thing, but I like his brother..")
	return (neural_network)

def load_weights_CNN(CNN_MODEL_WEIGHTS_FILE, word_to_vec_mapping, max_len, train_we):
	neural_network = Conv_NN(word_to_vec_mapping, max_len, train_we)
	neural_network.model.load_weights(CNN_MODEL_WEIGHTS_FILE)
	# neural_network.prediction_on_new_example("I hate this Collin thing, but I like his brother..")
	return (neural_network)


def train_LSTM(TRAIN_DATA_FILE, DEV_DATA_FILE, word_to_vec_mapping, max_len, train_we):
	neural_network = LSTM_NN(TRAIN_DATA_FILE, word_to_vec_mapping, max_len, train_we)
	neural_network.train(dev_data_file=DEV_DATA_FILE)
	neural_network.prediction_on_new_example("I hate this Collin thing, but I like his brother..")
	return (neural_network)

def load_weights_LSTM(LSTM_MODEL_WEIGHTS_FILE, TRAIN_DATA_FILE, word_to_vec_mapping, max_len, train_we):
	neural_network = LSTM_NN(TRAIN_DATA_FILE, word_to_vec_mapping, max_len, train_we)
	neural_network.model.load_weights(LSTM_MODEL_WEIGHTS_FILE)
	neural_network.prediction_on_new_example("I hate this Collin thing, but I like his brother..")
	return (neural_network)


def train_CNN_LSTM_ensemble(CNN_MODEL_WEIGHTS_FILE, LSTM_MODEL_WEIGHTS_FILE, TRAIN_DATA_FILE, word_to_vec_mapping, max_len, train_we, DEV_DATA_FILE=None):
	neural_network = CNN_LSTM_Ensemble(CNN_MODEL_WEIGHTS_FILE, LSTM_MODEL_WEIGHTS_FILE, TRAIN_DATA_FILE, word_to_vec_mapping, max_len, train_we)
	neural_network.train(train_data_file=TRAIN_DATA_FILE, dev_data_file=DEV_DATA_FILE)
	# neural_network.prediction_on_new_example("I hate this Collin thing, but I like his brother..")
	return (neural_network)

def load_weights_CNN_LSTM_ensemble(CNN_LSTM_MODEL_WEIGHTS_FILE, CNN_MODEL_WEIGHTS_FILE, LSTM_MODEL_WEIGHTS_FILE, TRAIN_DATA_FILE, word_to_vec_mapping, max_len, train_we):
	neural_network = CNN_LSTM_Ensemble(CNN_MODEL_WEIGHTS_FILE, LSTM_MODEL_WEIGHTS_FILE, TRAIN_DATA_FILE, word_to_vec_mapping, max_len, train_we)
	neural_network.model.load_weights(CNN_LSTM_MODEL_WEIGHTS_FILE)
	# neural_network.prediction_on_new_example("I hate this Collin thing, but I like his brother..")
	return (neural_network)

def test_model(model):
	for testset in testsets:
		Y_hat, Y = model.evaluate(testset)
		# model.show_errors(testset[0])


def main(argv):
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("-m", "--model", type=str, default=sys.stdin)
	parser.add_argument("-p", "--phase", type=str, default=sys.stdin)
	parser.add_argument("-mpf", "--model_pickle_file", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-mwf", "--model_weights_file", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-mwftr1", "--model_weights_file_train_1", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-mwftr2", "--model_weights_file_train_2", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-trd", "--train_data", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-devd", "--dev_data", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-we", "--we_file", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-ml", "--max_len", type=int, default=sys.stdin)
	parser.add_argument("-trwe", "--train_we", action="store_true")
	parser.add_argument("-no_trwe", "--no_train_we", action="store_false")
	parser.add_argument("-pt", "--perform_test", action="store_true")
	parser.add_argument("-no_pt", "--no_perform_test", action="store_false")

	args = vars(parser.parse_args())
	MODEL_TYPE = args["model"]
	PHASE = args["phase"]
	MODEL_PICKLE_FILE = str(args["model_pickle_file"].name)
	MODEL_WEIGHTS_FILE = str(args["model_weights_file"].name)
	MODEL_WEIGHTS_FILE_TRAIN_1 = str(args["model_weights_file_train_1"].name)
	MODEL_WEIGHTS_FILE_TRAIN_2 = str(args["model_weights_file_train_2"].name)
	TRAIN_DATA_FILE = str(args["train_data"].name)
	DEV_DATA_FILE = str(args["dev_data"].name)
	WORD_EMBEDDINGS_FILE = str(args["we_file"].name)
	TRAIN_WE = args["train_we"]
	MAX_LEN = args["max_len"]

	word_to_vec_mapping = load_word_embeddings(WORD_EMBEDDINGS_FILE)

	if PHASE == "train":
		if MODEL_TYPE == "DNN":
			model = train_DNN(TRAIN_DATA_FILE, DEV_DATA_FILE, word_to_vec_mapping)
			# pickle.dump(model, open("../models/Averaged_GloVe_"+str(word_to_vec_mapping.vector_size)+"d_"+str(MODEL_TYPE)+"_model.p", "wb"))
		elif MODEL_TYPE == "CNN":
			model = train_CNN(TRAIN_DATA_FILE, DEV_DATA_FILE, word_to_vec_mapping, MAX_LEN, TRAIN_WE)
		elif MODEL_TYPE == "LSTM":
			model = train_LSTM(TRAIN_DATA_FILE, DEV_DATA_FILE, word_to_vec_mapping, MAX_LEN, TRAIN_WE)
		elif MODEL_TYPE == "CNN_LSTM_Ensemble":
			model = train_CNN_LSTM_ensemble(MODEL_WEIGHTS_FILE_TRAIN_1, MODEL_WEIGHTS_FILE_TRAIN_2, TRAIN_DATA_FILE, word_to_vec_mapping, MAX_LEN, TRAIN_WE, DEV_DATA_FILE)
			# if TRAIN_WE == True:
				# pickle.dump(model, open("../models/GloVe_"+str(word_to_vec_mapping.vector_size)+"d_TrainableWE_"+str(MODEL_TYPE)+"_padding"+str(MAX_LEN)+"d_model.p", "wb"))
			# elif TRAIN_WE == False:
				# pickle.dump(model, open("../models/GloVe_"+str(word_to_vec_mapping.vector_size)+"d_NotTrainableWE_"+str(MODEL_TYPE)+"_padding"+str(MAX_LEN)+"d_model.p", "wb"))

	elif PHASE == "load_weights":
		if MODEL_TYPE == "DNN":
			model = load_weights_DNN(MODEL_WEIGHTS_FILE, word_to_vec_mapping)
		elif MODEL_TYPE == "CNN":
			model = load_weights_CNN(MODEL_WEIGHTS_FILE, word_to_vec_mapping, MAX_LEN, TRAIN_WE)
		elif MODEL_TYPE == "LSTM":
			model = load_weights_LSTM(MODEL_WEIGHTS_FILE, TRAIN_DATA_FILE, word_to_vec_mapping, MAX_LEN, TRAIN_WE)
		elif MODEL_TYPE == "CNN_LSTM_Ensemble":
			model = load_weights_CNN_LSTM_ensemble(MODEL_WEIGHTS_FILE, MODEL_WEIGHTS_FILE_TRAIN_1, MODEL_WEIGHTS_FILE_TRAIN_2, TRAIN_DATA_FILE, word_to_vec_mapping, MAX_LEN, TRAIN_WE)


	# elif PHASE == "load_model":
		# model = pickle.load(open(MODEL_PICKLE_FILE, "rb"))

	if args["perform_test"] == True:
		test_model(model)

if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import twokenize, re, sys, argparse
import numpy as np

'''Twitter data preprocessing module'''
'''The module also creates a .txt file containing the processed tweets'''

#done: For now, keep tweet# and class in final preprocessed data
#done: Handle hashtags (split uncommon, made up ones)

WORDS_COUNT_FILE = '../data/spelling_correction/big.txt'

def read_words(text): 
	return re.findall(r'\w+', text.lower())

def process_tweets(ORIGIN_TRAINING_FILE, PREPROC_TRAINING_FILE):
	processed_tweets = list() #final list of processed tweets (as list of tokens)
	urls = re.compile(r"http\S+") #find urls starting with 'http'
	urls2 = re.compile(r"www\.\w+\S+") #find urls of type www...
	urls3 = re.compile(r"\S+\.com") #find urls ending in .com
	stop_words = stopwords.words('english') #create list of stop words in the English language
	non_alphanum = re.compile(r"[^\w+\-\.\:]") #find all non-alphanumeric characters except '-', '.' and ':'
	usr_mentions = re.compile(r"\@{1}[a-zA-Z0-9]+") #find user mentions
	hashtags = re.compile(r"\#{1}\S+") #hashtags starting with letters
	pos_emojis = ([":)", ";)", ";-)", ":-)", ":))", ":-))", ";))", ";-))", 
	":o", "^^", "^_^", "<3", ":3", "\m/", "\_()_/", "\o/", "\_/"])
	neg_emojis = (["-__-", ":(", ":((", ":-(", ":-(("])
	words_dict = Counter(read_words(open(WORDS_COUNT_FILE).read())) #dictionary of words:appearance count in a text file of about 1 mil words

	for line in ORIGIN_TRAINING_FILE:
		result = list() #store result after processing is done
		line = twokenize.tokenizeRawTweetText(line)
		for index, word in enumerate(line):
			if index == 0:
				result.append(word)
				continue

			if (re.match(urls, word) or re.match(urls2, word) or re.match(urls3, word)):
				result.append(" URL ")
				continue

			if (re.match(usr_mentions, word)):
				result.append(" userMnt ") #replace user mentions with " userMnt "
				continue

			if re.match(hashtags, word):
				hashtag_results = correct_hashtags(word[1:].lower(), words_dict)
				for word in hashtag_results:
					result.append(word)
				continue

			if (word in pos_emojis):
				result.append(" posEmot ")
				continue

			if (word in neg_emojis):
				result.append(" negEmot ")
				continue

			if (word in stop_words) or (re.match(non_alphanum, word)):
				result.append(" ")
				continue

			result.append(word)

		processed_tweets.append(result)
		PREPROC_TRAINING_FILE.write(' '.join(result) + "\n")
	return(processed_tweets)

def correct_hashtags(hashtag, words_dict):
	word_result = list()
	result = list()
	hashtag_length = len(hashtag)
	claimed = np.full(hashtag_length, False, dtype=bool)
	for n_ch in range(hashtag_length, 0, -1):
		for st_point in range(0, hashtag_length-n_ch+1):
			substring = hashtag[st_point:st_point+n_ch]
			if substring in words_dict:
				if ~np.any(claimed[st_point:st_point+n_ch]):
					claimed[st_point:st_point+n_ch] = True
					word_result.append((st_point, substring))
	word_result.sort()
	for _, substring in word_result:
		result.append(substring)
	return (result)
'''Slightly modified hashtag handling function from the work by Andrei Barsan, Bernhard Kratzwald and Nikolaos Kolitsas 
	https://github.com/bernhard2202/twitter-sentiment-analysis'''

def main(argv):
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("-of", "--orig_file", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-pf", "--preproc_file", type=argparse.FileType('w'), default=sys.stdout)
	args = vars(parser.parse_args())
	processed_tweets = process_tweets(args["orig_file"], args["preproc_file"])
	return(processed_tweets)

if __name__ == "__main__":
    main(sys.argv)
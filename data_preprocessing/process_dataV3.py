#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import twokenize, re, sys, argparse, txt_to_csv
sys.path.insert(0, "../model_building")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from NN_utils import *

'''Twitter data preprocessing module'''
'''The module also creates a .txt file containing the processed tweets'''

#TODO: implement more elaborate spelling correction to cover situations where words have higher edit distance than 2 


def process_tweets(ORIGIN_TRAINING_FILE, PREPROC_TRAINING_FILE):
	processed_tweets = list() #final list of processed tweets (as list of tokens)
	urls = re.compile(r"http\S+") #find urls starting with 'http'
	urls2 = re.compile(r"\swww\.\w+\S+") #find urls starting with ' www.'
	email_addresses = re.compile(r"[0-9a-zA-Z]+@[0-9a-zA-Z]+\.co[a-zA-Z]")
	stop_words = stopwords.words('english')
	whitelist = ["n't", "not", "no"]
	num = re.compile(r"\d+") #find numbers
	usr_mentions = re.compile(r"\@{1}[a-zA-Z0-9]+") #find user mentions
	smile = [":)", ":-)", "^^", "^_^"]
	sadface = [":(", ":-(", ":((", ":-(("]
	heart = ["<3", ":3"]
	lol_face = [":))", ":-))", ";))", ";-))"]
	neutral_face = ["-__-"]

	for line in ORIGIN_TRAINING_FILE:
		result = list() #store temporary processed line
		line = twokenize.tokenizeRawTweetText(line)
		for index, word in enumerate(line):
			if index == 0:
				result.append(word) #keep the tweet ID intact
				continue

			if (re.match(urls, word) or re.match(urls2, word) or re.match(email_addresses, word)):
				# result.append(" <url> ")
				result.append(" ")
				continue

			if word in stop_words:
				result.append(" ")
				continue

			if word[0] == "#" and word not in vocabulary:
				temp = re.sub(num, " <number> ", word[1:])
				if temp != word[1:]:
					result.append(temp)
				else:
					temp = split_hashtags(word[1:], vocabulary)
					result.append(temp)
				continue

			if (re.match(usr_mentions, word)):
				# result.append(" <user> ") #replace user mentions with <user>
				result.append(" ")
				continue

			if word in smile:
				result.append(" <smile> ")
				continue

			if word in sadface:
				result.append(" <sadface> ")
				continue

			if word in heart:
				result.append(" <heart> ")
				continue

			if word in lol_face:
				result.append(" <lolface> ")
				continue

			if word in neutral_face:
				result.append(" <neutralface> ")
				continue

			# word = correct_word(word, vocabulary)
			result.append(word)

		processed_tweets.append(result)
		PREPROC_TRAINING_FILE.write(' '.join(result) + "\n")

	return(processed_tweets)

def words(text, model): 
	# return (re.findall(r'\w+', text.lower()) in model)
	return (list(word for word in re.findall(r'\w+', open(text).read()) if word in model))

# we_model = load_word_embeddings("../data/word_embeddings/glove.twitter.27B.25d.txt")
we_model = load_word_embeddings("../data/word_embeddings/GoogleNews-vectors-negative300.bin")
vocabulary = Counter(words('../data/spelling_correction/big.txt', we_model)) #extract the vocabulary for the language model - used in the spelling corrector

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
	return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def known(words_set, vocabulary):
	return set(w for w in words_set if w in vocabulary)

def word_probability(word, vocabulary_length=sum(vocabulary.values())):
	# return dict_freqs.get(word).count
	return (vocabulary[word] / vocabulary_length)

def candidates(word, vocabulary):
	return known([word], vocabulary).union(known(edits1(word), vocabulary)).union(known(edits2(word), vocabulary))

def correct_word(word, vocabulary):
	if len(candidates(word,vocabulary)) > 0:
		return (max(candidates(word, vocabulary), key=word_probability))
	else: 
		return (word)
'''Slightly modified spelling corrector based on edit distance by Peter Norvig
	https://norvig.com/spell-correct.html'''


def split_hashtags(hashtag, vocabulary):
	'''Splits hashtags into words. Returns the token "<hashtag>" in case no words are found in the hashtag'''
	'''Ex: "#2016IsMagical" -> "2016 is magical"'''
	split_hashtag = []
	length = len(hashtag)
	words_found = []
	claimed_letters = np.full(length, False, dtype=bool) #any of the letters can form a word initially
	for n in range(length, 0, -1):  
		for s in range(0, length-n+1):
			substring = hashtag[s:s+n]
			if substring.lower() in vocabulary:
				if ~np.any(claimed_letters[s:s+n]): #selected letters are not yet forming a word
					claimed_letters[s:s+n] = True
					words_found.append((s, substring.lower()))
	if words_found:
		words_found.sort()
		for _, substring in words_found:
			split_hashtag.append(substring)
		return ' '.join(split_hashtag)
	elif not words_found:
		return (" <hashtag> ")

def main(argv):
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("-of", "--orig_file", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-pf", "--preproc_file", type=argparse.FileType('w'), default=sys.stdout)
	args = vars(parser.parse_args())
	processed_tweets = process_tweets(args["orig_file"], args["preproc_file"])
	return(processed_tweets)

if __name__ == "__main__":
	main(sys.argv)
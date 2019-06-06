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

#done: For now, keep tweet# and class in final preprocessed data
#done: Substitute numbers with "num" and alphanum chs with "alphaNum" except:
		# - tweet numbers
#done: Might be better to remove mentions (@'s)
#done: Implement spelling correction using multiple methods 
#done: Implement spelling correction
#TODO: Handle hashtags (split uncommon, made up ones)

def process_tweets(ORIGIN_TRAINING_FILE, PREPROC_TRAINING_FILE):
	processed_tweets = list() #final list of processed tweets (as list of tokens)
	stop_words = stopwords.words('english')
	urls = re.compile(r"http\S+") #find urls starting with 'http'
	urls2 = re.compile(r"www\.\w+\S+") #find urls of type www...
	urls3 = re.compile(r"\S+\.com") #find urls ending in .com
	alphanum1 = re.compile(r"[a-zA-Z]+\d+[a-zA-Z]+")  #find letter-digit(s) patterns
	alphanum2 = re.compile(r"[a-zA-Z]+\d+")  #find letter-digit(s) patterns
	alphanum3 = re.compile(r"\d+([a-zA-Z]+)\d+") #find digit(s)-letter-digit(s) patterns
	alphanum4 = re.compile(r"[a-zA-Z]+\-[a-zA-Z]+") #find letter-(dash)-letter pattern (we want to leave these as they are)
	non_alphanum = re.compile(r"[^\w+\-\.\:]") #find all non-alphanumeric characters except '[', ']', '-', '.' and ':'
	num = re.compile(r"\d+") #find natural numbers
	num2 = re.compile(r"\#\d+") #find hashtags followed by numbers
	num3 = re.compile(r"\d+\.\d+") #find real numbers
	num4 = re.compile(r"\d+\:\d+") #find time-like patterns
	num5 = re.compile(r"\d+\-\d+") #number followed/preceeded by dashes
	num6 = re.compile(r"\d+\-") #number followed/preceeded by dashes
	num7 = re.compile(r"\-\d+") #number followed/preceeded by dashes
	exclamations = re.compile(r"\!+") #one or more excl marks are kept intact
	questions = re.compile(r"\?{2,}") #same with two or more question marks
	excl_quest = re.compile(r"\?+\!+") #?! patterns are kept as they are
	ellipses = re.compile(r"\.{2,}") #ellipses are informative - might indicate dissapointment, etc. they are kept intact
	usr_mentions = re.compile(r"\@{1}[a-zA-Z0-9]+") #find user mentions
	hashtags1 = re.compile(r"\#{1}[a-zA-Z]+[0-9]+") #hashtags starting with letters
	hashtags2 = re.compile(r"\#{1}[0-9]+[a-zA-Z]+") #hashtags starting with a number
	pos_emojis = ([":)", ";)", ";-)", ":-)", ":))", ":-))", ";))", ";-))", 
	":o", "^^", "^_^", "<3", ":3", "\m/", "\_()_/", "\o/", "\_/"])
	neg_emojis = (["-__-", ":(", ":((", ":-(", ":-(("]) #avoid removing informative emoticons when removing non-alphanumeric characters

	for line in ORIGIN_TRAINING_FILE:
		result = list() #store result after processing is done
		line = twokenize.tokenizeRawTweetText(line)
		for index, word in enumerate(line):
			if index == 0:
				result.append(word)
			# dist_1 = set()
			# dist_2 = set()
			# candidate_words = set()
			if (re.match(urls, word) or re.match(urls2, word) or re.match(urls3, word)):
				result.append(" URL ")
				continue

			if word in stop_words:
				result.append(" ")
				continue

			if (index == 0): #skip the tweet ID
				result.append("")
				continue

			if (re.match(excl_quest, word)):
				result.append("?!")
				continue

			if (re.match(exclamations, word)):
				result.append("!!")
				continue

			if (re.match(questions, word)):
				result.append("??")
				continue

			if (re.match(ellipses, word)):
				result.append("..")
				continue

			if (re.match(hashtags1, word) or re.match(hashtags2, word)):
				result.append(word)
				continue

			if (word in pos_emojis):
				result.append(" posEmot ")
				continue

			if (word in neg_emojis):
				result.append(" negEmot ")
				continue

			if (index == 1 or re.match(alphanum4, word)):
				result.append(word)
				continue

			if (re.match(usr_mentions, word)):
				result.append(" userMnt ") #replace user mentions with [usermnt]
				continue

			#Otherwise, process the word normally
			word = re.sub(non_alphanum, " ", word) #Remove all non-alphanumeric characters
			word = re.sub(alphanum1, " alphaNum ", word)
			word = re.sub(alphanum2, " alphaNum ", word)
			word = re.sub(alphanum3, " alphaNum ", word)
			word = re.sub(num3, " num.num ", word)
			word = re.sub(num4, " num:num ", word)
			word = re.sub(num5, " num-num ", word)
			word = re.sub(num6, " num-", word)
			word = re.sub(num7, "-num ", word)
			word = re.sub(num, "num", word)
			word = re.sub(num2, "num", word)
			# word = correct_word(word, vocabulary)
			#TODO: test spelling corrector 
			# was not able to test with it, makes preprocessing very time consuming (~12h to process training data)
			
			result.append(word)

		processed_tweets.append(result)
		PREPROC_TRAINING_FILE.write(' '.join(result) + "\n")
	return(processed_tweets)



# def words(text, model): 
# 	# return (re.findall(r'\w+', text.lower()) in model)
# 	return (list(word for word in re.findall(r'\w+', open('../data/spelling_correction/big.txt').read()) if word in model))

# we_model = load_word_embeddings("../data/word_embeddings/glove.twitter.27B.25d.txt")
# vocabulary = Counter(words(open('../data/spelling_correction/big.txt').read(), we_model)) #extract the vocabulary for the language model - used in the spelling corrector

# def edits1(word):
#     letters    = 'abcdefghijklmnopqrstuvwxyz'
#     splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
#     deletes    = [L + R[1:]               for L, R in splits if R]
#     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
#     replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
#     inserts    = [L + c + R               for L, R in splits for c in letters]
#     return set(deletes + transposes + replaces + inserts)

# def edits2(word):
# 	return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# def known(words_set, vocabulary):
# 	return set(w for w in words_set if w in vocabulary)

# def word_probability(word, vocabulary_length=sum(vocabulary.values())):
# 	# return dict_freqs.get(word).count
# 	return (vocabulary[word] / vocabulary_length)

# def candidates(word, vocabulary):
# 	return known([word], vocabulary).union(known(edits1(word), vocabulary)).union(known(edits2(word), vocabulary))

# def correct_word(word, vocabulary):
# 	if len(candidates(word,vocabulary)) > 0:
# 		return (max(candidates(word, vocabulary), key=word_probability))
# 	else: 
# 		return (word)
'''Slightly modified spelling corrector based on edit distance by Peter Norvig
	https://norvig.com/spell-correct.html'''

def main(argv):
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("-of", "--orig_file", type=argparse.FileType('r'), default=sys.stdin)
	parser.add_argument("-pf", "--preproc_file", type=argparse.FileType('w'), default=sys.stdout)
	args = vars(parser.parse_args())
	processed_tweets = process_tweets(args["orig_file"], args["preproc_file"])
	return(processed_tweets)

if __name__ == "__main__":
	main(sys.argv)
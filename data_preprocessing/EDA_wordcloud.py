from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

data_filename = "../data/processed/csv/proc-twitter-training-data.csv"

data = pd.read_csv(data_filename)


neg_tweets = data[data.Sentiment == 0]
neu_tweets = data[data.Sentiment == 1]
pos_tweets = data[data.Sentiment == 2]

neg_text = list()
neu_text = list()
pos_text = list()

for tweet in neg_tweets.Tweet:
	neg_text.append(tweet)
neg_text = pd.Series(neg_text).str.cat(sep=' ')

for tweet in neu_tweets.Tweet:
	neu_text.append(tweet)
neu_text = pd.Series(neu_text).str.cat(sep=' ')

for tweet in pos_tweets.Tweet:
	pos_text.append(tweet)
pos_text = pd.Series(pos_text).str.cat(sep=' ')

negative_wordcloud = WordCloud(width=1600, height=1600, max_font_size=200).generate(neg_text)
neutral_wordcloud = WordCloud(width=1600, height=1600, max_font_size=200).generate(neu_text)
positive_wordcloud = WordCloud(width=1600, height=1600, max_font_size=200).generate(pos_text)

plt.figure()
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(neutral_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
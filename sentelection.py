#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import codecs
import csv
import nltk.stem
import nltk.corpus
import nltk
import sys
# allows csv to do utf-8 encoding
reload(sys)
sys.setdefaultencoding('utf-8')

def get_arguments():
	"""
	Gets arguments.
	"""

def read_senteval_csv(to_read):
	"""
	Reads tweets from senteval data set
	"""

	tweet_list=[]

	with open(to_read, 'r') as open_file:
		file_reader = csv.reader(open_file, delimiter=",")
		[tweet_list.append(row) for row in file_reader]
	
	return tweet_list[1:]

def read_tagged_senteval(to_read):

	tweet_list = []

	with open(to_read, 'r') as open_file:
		tweet = []
		for i, row in enumerate(open_file):
			split_row = row.rstrip().split("\t")
			if len(split_row) == 1:
				# end of tweet
				tweet_list.append(tweet)
				tweet = []
			else:
				tweet.append(split_row)

	return tweet_list[1:]


def join_tweets(tagged_tweets, semeval_tweets):
	"""
	Appends the tokenized and tagged tweets to the original semeval tweets
	At the end you'll have a list and each entry of the list will be, by index:
	1) unsplit tweet text
	2) subject/target (eg Atheism, Hillary Clinton)
	3) stance
	4) opinion towards
	5) sentiment
	6) Tweeboparser split and tagged tweets (another list)
	"""
	for i, line in enumerate(tagged_tweets):
		semeval_tweets[i].append(line)

	return semeval_tweets
			
def normalizer(unnormalized_list):
	"""
	Uses the nltk wordnet lemmatizer to lemmatize all the nouns, verbs, adjectives and adverbs
	IF it is lemmatized, it is put in the position after the unlemmatized spot in the
	token list
	"""
	lemmatizer = nltk.stem.WordNetLemmatizer()

	for i, token_list in enumerate(unnormalized_list):
		for j,thing in enumerate(token_list[5]):
			if thing[3] == 'V':
				unnormalized_list[i][5][j][2] = lemmatizer.lemmatize(thing[1], 'v')
			elif thing[3] == 'N':
				unnormalized_list[i][5][j][2] = lemmatizer.lemmatize(thing[1])
			elif thing[3] == 'R':
				unnormalized_list[i][5][j][2] = lemmatizer.lemmatize(thing[1], 'r')
			elif thing[3] == 'A':
				unnormalized_list[i][5][j][2] = lemmatizer.lemmatize(thing[1], 'a')
	return unnormalized_list

def dict_compare(token, dict_to_increment):
	if token in dict_to_increment:
		dict_to_increment[token] += 1
	else:
		dict_to_increment[token] = 1
	return dict_to_increment

def sort_dict(dict_to_sort):
	return [(k, dict_to_sort[k]) for k in sorted(dict_to_sort, key=dict_to_sort.get, reverse=True)]

def ngram_extractor(to_extract):
	""" This function gets the top n unigrams, bigrams and trigrams
	right now they're lowercased
	"""

	n = 100

	unigram_dict = {}
	bigram_dict = {}
	trigram_dict = {}

	stopwords = nltk.corpus.stopwords.words('english')
	stopwords = stopwords + [',', '.', '!', '"', '?', ':', "'", '&']

	for i, tweet in enumerate(to_extract):
		# check to see if there's a normalized form
		for i,token_list in enumerate(tweet[5]):
			if token_list[2] != '_' and token_list[2] not in stopwords:
				unigram_dict = dict_compare(token_list[2].lower(), unigram_dict)
			elif token_list[1] not in stopwords:
				unigram_dict = dict_compare(token_list[1].lower(), unigram_dict)

	for i, tweet in enumerate(to_extract):
		tokens = []
		for token_list in tweet[5]:
			if token_list[2] != "_":
				tokens.append(token_list[2].lower())
			else:
				tokens.append(token_list[1].lower())

		# fill out bigrams dictionary
		for i in nltk.ngrams(tokens,2):
			bigram_dict = dict_compare(i, bigram_dict)

		# I'm using _EOT_ to signify the tweet boundaries
		bigram_dict = dict_compare(("_EOT_", tokens[0]), bigram_dict)
		bigram_dict = dict_compare((tokens[len(tokens)-1], "_EOT_"), bigram_dict)

		# and trigrams
		for i in nltk.ngrams(tokens, 3):
			trigram_dict = dict_compare(i, trigram_dict)

		trigram_dict = dict_compare(("_EOT_", tokens[0], tokens[1]), trigram_dict)
		trigram_dict = dict_compare((tokens[len(tokens)-2], tokens[len(tokens)-1], "_EOT_"), trigram_dict)

	sorted_unigrams = sort_dict(unigram_dict)
	sorted_bigrams = sort_dict(bigram_dict)
	sorted_trigrams = sort_dict(trigram_dict)

	return (sorted_unigrams[:n], sorted_bigrams[:n], sorted_trigrams[:n])

def main():
	# read the untagged tweets
	semeval_path = "data/semeval_2016_data/StanceDataset/"
	first_path = semeval_path + "train_two.csv"
	train_tweets = read_senteval_csv(first_path)
	second_path = semeval_path + "test_two.csv"
	test_tweets = read_senteval_csv(second_path)

	# read tagged tweets
	tagged_train_tweets = read_tagged_senteval(semeval_path+"train_two_text.txt.predict")
	tagged_test_tweets = read_tagged_senteval(semeval_path+"test_two_text.txt.predict")

	# join them
	joined_train = join_tweets(tagged_train_tweets, train_tweets)
	joined_test = join_tweets(tagged_test_tweets, test_tweets)

	# lemmatizes the tokens
	normalized_train = normalizer(joined_train)

	feature_ngrams = ngram_extractor(normalized_train)
	print feature_ngrams

if __name__ == "__main__":
	main()

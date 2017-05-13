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
import unicodecsv as csv
from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

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

	with codecs.open(to_read, 'r','utf-8') as open_file:
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
				unnormalized_list[i][5][j][2] = unicode(lemmatizer.lemmatize(thing[1], 'v'))
			elif thing[3] == 'N':
				unnormalized_list[i][5][j][2] = unicode(lemmatizer.lemmatize(thing[1]))
			elif thing[3] == 'R':
				unnormalized_list[i][5][j][2] = unicode(lemmatizer.lemmatize(thing[1], 'r'))
			elif thing[3] == 'A':
				unnormalized_list[i][5][j][2] = unicode(lemmatizer.lemmatize(thing[1], 'a'))
	return unnormalized_list

def dict_compare(token, dict_to_increment):
	if token in dict_to_increment:
		dict_to_increment[token][0] += 1
	else:
		dict_to_increment[token] = [1,0]
	return dict_to_increment

def sort_dict(dict_to_sort):
	return [(k, dict_to_sort[k]) for k in sorted(dict_to_sort, key=dict_to_sort.get, reverse=True)]

def ngram_extractor(to_extract):
	""" This function gets the top n unigrams, bigrams and trigrams
	right now they're lowercased
	"""
	print "Extracting N-grams"

	n = 500

	unigram_dict = {}
	bigram_dict = {}
	trigram_dict = {}

	stopwords = nltk.corpus.stopwords.words('english')
	stopwords = stopwords + [',', '.', '!', '"', '?', ':', "'", '&']
	
	# convert them to unicode
	stopwords = [unicode(i) for i in stopwords]

	for i, tweet in enumerate(to_extract):
		tweet_unigrams = []

		# check to see if there's a normalized form
		for i,token_list in enumerate(tweet[5]):
			if token_list[2] != '_' and token_list[2] not in stopwords:
				# increment absolute count
				unigram_dict = dict_compare(token_list[2].lower(), unigram_dict)
				# this is used to increment the document count
				tweet_unigrams.append(token_list[2].lower())
				
			elif token_list[1] not in stopwords:
				unigram_dict = dict_compare(token_list[1].lower(), unigram_dict)
				tweet_unigrams.append(token_list[1].lower())

		# increment number of documents
		for unigram in set(tweet_unigrams):
			unigram_dict[unigram][1] += 1

	for i, tweet in enumerate(to_extract):
		tokens = []
		bigrams = []
		trigrams = []

		for token_list in tweet[5]:
			if token_list[2] != "_":
				tokens.append(token_list[2].lower())
			else:
				tokens.append(token_list[1].lower())

		# fill out bigrams dictionary
		for i in nltk.ngrams(tokens,2):
			bigrams.append(i)
			bigram_dict = dict_compare(i, bigram_dict)

		# I'm using _EOT_ to signify the tweet boundaries
		bigram_dict = dict_compare(("_EOT_", tokens[0]), bigram_dict)
		bigram_dict = dict_compare((tokens[len(tokens)-1], "_EOT_"), bigram_dict)

		# this increments the document counter
		for bigram in set(bigrams):
			bigram_dict[bigram][1] += 1
		bigram_dict[("_EOT_", tokens[0])][1] += 1
		bigram_dict[(tokens[len(tokens)-1], "_EOT_")][1] += 1

		# and trigrams
		for i in nltk.ngrams(tokens, 3):
			trigram_dict = dict_compare(i, trigram_dict)
			trigrams.append(i)

		trigram_dict = dict_compare(("_EOT_", tokens[0], tokens[1]), trigram_dict)
		trigram_dict = dict_compare((tokens[len(tokens)-2], tokens[len(tokens)-1], "_EOT_"), trigram_dict)


		for trigram in set(trigrams):
			trigram_dict[trigram][1] += 1
		trigram_dict[("_EOT_", tokens[0], tokens[1])][1] += 1
		trigram_dict[(tokens[len(tokens)-2], tokens[len(tokens)-1], "_EOT_")][1] += 1

	sorted_unigrams = sort_dict(unigram_dict)
	sorted_bigrams = sort_dict(bigram_dict)
	sorted_trigrams = sort_dict(trigram_dict)

	return (sorted_unigrams[:n], sorted_bigrams[:n], sorted_trigrams[:n])

def condensed_form(to_condense):
	"""
	Returns a form of the token without any duplication
	"""

	to_condense = to_condense.lower()

	condensed_token = ''
	last_char = ' '

	for character in to_condense:
		if character == last_char:
			pass
		else:
			condensed_token = condensed_token + character

		last_char = character

	return condensed_token

def lengthened_validator(tweet_dict):
	"""
	If a character is never tripled up, it doesn't really matter
	"""


	for token_form in tweet_dict:
		count = 0
		last_char = ' '
		for character in token_form:
			if character == last_char:
				count += 1
				last_char = character
				if count > 2:
					return True
			else:
				last_char = character
				count = 0

	return False


def lengthened_words(tweets):
	"""
	Returns a dictionary 
	Each key is a condensed form of a word each value is another dictionary
	where each 'full word' is a key and its occurence is the value

	Heath - you might want to correct for numbers, punctuation
	"""
	condensed_token = ''
	full_token = ''
	condensed_dict = {}
	for tweet in tweets:
		for token_list in tweet[5]:
			if token_list[2] == '_':
				condensed_token = condensed_form(token_list[1])
				full_token = token_list[1].lower()
			else:
				condensed_token = condensed_form(token_list[2])
				full_token = token_list[2].lower()
			
			if condensed_token not in condensed_dict:
				condensed_dict[condensed_token] = {full_token:1}
			else:
				if full_token not in condensed_dict[condensed_token]:
					condensed_dict[condensed_token][full_token] = 1
				else:
					condensed_dict[condensed_token][full_token] += 1

	# check to see if it is actually a valid doubled form
	legit_dict = {}
	for thing in condensed_dict:
		# if there are no triple characters, delete the entry
		if lengthened_validator(condensed_dict[thing]):
			legit_dict[thing] = condensed_dict[thing]
		

	return legit_dict

def n_gram_dict_updater(n_gram, n_gram_dict):
	"""
	Updates a dict's count for the given n_gram
	"""
	if n_gram in n_gram_dict:
		n_gram_dict[n_gram][0] += 1
	else:
		n_gram_dict[n_gram] = [1,0]

	return n_gram_dict

def character_n_grams(tweet_dict):
	"""
	This returns a tuple of two dictionaries, character bigrams and trigrams
	The entry in each dictionary is the character n-gram
	The value is a list. The first value is the absolute count, the second
	is the number of documents that it occurs in, so we can compute tf idf
	"""
	char_bi_grams = {}
	char_tri_grams = {}

	for tweet_list in tweet_dict:
		tweet_bi_grams = []
		tweet_tri_grams = []

		# update absolute count
		for bi_gram in [tweet_list[0][i:i+2] for i in range(len(tweet_list[0])-1)]:
			char_bi_grams = n_gram_dict_updater(bi_gram, char_bi_grams)
			tweet_bi_grams.append(bi_gram)

		# and increment for number of documents
		for bi_gram in set(tweet_bi_grams):
			char_bi_grams[bi_gram][1] += 1
				

		for tri_gram in [tweet_list[0][i:i+3] for i in range(len(tweet_list[0])-2)]:
			char_tri_grams = n_gram_dict_updater(tri_gram, char_tri_grams)
			tweet_tri_grams.append(tri_gram)

		for tri_gram in set(tweet_tri_grams):
			char_tri_grams[tri_gram][1] += 1

	return (char_bi_grams, char_tri_grams)

def feature_maker(tweets, word_ngrams):

	feature_set = []

	# go through tweets one by one
	for tweet in tweets:
		features = []
		
		# start with unigrams
		# tuple of all unigrams in a tweet
		unigrams = []
		for token in tweet[5]:
			if token[2] == u'_':
				unigrams.append(token[1])
			else:
				unigrams.append(token[2])

		unigrams = tuple(unigrams)

		# for every feature word, if it's in the unigram list, give it a 1
		# otherwise a 0
		for unigram in word_ngrams[0]:
			if unigram[0] in unigrams:
				features.append(1)
			else:
				features.append(0)

		# now we see if it has an exclamation point
		has_exclamation = False
		for unigram in unigrams:
			if '!' in unigram:
				has_exclamation = True

		if has_exclamation:
			features.append(1)
		else:
			features.append(0)

		# new if it has a hashtag
		has_hash = False
		for unigram in unigrams:
			if '#' in unigram:
				has_hash = True

		if has_hash:
			features.append(1)
		else:
			features.append(0)

		# the feature set is a list of lists
		feature_set.append(features)

	return feature_set

def label_maker(tweets):
	"""
	This returns a list of list, right now it's just about whether it is neutral
	or not
	"""

	label_set = []

	for tweet in tweets:
		if tweet[2] == 'NONE':
			label_set.append(0)
		else:
			label_set.append(1)
	
	return label_set

def label_balancer(features, labels):
	"""
	Here's the deal, the nones are outnumbered by 3-1, so I need to balance it
	"""

	new_features = []
	new_labels = []

	for i, label in enumerate(labels):
		if label == 0:
			new_features.append(features[i])
			new_features.append(features[i])
			new_labels.append(labels[i])
			new_labels.append(labels[i])
		else:
			new_features.append(features[i])
			new_labels.append(labels[i])

	return (new_features, new_labels)
	
			
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
	normalized_test = normalizer(joined_test)

	# note to change the number of ngrams used, edit the variable n in the
	# ngram extractor function, right now it's 50
	feature_ngrams = ngram_extractor(normalized_train)

	# create a dictionary of lengthened words
	lengthened_dict = lengthened_words(joined_train)

	# get character ngrams
	n_gram_dicts = character_n_grams(joined_train)


	# first we're going to start separating things with sentiment vs things without
	# start constructing feature set
	train_features = feature_maker(normalized_train, feature_ngrams)
	test_features = feature_maker(normalized_test, feature_ngrams)

	# give it labels
	train_labels = label_maker(normalized_train)
	test_labels = label_maker(normalized_test)

	balanced = label_balancer(train_features, train_labels)
	balanced_train_features = balanced[0]
	balanced_train_labels = balanced[1]

			


	clf = svm.SVC()
	clf.fit(balanced_train_features, balanced_train_labels)

	neut_predictions = clf.predict(test_features)
	print "SVM"
	print(confusion_matrix(test_labels, neut_predictions))
	print f1_score(test_labels, neut_predictions),'\n'

	naive_bayes = GaussianNB()
	naive_bayes.fit(train_features, train_labels)
	nb_predictions = naive_bayes.predict(test_features)

	print "NB"
	print (confusion_matrix(test_labels, nb_predictions))
	print f1_score(test_labels, nb_predictions), '\n'

	random_forest = RandomForestClassifier()
	random_forest.fit(train_features, train_labels)
	rf_predictions = random_forest.predict(test_features)

	print "RF"
	print (confusion_matrix(test_labels, rf_predictions))
	print f1_score(test_labels, rf_predictions), '\n'

	logistic_regression = LogisticRegression()
	logistic_regression.fit(train_features, train_labels)
	lr_predictions = logistic_regression.predict(test_features)
	
	print"LR"
	print (confusion_matrix(test_labels, lr_predictions))
	print f1_score(test_labels, lr_predictions), '\n'

	estimators=[('lr', logistic_regression), ('gnb', naive_bayes), ('rf', random_forest)]
	voting_classifier = VotingClassifier(estimators, voting='hard', weights=[1,1,5])
	voting_classifier.fit(train_features, train_labels)
	
	vc_predictions = voting_classifier.predict(test_features)
	print "VC"
	print (confusion_matrix(test_labels, vc_predictions))
	print f1_score(test_labels, vc_predictions)
	
	
	



if __name__ == "__main__":
	main()

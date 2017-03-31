#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import codecs
import csv
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

	# get n-grams


if __name__ == "__main__":
	main()

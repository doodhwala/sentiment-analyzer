import pytreebank
import os
import csv
import re
from gensim.models import Word2Vec

sentiment = [0, 1, 2, 3, 4]
# sentiment = [0, 0, 1, 2, 2]
datatype = ["train", "dev", "test"]

dataset = pytreebank.load_sst('trees')
print 'Dataset loaded...'

stop_words = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'only', u'own', u'same', u'so', u'than', u'too', u's', u't', u'will', u'just', u'now', u'd', u'll', u'm', u'o', u've', u'y', u'ma']

def clean(sentence):
	sentence = re.sub('[#.,$%\-|~\/&\"\'`*+=!?;:()^]', '', sentence)
	sentence = sentence.lower()

	sentence = [word for word in sentence.split() if word not in stop_words]
	return ' '.join(sentence)

def create_csv():
	for i in datatype:
		csvfile = open(i + '.csv', 'wb')
		# csvfile = open(i + '2.csv', 'wb')
		writer = csv.writer(csvfile)
		for j in dataset[i]:
			example = j
			# extract spans from the tree
			for label, sentence in example.to_labeled_lines():
				sentence = clean(sentence)
				if sentence:
					writer.writerow([sentence.encode('utf-8'), sentiment[label]])
		csvfile.close()

def create_sentence_csv():
	f = open('train_sentence.txt', 'w')
	for j in dataset["train"]:
		example = j
		span_sentence = True
		for label, sentence in example.to_labeled_lines():
			if span_sentence:
				sentence = clean(sentence)
				if sentence:
					f.write(sentence.encode('utf-8'))
					f.write('\n')
				span_sentence = False
	f.close()

def create_model():
	f = open('trees/train_sentence.txt', 'r')
	lines = f.read().split('\n')
	sentences = []
	for i in lines:
		sentences.append(i.split())
	model = Word2Vec(sentences, window=5, size=32, iter=100, min_count=1)
	model.save('model')
	f.close()

if __name__ == '__main__':
	create_csv()
	create_sentence_csv()
	create_model()
import re
import sys
import pytreebank
import os
import preprocess
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

def get_phrase(sentence):
	phrase = ' '.join(re.findall('[a-zA-Z]+', sentence))
	return phrase

def get_part(string):
	brackets = 0
	part = ''
	for char in string:
		if char == '(':
			brackets += 1
		elif char == ')':
			brackets -= 1
		if brackets == 0:
			break
		part += char
	return part

def create_phrases(tree_string):
	# java -cp "*" -mx5g edu.stanford.nlp.sentiment.BuildBinarizedDataset -input tree_string
	#tree_string = '(2 (2 (2 Stanford) (2 University)) (2 (2 (2 is) (2 (2 located) (2 (2 in) (2 (2 (2 California) (2 .)) (2 (2 It) (2 (2 is) (2 (2 (2 (2 a) (2 (2 great) (2 university))) (2 ,)) (2 (2 founded) (2 (2 in) (2 1891)))))))))) (2 .)))'
	with open('temp.txt', 'w') as f:
		f.write('2 ' + tree_string)
	os.system('java -cp "../stanford-corenlp-full-2016-10-31/*" -mx5g edu.stanford.nlp.sentiment.BuildBinarizedDataset -input temp.txt > output.txt')
	with open('output.txt', 'r') as f:
		tree_string = f.read()
		# print tree_string
		tree = pytreebank.create_tree_from_string(tree_string)
		return tree.to_labeled_lines()

def create_tree(sentence):
	text = (sentence)
	output = nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,sentiment', 'outputFormat': 'json', 'binaryTree': True})
	output = output['sentences'][0]['parse'].split('\n')
	output = [line.strip() for line in output]
	output = ' '.join(output)
	return output

if __name__ == '__main__':
	sentence = raw_input('Enter a sentence to parse: ')
	output = create_phrases(sentence)
	print output
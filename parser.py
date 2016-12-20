import re
import preprocess
import pytreebank
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
	phrases = []
	tree_string = re.sub('\([a-zA-Z]+ ', '(2 ', tree_string)[3:-1]
	print tree_string
	# tree_string = '(2 (2 (2 (2 what)) (2 (2 a) (2 cool) (2 python) (2 package))))'
	# tree_string = '(4 (2 what) (3 (2 a) (3 (3 (3 cool) (2 python)) (2 package))))'

	phrases = set()
	for index in xrange(len(tree_string)):
		char = tree_string[index]
		if char == '(':
			part = get_part(tree_string[index:])
			# print 'part: ', part
			phrases.add(get_phrase(part))
	return list(phrases)
'''
def create_phrases(tree_string):
	# tree_string = '(2 (2 (2 global) (2 warming)) (2 (2 is) (2 (2 a) (2 hoax))))'
	# tree_string = re.sub('\([a-zA-Z]+ ', '(2 ', tree_string)[3:-1]
	# print tree_string
	# tree_string = '(2 (2 aliens)) (2 (2 are) (2 (2 attacking) (2 (2 my) (2 planet))))'
	tree = pytreebank.create_tree_from_string(tree_string)
	print tree.to_labeled_lines()
'''
def create_tree(sentence):
	text = (sentence)
	output = nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,sentiment', 'outputFormat': 'json'})
	output = output['sentences'][0]['parse'].split('\n')
	output = [line.strip() for line in output]
	output = ' '.join(output)
	return output

if __name__ == '__main__':
	sentence = raw_input('Enter a sentence to parse: ')
	output = create_tree(sentence)
	print output
	print create_phrases(output)
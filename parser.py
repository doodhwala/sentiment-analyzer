from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

def create_tree():
	text = (raw_input())
	output = nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,depparse,parse', 'outputFormat': 'json'})
	output = output['sentences'][0]['parse'].split('\n')
	output = [line.strip() for line in output]
	output = ' '.join(output)
	return output

if __name__ == '__main__':
	sentence = raw_input('Enter a sentence to parse:')
	output = create_tree()
	print output
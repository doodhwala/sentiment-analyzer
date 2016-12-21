import re

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
 
def create_tree():
	with open('output.txt', 'r') as f:
		tree_string = f.readline()
		phrases = eval(f.readline())
		phrases, ratings = zip(*phrases)
		phrases = list(phrases)
		ratings = list(ratings)
		for index in xrange(len(tree_string)):
			char = tree_string[index]
			if char == '(':
				part = get_part(tree_string[index:])
				phrase = get_phrase(part)
				# print part, phrase
				if phrase in phrases:
					rating = ratings[phrases.index(phrase)]
					if TYPE == 3:
						rating += 1
					tree_string = tree_string[:index+1] + str(rating) + tree_string[index+2:]
	with open('output.txt', 'w') as f:
		f.write(tree_string)

if __name__ == '__main__':
	TYPE = 5
	create_tree()
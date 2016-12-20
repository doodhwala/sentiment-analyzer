from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

text = (raw_input())
output = nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,depparse,parse', 'outputFormat': 'json'})
output = output['sentences'][0]['parse'].split('\n')
output = [line.strip() for line in output]
output = ' '.join(output)
print output
'''
GETTING FAMILIAR WITH BASIC Stanford CoreNLP TASKS

Stanford CoreNLP Python tutorial: https://towardsdatascience.com/natural-language-processing-using-stanfords-corenlp-d9e64c1e1024 
Debugging nlp.annotate() outputs: https://stackoverflow.com/questions/39605817/is-there-any-way-to-have-pycorenlps-nlp-annotate-always-return-the-same-typ 
'''

from pycorenlp import StanfordCoreNLP
import json
nlp = StanfordCoreNLP('http://localhost:9000')

file = open('text.txt', 'r')

text1 = file.read()
result1 = nlp.annotate(text1,
                   properties={
                       'annotators': 'sentiment, ner, pos',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

if type(result1) is str or type(result1) is unicode:
    result1 = json.loads(result1, strict=False)


# sentiment analysis results
print('\nSentiment Analysis\n----------')
for s in result1["sentences"]:
    print("{}: '{}': {} (Sentiment Value) {} (Sentiment)".format(
        s["index"],
        " ".join([t["word"] for t in s["tokens"]]),
        s["sentimentValue"], s["sentiment"]))


# pos tagging results
print('\nPoS Tagging\n----------')
pos1 = []
for word in result1["sentences"][2]["tokens"]:
    pos1.append('{} ({})'.format(word["word"], word["pos"]))
    
" ".join(pos1)

for elem in pos1:
    print(elem, end=' ')
print('\n')

# ner 
print('\nNER\n----------')
text2 = "The earphones Jim bought for Jessica while strolling through the Apple store at the airport in Chicago, USA, were great."
result2 = nlp.annotate(text2,
                   properties={
                       'annotators': 'sentiment, ner, pos',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

if type(result2) is str or type(result2) is unicode:
    result2 = json.loads(result2, strict=False)

pos2 = []
for word in result2["sentences"][0]['tokens']: # [1] changed to [0]
    pos2.append('{} ({})'.format(word['word'], word['ner']))
    
" ".join(pos2)

for elem in pos2:
    print(elem, end=' ')
print('\n')
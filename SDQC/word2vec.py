import pickle
import pprint as pp
import numpy
import nltk
import re

import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

import inflect
p=inflect.engine()

from autocorrect import spell

from gensim.models import Word2Vec,KeyedVectors
my_model=KeyedVectors.load_word2vec_format('glove.twitter_50d.bin',binary=True)

#files=['1-tweets.pkl','2-tweets.pkl','4-tweets.pkl','5-tweets.pkl','6-tweets.pkl','8-tweets.pkl']  #train
#files=['1-tweets.pkl','2-tweets.pkl','3-tweets.pkl','4-tweets.pkl','5-tweets.pkl','6-tweets.pkl','7-tweets.pkl','8-tweets.pkl']  #all tweets
files=['3-tweets.pkl']
#files1=['1-tweets.pkl']
#labels=['1-ebola-essien_labels.pkl','2-putinmissing_labels.pkl','4-prince-toronto_labels.pkl','5-ferguson_labels.pkl','6-ottawashooting_labels.pkl','7-charliehebdo_labels.pkl','8-sydneysiege_labels.pkl']
#files=['7-tweets.pkl']
#labels=['1-ebola-essien_labels.pkl','2-putinmissing_labels.pkl','4-prince-toronto_labels.pkl','5-ferguson_labels.pkl','6-ottawashooting_labels.pkl','8-sydneysiege_labels.pkl'] #train
#labels=['3-germanwings-crash_labels.pkl','7-charliehebdo_labels.pkl']


sentences=[]
maxx=0
a=""
num_count=0
for i in range(0,len(files)):
  with open(files[i],'rb') as r:
    tweets=pickle.load(r)
    for j in range(0,len(tweets)):
      #tweets[j] is one conversation set(list of tweets,edges)
      #s=[]
      #for tokens in word_tokenize(tweets[j]):
      #  s.append(tokens)
      for t in tweets[j]:
        #print t
        text=t[0].encode('ascii','ignore')

        
        text=re.sub("-"," - ",text)
        text=re.sub(r"\.+"," . ",text)
        text=re.sub("/"," or ",text)
        text=re.sub(","," , ",text)
        text=re.sub("\*"," * ",text)
        text=re.sub("'"," ' ",text)
        text=re.sub("srry","sorry",text)
        text=re.sub("fkn","fucking",text)
        text=re.sub("f\*\*\*ing","fucking",text)
        text=re.sub("somthng","something",text)
        text=re.sub("s\*\*t","shit",text)
        if "germanwings" in text:
          print "LOL"
          text=text.replace("germanwings","german  wings")
          print text
        text=text.replace("airtravel","air travel")
        tokens=nltk.word_tokenize(text.lower())
        for token in tokens:
          if unicode(token,'utf-8').isnumeric():
            num_count+=1
            num2words=p.number_to_words(token)
            text=re.sub(token," "+num2words+" ",text)
            text=re.sub("-"," - ",text)
            text=text.encode('ascii','ignore')
          if token not in my_model:
            text=text.replace(token,spell(token))
            

        tokens=nltk.word_tokenize(text.lower())
        if len(tokens)>maxx:
          maxx=len(tokens)
          a=text
        #if 'dying' in tokens:
          #print t[0]
        sentences.append(tokens)
"""
with open('all-tweets.pkl','wb') as w:
  pickle.dump(sentences,w)
with open('all-tweets.pkl','rb') as r:
  sentences=pickle.load(r)


from gensim.models import Word2Vec

model=Word2Vec(sentences,size=100,min_count=1,sg=1)

print "Word Embeddung Model is SUCCESSFULLY created"
print "MAX FEATURES",maxx,a
words=list(model.wv.vocab)

print len(words)
print words[1],words[2],words[3]

#model.wv.save_word2vec_format('my_first_model.bin')
model.save('my_first_model1.bin')

my_model=Word2Vec.load('my_first_model1.bin')"""
#my_model=Word2Vec.load('glove_50d.bin')
"""
print my_model['hate']
#print my_model['Does']
print my_model['does']
print my_model['?']
print my_model['dying']
if '.' in my_model:
  print "SAI"
"""

count=0
for i in range(0,len(sentences)):
  for j in sentences[i]:
    if j not in my_model:
      count+=1
      #print j,sentences[i]
print count,num_count

#print my_model['german']
#print my_model['wings']
print "MAX FEATURES",maxx,a

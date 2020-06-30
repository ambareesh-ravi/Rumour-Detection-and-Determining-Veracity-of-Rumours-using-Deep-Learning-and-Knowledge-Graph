import json

import nltk
import pycrfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from nltk import word_tokenize
import string
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()
from Lexicons import false_antonyms,false_synonyms,negation_words,SpeechAct
import numpy,re
import time
import scikitplot as skplt
import matplotlib.pyplot as plt
from Save_Output import save_output

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

def check_punc(text,sym):
	tokens=word_tokenize(text)
	cnt=0
	for t in tokens:
		if t==sym:
			cnt+=1
	return cnt

def content_formatting(text,type):
	if type=='capital_ratio':
		capital_letters= sum(1 for w in text if w.isupper())
		f=float(capital_letters/len(text))
		return f
	tokens =word_tokenize(re.sub(r'([^\s\w]|_)+', '',text.lower()))
	count=0
	for token in tokens:
		if token in type:
			count+=1
	return count

def word_count(text,type):
	count=0
	if type=='positive':
		file=open("positive-words.txt")
	elif type=='negative':
		file=open("negative-words1.txt")
	elif type=='swear':
		file=open("swear-words.txt")
	elif type=='question':
		file=open("question-words.txt")
	for token in word_tokenize(text):
		for line in file.readlines():
			split_line=string.split(line)
			if token.lower()==split_line[0].lower():
				count=count+1
	return count

def sentiment_analysis(text,type):
             """
	blob=TextBlob(text)
	if type=='polarity':
		return blob.sentiment.polarity
	if type=='sub':
		return blob.sentiment.subjectivity
	"""
             vs = analyzer.polarity_scores(text)
             return vs[type]
def get_extra_features(tweets):
	feat=[]
	for text in tweets:
		f=[]
		f.append(check_pos_tag(text, 'noun'))
		f.append(check_pos_tag(text, 'verb'))
		f.append(check_pos_tag(text, 'adj'))
		f.append(check_pos_tag(text, 'adv'))
		f.append(check_pos_tag(text, 'pron'))
		f.append(check_punc(text,'?'))
		f.append(check_punc(text,'!'))
		f.append(content_formatting(text,'capital_ratio'))
		f.append(word_count(text,'positive'))
		f.append(word_count(text,'negative'))
		f.append(word_count(text,'swear'))
		f.append(word_count(text,'question'))
		f.append(sentiment_analysis(text,'pos'))
		f.append(sentiment_analysis(text,'neg'))
		f.append(sentiment_analysis(text,'compound'))
		f.append(content_formatting(text,'false_synonyms'))
		f.append(content_formatting(text,'false_antonyms'))
		f.append(content_formatting(text,'negation_words'))
		#f.append(sentiment_analysis(text,'polarity'))
		#f.append(sentiment_analysis(text,'sub'))
		feat.append(f)
	arr=numpy.array(feat)
	print "Extra",arr.shape
	return arr
	

def main():
	
	
	with open('dataset1.json','rb') as read_handler:
		final=json.load(read_handler)
	
	with open('dev-key.json','r') as r:
		dev=json.load(r)


	tweet_data=[]
	labels=[]
	key_list=[]
	for key,value in dev['subtaskaenglish'].items():
		key_list.append(str(key))

	available_keys=[]
	for i in range(len(key_list)):
		for key,value in final.items():
			for j in range(len(value)):
				if key_list[i]==str(value[j]['id']):
					tweet=value[j]['source']['text']
					tweet=tweet.encode('ascii','ignore')
					tweet_data.append(tweet)
					available_keys.append(key_list[i])
				else:
					for k in range(len(value[j]['replies'])):
						if key_list[i]==str(value[j]['replies'][k]['id']):
							tweet=value[j]['replies'][k]['text']
							tweet=tweet.encode('ascii','ignore')
							tweet_data.append(tweet)
							available_keys.append(key_list[i])
				
			
	print len(tweet_data)
	print len(key_list)
	print len(available_keys)
	with open("tfidf_vectorizer.pkl","rb") as r:
		vectorizer=pickle.load(r)
	features_test=vectorizer.transform(tweet_data)
	
	
	print "Features successfully extracted"
	print type(features_test)
	
	
	features_test=features_test.toarray()
	extra_features_test=get_extra_features(tweet_data)
	features_test=numpy.concatenate((features_test,extra_features_test),axis=1)

	
	
	
	with open('LR_model.pkl','rb') as r:
		model=pickle.load(r)
	predictions=model.predict(features_test)
	dummy=[]
     
	save_output(available_keys,predictions,dummy,dummy,dummy)
	

if __name__=="__main__":
	main()
	
	


from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy
from nltk import word_tokenize
import string
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()
from sarcasm import getSarcasmScore
import re,math
from collections import Counter
from Lexicons import false_antonyms,false_synonyms,negation_words,SpeechAct

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}


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
	if sym in text:
		return 1
	else:
		return 0

def content_formatting(text,type):
	if type=='capital_ratio':
		capital_letters= sum(1 for w in text if w.isupper())
		if len(text)==0:
			return 0.
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
	vs = analyzer.polarity_scores(text)
	return vs[type]

def cosine_similarity(text1,text2):
	
	split_words=re.compile(r'\w+')
	words=split_words.findall(text1)
	vec1=Counter(words)
	words=split_words.findall(text2)
	vec2=Counter(words)

	
	intersection=set(vec1.keys()) & set(vec2.keys())
	numerator=sum([vec1[x]*vec2[x] for x in intersection])
	a=sum([vec1[x]**2 for x in vec1.keys()])
	b=sum([vec2[x]**2 for x in vec2.keys()])
	denominator=math.sqrt(a)*math.sqrt(b)
	
	
	if not denominator:
		return 0.0
	else:
		return float(numerator)/denominator
	
def get_speechAct_vector(text):
	sv=[]
	for key in SpeechAct.keys():
		count=0
		for verb in SpeechAct[key]:
			if verb in text.lower():
				count+=1
		sv.append(count)
	return sv
def get_extra_features(tweets):
	feat=numpy.zeros(59)#22+37
	is_source=1
	for i in range(0,len(tweets)):
		f=[]
		f.append(check_pos_tag(tweets[i], 'noun'))
		f.append(check_pos_tag(tweets[i], 'verb'))
		f.append(check_pos_tag(tweets[i], 'adj'))
		f.append(check_pos_tag(tweets[i], 'adv'))
		f.append(check_pos_tag(tweets[i], 'pron'))
		f.append(check_punc(tweets[i],'?'))
		f.append(check_punc(tweets[i],'!'))
		f.append(check_punc(tweets[i],'.'))
		f.append(content_formatting(tweets[i],'capital_ratio'))
		f.append(word_count(tweets[i],'positive'))
		f.append(word_count(tweets[i],'negative'))
		f.append(word_count(tweets[i],'swear'))
		f.append(word_count(tweets[i],'question'))
		f.append(sentiment_analysis(tweets[i],'pos'))
		f.append(sentiment_analysis(tweets[i],'neg'))
		f.append(sentiment_analysis(tweets[i],'compound'))
		f.append(content_formatting(tweets[i],'false_synonyms'))
		f.append(content_formatting(tweets[i],'false_antonyms'))
		f.append(content_formatting(tweets[i],'negation_words'))
		f.append(getSarcasmScore(tweets[i]))
		f.extend(get_speechAct_vector(tweets[i]))#37 Features
		if is_source==1:
			#f.append(1)
			f.append(cosine_similarity(tweets[i],tweets[i]))#Just compare with the same tweet
			is_source=0
		else:
			#f.append(0)
			f.append(cosine_similarity(tweets[i],tweets[i-1]))#Compare with the previous tweet
		f.append(cosine_similarity(tweets[i],tweets[0]))#Compare with the Source tweet
		f_=numpy.array(f)
		feat=numpy.vstack((feat,f_))
	feat=numpy.delete(feat,(0),axis=0)
	return feat
	

def get_tfidf_features(tweets_test):
	
	#TFIDF VECTORIZER
	with open("Trained_tfidf_model.pkl","r") as twt_train:
		vectorizer = pickle.load(twt_train)
	features_test_tfidf=vectorizer.transform(tweets_test)
	#This will create a scipy.sparse.csr.csr_matrix which should be converted to an array
	features_test_tfidf=features_test_tfidf.toarray()
	return features_test_tfidf

def get_main_features(tweets_test,feature_set=['tfidf']):
	if 'tfidf' in feature_set:
		return get_tfidf_features(tweets_test)
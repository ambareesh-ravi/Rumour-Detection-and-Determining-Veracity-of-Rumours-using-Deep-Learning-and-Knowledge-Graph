
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy
from nltk import word_tokenize
import string
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()
from Sarcasm_Detection_Module.sarcasm import getSarcasmScore
import re,math
from collections import Counter
from Data.Lexicons import false_antonyms,false_synonyms,negation_words,SpeechAct
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
	"""
	tokens=word_tokenize(text)
	cnt=0
	for t in tokens:
		if t==sym:
			cnt+=1
	return cnt"""
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
		file=open("Data/positive-words.txt")
		#print ("Done")
	elif type=='negative':
		file=open("Data/negative-words.txt")
	elif type=='swear':
		file=open("Data/swear-words.txt")
	elif type=='question':
		file=open("Data/question-words.txt")
	for token in word_tokenize(text):
		for line in file.readlines():
			split_line=string.split(line)
			if token.lower()==split_line[0].lower():
				count=count+1
	return count

def sentiment_analysis(text,type):
	vs = analyzer.polarity_scores(text)
	return vs[type]
	"""
	blob=TextBlob(text)
	if type=='polarity':
		return blob.sentiment.polarity*100
	if type=='sub':
		return blob.sentiment.subjectivity
	"""

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
	feat=numpy.zeros(58)#21+37
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
		#f.append(getSarcasmScore(tweets[i]))
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

def manual_tokenize(text,remove_stopwords=False):
	text=re.sub("-"," - ",text)
	text=re.sub(r"\.+"," . ",text)
	text=re.sub("/"," or ",text)
	text=re.sub(","," , ",text)
	text=re.sub("\*"," * ",text)
	text=re.sub("'"," ' ",text)
	text=re.sub("="," = ",text)
	text=re.sub("srry","sorry",text)
	text=re.sub("fkn","fucking",text)
	text=re.sub("f\*\*\*ing","fucking",text)
	text=re.sub("somthng","something",text)
	text=re.sub("s\*\*t","shit",text)
	text=re.sub("&amp;","and",text)
	text=text.replace("Germanwings","german  wings")
	text=text.replace("kouachi","X")
	text=text.replace("sydneysiders","Sydney siders")
	text=text.replace("airtravel","air travel")
	str_text = re.sub("[^a-zA-Z]", " ", text)
	words = word_tokenize(str_text.lower())
	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if w not in stops]
	return words
		
def get_word_embeddings(tweets_train,tweets_test,remove_stopwords=False):
	with open('vocab2vecpy2.pkl','rb') as r:
		model=pickle.load(r)
	avg=False
	w2v_train=numpy.zeros(300)
	w2v_test=numpy.zeros(300)
	temp=numpy.zeros(300)
	for tweet in tweets_train:
		words_list=manual_tokenize(tweet)
		for w in range(len(words_list)):
			if words_list[w] in model:
				temp += model[words_list[w]]
		if avg and len(words_list) != 0:
			final_w2v = temp/len(words_list)
		else:
			final_w2v = temp
		w2v_train=numpy.vstack((w2v_train,final_w2v))
	
	
	for tweet in tweets_test:
		words_list=manual_tokenize(tweet)
		for w in range(len(words_list)):
			if words_list[w] in model:
				temp += model[words_list[w]]
		if avg and len(words_list) != 0:
			final_w2v = temp/len(words_list)
		else:
			final_w2v = temp
		w2v_test=numpy.vstack((w2v_test,final_w2v))
	
	print w2v_train.shape
	w2v_train=numpy.delete(w2v_train,(0),axis=0)
	w2v_test=numpy.delete(w2v_test,(0),axis=0)
	return w2v_train,w2v_test
	
def get_tfidf_features(tweets_train,tweets_test):
	
	#TFIDF VECTORIZER
	vectorizer=TfidfVectorizer(sublinear_tf=True,analyzer="word",stop_words="english",ngram_range=(1,3),max_df=0.65,max_features=2000)
	features_train_tfidf=vectorizer.fit_transform(tweets_train)
	with open('Saved_Model/sdqc_tfidf_vectorizer.pkl','wb') as w:
		pickle.dump(vectorizer,w)
	vectorizer=pickle.load(open('Saved_Model/sdqc_tfidf_vectorizer.pkl','rb'))
	features_test_tfidf=vectorizer.transform(tweets_test)
	#This will create a scipy.sparse.csr.csr_matrix which should be converted to an array
	features_train_tfidf=features_train_tfidf.toarray()
	features_test_tfidf=features_test_tfidf.toarray()
	return features_train_tfidf,features_test_tfidf

def get_main_features(tweets_train,tweets_test,feature_set=['avgw2']):
	if 'avgw2v' in feature_set:
		return get_word_embeddings(tweets_train,tweets_test)
	else:
		return get_tfidf_features(tweets_train,tweets_test)

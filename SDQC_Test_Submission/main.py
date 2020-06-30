import json
import SDQC_Preprocessing
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
import numpy
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
import time
import scikitplot as skplt
import matplotlib.pyplot as plt

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
	blob=TextBlob(text)
	if type=='polarity':
		return blob.sentiment.polarity
	if type=='sub':
		return blob.sentiment.subjectivity
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
		f.append(sentiment_analysis(text,'polarity'))
		f.append(sentiment_analysis(text,'sub'))
		feat.append(f)
	arr=numpy.array(feat)
	print "Extra",arr.shape
	return arr
"""trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))"""
def main():
	
	with open('labelled.json','r') as d:
		label=json.load(d)
	ob=SDQC_Preprocessing.Preprocessing()
	lab=[]
	lab=ob.label_tweets(label)
	with open('data.json','r') as d:
		data=json.load(d)
	
	final=ob.strip_tags_urls(lab,data)
	for key,value in final.items():
		if 'tweet' not in final[key]:
			del final[key]
	with open('dataset.json','w') as dss:
		json.dump(final,dss,sort_keys=True)

	with open('dataset.pkl','wb') as write_handler:
		pickle.dump(final,write_handler,protocol=pickle.HIGHEST_PROTOCOL)
                            
            
	## Open the dataset from the pickle file
	
	with open('dataset.pkl','rb') as read_handler:
		final=pickle.load(read_handler)


	tweet_data=[]
	labels=[]
	for key,value in final.items():
		tweet=final[key]['tweet']
		tweet=tweet.encode('ascii','ignore')
		tweet_data.append(tweet)
		label=str(final[key]['label'])
		if label=="support":
			labels.append(0)
		elif label=="deny":
			labels.append(1)
		elif label=="comment":
			labels.append(2)
		elif label=="query":
			labels.append(3)
	print "Total tweets are : ", len(tweet_data)
	
	tweets_train, tweets_test, labels_train, labels_test = train_test_split(tweet_data,labels, test_size=0.33)
	print "Dataset successfully Preprocessed"
	
	"""
	vectorizer=TfidfVectorizer(sublinear_tf=True,analyzer="word",stop_words="english",ngram_range=(1,3),max_df=0.65,max_features=1000)
	features_train_transformed=vectorizer.fit_transform(tweets_train)
	features_test_transformed=vectorizer.transform(tweets_test)
	print "Features successfully extracted"
	print type(features_train_transformed)"""
	labels_train=numpy.array(labels_train)
	labels_test=numpy.array(labels_test)
	"""
	#features_train_transformed=features_train_transformed.toarray()
	#features_test_transformed=features_test_transformed.toarray()
	selector=SelectPercentile(f_classif,percentile=70)
	selector.fit(features_train_transformed,labels_train)
	features_train_transformed=selector.transform(features_train_transformed).toarray()
	features_test_transformed=selector.transform(features_test_transformed).toarray()
	print "Features Selection is done successfully "
	
	
	features_train_transformed.dump("features_train.pkl")
	features_test_transformed.dump("features_test.pkl")
	features_train_transformed=numpy.load("features_train.pkl")
	features_test_transformed=numpy.load("features_test.pkl")
	"""

	"""
	#extra_features_train=get_extra_features(tweets_train)
	#extra_features_test=get_extra_features(tweets_test)
	start_time=time.time()
	features_train_transformed=get_extra_features(tweets_train)
	print ("--- Time taken to extract features from the training data is %s seconds " % (time.time()-start_time))
	features_test_transformed=get_extra_features(tweets_test)
	#print (features_train_transformed.shape,extra_features_train.shape)
	#print (features_test_transformed.shape,extra_features_test.shape)
	#features_train_transformed=numpy.concatenate((features_train_transformed,extra_features_train),axis=1)
	#features_test_transformed=numpy.concatenate((features_test_transformed,extra_features_test),axis=1)
	#print len(vectorizer.get_feature_names())
	print (features_train_transformed.shape)
	print (features_test_transformed.shape)
	#Save the 2D Vectors
	features_train_transformed.dump("features_train.pkl")
	features_test_transformed.dump("features_test.pkl")
	labels_train.dump("labels_train.pkl")
	labels_test.dump("labels_test.pkl")
	"""
	temp1="Train\extra_"
	temp2="Test\extra_"
	features_train_transformed=numpy.load(temp1+"features_train.pkl")
	print "Train Data Reloaded"
	features_test_transformed=numpy.load(temp2+"features_test.pkl")
	print "Test Data Reloaded"
	labels_train=numpy.load("Train\labels_train.pkl")
	print "Label Train Data Reloaded"
	labels_test=numpy.load("Test\labels_test.pkl")
	print "Label Test Data Reloaded"
	l_tr=[]
	for l in labels_train:
		l_tr.extend(l)
	l_te=[]
	for l in labels_test:
		l_te.extend(l)
	l_tr=numpy.array(l_tr)
	l_te=numpy.array(l_te)
	features_train_transformed=numpy.concatenate((features_train_transformed,numpy.load("Train\main_features_train.pkl")),axis=1)
	features_test_transformed=numpy.concatenate((features_test_transformed,numpy.load("Test\main_features_test.pkl")),axis=1)
	"""
	ftrl=features_train_transformed.tolist()
	ftel=features_test_transformed.tolist()
	ltr=labels_train.tolist()
	lte=labels_test.tolist()
	ftrl1=[]
	ftrl1.append(ftrl)
	ftel1=[]
	ftel1.append(ftel)
	ltr1=[]
	ltr1.append(ltr)
	lte1=[]
	lte1.append(lte)
	features_train_transformed=numpy.array(ftrl1)
	features_test_transformed=numpy.array(ftel1)
	labels_train=numpy.array(ltr1)
	labels_test=numpy.array(lte1)
	#print len(vectorizer.get_feature_names())
	"""
	"""
	print (features_train_transformed.shape)
	print (features_train_transformed[0].shape[1])
	print (features_test_transformed.shape)
	print (labels_train.shape)
	print (features_train_transformed)
	print (labels_train)
	"""
	#from sklearn.naive_bayes import GaussianNB
	#model=GaussianNB()
	#from sklearn import svm
	#model=svm.SVC()
	#from xgboost import XGBClassifier
	#model=XGBClassifier()
	#model=ChainCRF()
	#ssvm=FrankWolfeSSVM(model=model,C=.1,max_iter=11)
	from sklearn.linear_model import LogisticRegression
	#print type(features_train_transformed[0]),len(labels_train)
	print "Training the Model"
	start_time=time.time()
	model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', max_iter=550)
	model.fit(features_train_transformed,l_tr)
	#ssvm.fit(features_train_transformed,labels_train)
	print ("--- Time taken to train the classifier is %s seconds " % (time.time()-start_time))
	print "Model Successfully trained"
	predictions=model.predict(features_test_transformed)
	
	print accuracy_score(l_te,predictions)
	print(classification_report(l_te, predictions,target_names=["support","deny","query","comment"]))
	#print ("Test score with chain CRF : %f" % ssvm.score(features_test_transformed,labels_test))
	skplt.metrics.plot_confusion_matrix(l_te,predictions)
	plt.show()
	
"""	
# Create a mapping of labels to indices
	labels = {"support": 0, "deny": 1, "query":2 ,"comment":3}
	predictions = np.array([labels[tag] for row in y_pred for tag in row])
	truths = np.array([labels[tag] for row in y_test for tag in row])
	print "Truth"
	print truths
	print "Predictions"
	print predictions
	s=0
	d=0
	c=0
	q=0
	for i in range(0,len(truths)):
		if truths[i]==0:
			s=s+1
		if truths[i]==1:
			d=d+1
		if truths[i]==2:
			q=q+1
		if truths[i]==3:
			c=c+1
	
	print s,d,q,c
	s=0
	d=0
	c=0
	q=0
	for i in range(0,len(predictions)):
		if predictions[i]==0:
			s=s+1
		if predictions[i]==1:
			d=d+1
		if predictions[i]==2:
			q=q+1
		if predictions[i]==3:
			c=c+1
	
	print s,d,q,c
"""		

if __name__=="__main__":
	main()
	
	

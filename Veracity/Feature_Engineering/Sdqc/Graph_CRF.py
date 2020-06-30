import pickle
import json

import nltk
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.metrics import accuracy_score,confusion_matrix
from scipy.sparse import csr_matrix
import numpy
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM,SubgradientSSVM
from pystruct.utils import SaveLogger
import time

import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
from gensim.models import Word2Vec,KeyedVectors

import pprint as pp
import re

from Sdqc_feature_extraction import *
import scikitplot as skplt
import matplotlib.pyplot as plt

def get_features_and_edges(features,edges):
	features_list=[]
	features_index=0
	for i in range(0,len(edges)):
		f_i=numpy.zeros(features[0].shape)
		for j in range(0,edges[i].shape[1]+1):	# Edges size is 1-number of feature_set
			#f_i=numpy.insert(f_i,j,features[features_index])
			f_i=numpy.vstack((f_i,features[features_index]))
			features_index+=1
		f_i=numpy.delete(f_i,(0),axis=0)
		x=(f_i,edges[i])
		features_list.append(x)

	return features_list


def array_to_list(labels):
	labels_list=[]
	#print count
	for i in range(0,labels.shape[0]):
		labels_list.append(labels[i])
		#labels_1D=numpy.hstack((labels_1D,labels[i]))
	return labels_list

def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.metrics.plot_confusion_matrix(yte, ypred)
    plt.show()

def feature_scaling(features_train_transformed,features_test_transformed):
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        features_train_scaled=scaler.fit_transform(features_train_transformed)
        features_test_scaled=scaler.fit_transform(features_test_transformed)
        return features_train_scaled,features_test_scaled

def main():
	
	tweets_data_train=[]
	with open('Train\dataset_train.pkl','rb') as r:
		tweets_set=pickle.load(r)
	
	for i in range(0,len(tweets_set)):
		for j in range(0,len(tweets_set[i])):
			t=tweets_set[i][j][1].encode('ascii','ignore')
			tweets_data_train.append(t)
	"""
	features_train_transformed=get_extra_features(tweets_data_train)
	print (features_train_transformed.shape)
	features_train_transformed.dump("Train\extra_features_train.pkl")
	
	extra_features_train=numpy.load("Train\extra_features_train.pkl")
	print ("EXTRA FEATURES FOR TRAIN DATA IS SUCCESSFULLY EXTRACTED")
	"""

	
	tweets_data_test=[]
	with open('Test\dataset_test.pkl','rb') as r:
		tweets_set=pickle.load(r)
	for i in range(0,len(tweets_set)):
		for j in range(0,len(tweets_set[i])):
			t=tweets_set[i][j][1].encode('ascii','ignore')
			tweets_data_test.append(t)
	"""
	features_test_transformed=get_extra_features(tweets_data_test)
	features_test_transformed.dump("Test\extra_features_test.pkl")
	
	extra_features_test=numpy.load("Test\extra_features_test.pkl")
	print ("EXTRA FEATURES FOR TEST DATA IS SUCCESSFULLY EXTRACTED")
	"""

	
	#TFIDF VECTORIZER
	features_train_tfidf,features_test_tfidf=get_main_features(tweets_data_train,tweets_data_test)
	features_train_tfidf.dump("Train\main_features_train.pkl")
	features_test_tfidf.dump("Test\main_features_test.pkl")

	

	with open('Train\edges_train.pkl','rb') as e:
		edges_train=pickle.load(e)
	with open('Train\labels_train.pkl','rb') as l:
		labels_tr=pickle.load(l)
	with open('Test\edges_test.pkl','rb') as e:
		edges_test=pickle.load(e)
	with open('Test\labels_test.pkl','rb') as l:
		labels_te=pickle.load(l)
	
	
	#edges=numpy.array(edges)
	labels_tr=numpy.array(labels_tr)
	labels_te=numpy.array(labels_te)
	#labels_1D=numpy.zeros(1)

	labels_train=array_to_list(labels_tr)
	labels_test=array_to_list(labels_te)
	labels_test=numpy.array(labels_test)
	

	#labels_1D=numpy.delete(labels_1D,(0),0)
	"""

	selector=SelectPercentile(f_classif,percentile=70)
	selector.fit(features_train_tfidf,labels_1D)
	features_train_transformed=selector.transform(features_train_tfidf).toarray()
	features_test_transformed=selector.transform(features_test_tfidf).toarray()
	print "Features Selection is done successfully """

	

	#print features_test_tfidf.shape,extra_features_test.shape

	features_train_transformed=numpy.concatenate((features_train_tfidf,extra_features_train),axis=1)
	features_test_transformed=numpy.concatenate((features_test_tfidf,extra_features_test),axis=1)
	print ("TFIDF FEATURES ARE SUCCESSFULLY CREATED")
	#print features_train_transformed.shape,features_train_tfidf.shape
	#print features_test_transformed.shape,features_test_tfidf.shape
	
	#print len(edges)#,edges[0].shape #EDGES ARE LIST OF 2D ARRAYS of shape(2,length of one tree structure)
	#print labels.shape,labels[0].shape,labels[0]
	#print features_train_transformed[0].shape   #NUMBER OF FEATURES
	
	#features_train_transformed,features_test_transformed=feature_scaling(features_train_transformed,features_test_transformed)

	features_train=get_features_and_edges(features_train_transformed,edges_train)
	features_test=get_features_and_edges(features_test_transformed,edges_test)
	
	labels_train=numpy.array(labels_train)
	#print labels_train.shape
	model_name="GraphCRF_model"
	model=GraphCRF(directed=True)
	ssvm=FrankWolfeSSVM(model=model,C=1.0,max_iter=100,logger=SaveLogger(model_name + ".pickle", save_every=100))
	start_time=time.time()
	final_model=ssvm.fit(features_train,labels_train)
	print ("--- Time taken to train the classifier is %s seconds " % (time.time()-start_time))
	print ("YAAY ! A GRAPH CRF MODEL IS SUCCESSFULLY CREATED AND TRAINED") 
	
	print ("Charliehedbo event is the Test Data")
	pickle.dump(final_model,open('sdqc_final_model.pkl','wb'))
	ssvm=pickle.load(open('sdqc_final_model.pkl','rb'))
	#ssvm = SaveLogger(model_name+".pickle").load()
	X_test=[]
	y_test=[]
	for i in range(0,len(features_test)):
		if features_test[i][0].shape[0]>=3:
			X_test.append(features_test[i])
			y_test.append(labels_test[i])
	#print X_test
	
	#print ("Accuracy score with Graph CRF : %f" % ssvm.score(X_test,y_test))

	predictions=ssvm.predict(X_test)
	#PREDICTIONS AND y_TEST ARE LIST OF ARRAYS
	#pp.pprint(y_test)
	#pp.pprint(predictions)
	true=numpy.zeros(1)
	prediction=numpy.zeros(1)
	for i in range(0,len(predictions)):
		true=numpy.hstack((true,y_test[i]))
		prediction=numpy.hstack((prediction,predictions[i]))

	true=numpy.delete(true,(0),axis=0)
	prediction=numpy.delete(prediction,(0),axis=0)
	#print "TOTAL",true.shape[0]
	print (accuracy_score(true,prediction))
	with open('SDQC_Result.pkl','wb') as w:
		pickle.dump(prediction,w)
	print(classification_report(true, prediction,target_names=["support","deny","query","comment"]))
	print (confusion_matrix(true,prediction,labels=[0,1,2,3]))
	plot_cmat(true, prediction)

if __name__=="__main__":
	main()
     
  

import pickle
import json

import nltk
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile,f_classif
import numpy
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM,SubgradientSSVM
from pystruct.utils import SaveLogger
import time

import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
from gensim.models import Word2Vec,KeyedVectors

import re

from Sdqc_feature_extraction import *

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


def create_edges(test_data):
        arr1=[]
        arr2=[]
        final_edges=[]
        for k in range(len(test_data)):
                if k>=1:
                        arr1.append(k)
                        arr2.append(test_data[k][2])
                        arr1=numpy.array(arr1)
        arr2=numpy.array(arr2)
        two_arr=numpy.vstack((arr1,arr2))
        final_edges.append(two_arr)
        return final_edges

def feature_scaling(features_train_transformed,features_test_transformed):
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        features_train_scaled=scaler.fit_transform(features_train_transformed)
        features_test_scaled=scaler.fit_transform(features_test_transformed)
        return features_train_scaled,features_test_scaled

def extract_sdqc_feature(conv_id):
	
	with open('all_triplets.pkl','rb') as r:
		final_list=pickle.load(r)
	sdqc_features={}
	tweets_data_train=[]
	ids=[]
	for i in range(0,len(final_list)):
		ids_inner_list=[]
		for j in range(0,len(final_list[i])):
			ids_inner_list.append(final_list[i][j][0])
			t=final_list[i][j][1].encode('ascii','ignore')
			tweets_data_train.append(t)
		ids.append(ids_inner_list)
	
	test_data=[]
	for i in range(len(ids)):
		if ids[i][0][0]==conv_id:
			test_data=ids[i]
			break
	
	tweets_data_test=[]
	for i in range(len(test_data)):
		t=test_data[i][1].encode('ascii','ignore')
		tweets_data_test.append(t)

	if not tweets_data_test:
                return sdqc_features
	features_test_transformed=get_extra_features(tweets_data_test)
	features_test_transformed.dump("Test\extra_features_test.pkl")
	
	extra_features_test=numpy.load("Test\extra_features_test.pkl")
	#print ("EXTRA FEATURES FOR TEST DATA IS SUCCESSFULLY EXTRACTED")

	
	#TFIDF VECTORIZER
	vectorizer=pickle.load(open('sdqc_tfidf_vectorizer.pkl','rb'))
	print (tweets_data_test)
	features_test_tfidf=vectorizer.transform(tweets_data_test)
	#This will create a scipy.sparse.csr.csr_matrix which should be converted to an array
	features_test_tfidf=features_test_tfidf.toarray()

	edges_test=create_edges(test_data)
	
	features_test_transformed=numpy.concatenate((features_test_tfidf,extra_features_test),axis=1)
	#print ("TFIDF FEATURES ARE SUCCESSFULLY CREATED")
	
	features_test=get_features_and_edges(features_test_transformed,edges_test)
	
	ssvm=pickle.load(open('sdqc_final_model.pkl','rb'))
	X_test=[]
	for i in range(0,len(features_test)):
		if features_test[i][0].shape[0]>=3:
			X_test.append(features_test[i])

	predictions=ssvm.predict(X_test)
	for i in range(len(test_data)):
		sdqc_features[test_data[i][0]]=predictions[0][i]
	
	return sdqc_features

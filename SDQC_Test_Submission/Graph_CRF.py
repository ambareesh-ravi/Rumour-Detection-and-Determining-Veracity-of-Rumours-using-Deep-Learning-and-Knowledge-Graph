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
from Save_Output import save_output
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
from gensim.models import Word2Vec,KeyedVectors

import pprint as pp
import re

from Sdqc_feature_extraction import *
#import scikitplot as skplt
#import matplotlib.pyplot as plt

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
	tweets_data_test=[]
	with open("Test\\tweets_separated.pkl","r") as r:
		tweets_set=pickle.load(r)
	for i in range(0,len(tweets_set)):
		for j in range(0,len(tweets_set[i])):
			t=tweets_set[i][j].encode('ascii','ignore')
			tweets_data_test.append(t)
	#features_test_transformed=get_extra_features(tweets_data_test)
	#features_test_transformed.dump("Test\extra_features_test.pkl")
	extra_features_test=numpy.load("Test\extra_features_test.pkl")
	print "EXTRA FEATURES FOR TEST DATA IS SUCCESSFULLY EXTRACTED"

	#TweetTFIDFfeatures
	features_test_tfidf=get_main_features(tweets_data_test)
	
	
	with open("Test\edges_separated.pkl","r") as e:
		edges_test=pickle.load(e)
	print (len(edges_test))
	#edges=numpy.array(edges)
	
	
	features_test_transformed=numpy.concatenate((features_test_tfidf,extra_features_test),axis=1)
	print "TFIDF FEATURES ARE SUCCESSFULLY CREATED"
	print features_test_transformed.shape,features_test_tfidf.shape
	
	features_test=get_features_and_edges(features_test_transformed,edges_test)
	
	#pickle.dump(final_model,open('sdqc_final_model.pkl','wb'))
	ssvm=pickle.load(open('sdqc_final_model.pkl','rb'))
	X_test=[]
	
	for i in range(0,len(features_test)):
		#if features_test[i][0].shape[0]>=3:
		X_test.append(features_test[i])
			
	
	predictions=ssvm.predict(X_test)
	
	
	final_pred_list = []
	final_id_list = []
	
	with open("Test\ids_separated.pkl","r") as id:
		ids = pickle.load(id)
		
	for i in range(0,len(predictions)):
		for j in range(0,len(predictions[i])):
			final_pred_list.append(predictions[i][j])
			final_id_list.append(ids[i][j])
			
	dummy = []
	
	save_output(final_id_list,final_pred_list,dummy,dummy,dummy)	
		
		
		
	"""true=numpy.zeros(1)
	prediction=numpy.zeros(1)
	for i in range(0,len(predictions)):
		true=numpy.hstack((true,y_test[i]))
		prediction=numpy.hstack((prediction,predictions[i]))
	true=numpy.delete(true,(0),axis=0)
	prediction=numpy.delete(prediction,(0),axis=0)
	
	print "TOTAL",true.shape[0]
	print accuracy_score(true,prediction)
	print(classification_report(true, prediction,target_names=["support","deny","query","comment"]))
	print confusion_matrix(true,prediction,labels=[0,1,2,3])
	#plot_cmat(true, prediction)"""

if __name__=="__main__":
	main()
     
  

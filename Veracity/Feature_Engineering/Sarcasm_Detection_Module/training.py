# -*- coding: utf-8 -*-

import nltk
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
import pickle
import feature_extraction


pos_data=np.load('sarcasm_array.npy')
neg_data=np.load('non_sarcasm_array.npy')

#print 'Number of  sarcastic tweets :', len(pos_data)
#print 'Number of  non-sarcastic tweets :', len(neg_data)

cls_set = ['Non-Sarcastic','Sarcastic']
features = [] 

for tweet in pos_data:
    features.append((feature_extraction.get_features(tweet),cls_set[1]))
    
for tweet in neg_data:
    features.append((feature_extraction.get_features(tweet),cls_set[0]))

features=np.array(features)
targets=(features[0::,1]=='Sarcastic').astype(int)

vec = DictVectorizer()
features_vec = vec.fit_transform(features[0::,0]) 
pickle.dump(vec, open("features_dict.pkl",'wb'))

order=shuffle(range(len(features)))
targets=targets[order]
features_vec=features_vec[order,0::]


size = int(len(features) * .3) 

trainvec = features_vec[size:,0::]
train_targets = targets[size:]
testvec = features_vec[:size,0::]
test_targets = targets[:size]


pos_p=(train_targets==1)
neg_p=(train_targets==0)
ratio = np.sum(neg_p.astype(float))/np.sum(pos_p.astype(float))
new_trainvec=trainvec
new_train_targets=train_targets
for j in range(int(ratio-1.0)):
    new_trainvec=sp.sparse.vstack([new_trainvec,trainvec[pos_p,0::]])
    new_train_targets=np.concatenate((new_train_targets,train_targets[pos_p]))    



model = SVC(C=0.1,kernel='linear')
model.fit(new_trainvec,new_train_targets)
pickle.dump(model,open("sarcasm_model.pkl",'wb'))

output = model.predict(testvec)
clfreport = classification_report(test_targets, output, target_names=cls_set)
print (clfreport)
print (accuracy_score(test_targets, output))



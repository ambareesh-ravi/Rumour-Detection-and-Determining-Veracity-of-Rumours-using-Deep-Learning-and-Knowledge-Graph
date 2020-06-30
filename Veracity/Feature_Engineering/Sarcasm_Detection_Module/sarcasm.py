
import numpy as np
import pickle
import os
import feature_extraction
#file1 = open('Sarcasm_Detection_Module/features_dict.pkl', 'rb')
#file2 = open('Sarcasm_Detection_Module/sarcasm_model.pkl','rb',encoding="utf-8")

#vec = pickle.load(file1)
#classifier = pickle.load(file2)
with open('Sarcasm_Detection_Module/features_dict.pkl', 'rb') as f:
    feat_dict=pickle.load(f)
with open('Sarcasm_Detection_Module/sarcasm_model.pkl','rb') as m:
    model=pickle.load(m)

#file1.close()
#file2.close()
def getSarcasmScore(sentence):
    sentence = sentence.encode('ascii', 'ignore')
    features = feature_extraction.getallfeatureset(sentence)
    
    features_vec = feat_dict.transform(features)
    score = model.decision_function(features_vec)[0]
    percentage = int(round(2.0*(1.0/(1.0+np.exp(-score))-0.5)*100.0))
    
    return percentage

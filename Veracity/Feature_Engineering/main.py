
from extract_features_veracity import extract_tree_features
#import help_prep_functions
import numpy as np
import os,json
from keras.preprocessing.sequence import pad_sequences
import time

label_veracity={"true":0,"false":1,"unverified":2}

def convert_to_array_of_branches(tree_feature_dict, conversation):
    tree_features_array = []

    branches = conversation['branches']

    for branch in branches:
        branch_arr = np.zeros(406)
        for twid in branch:
            if twid in tree_feature_dict.keys():
                tweet_arr = tree_feature_dict[twid]
                #print (twid,tweet_arr.shape)
                branch_arr=np.vstack((branch_arr,tweet_arr))
        if branch_arr.shape[0]>=1:
            #branch_rep = np.asarray(branch_rep)
            branch_arr=np.delete(branch_arr,(0),axis=0)
            tree_features_array.append(branch_arr)
     
    return tree_features_array

def main():
    
    path = 'Saved_dataset_LOL'
    data = {}
    with open('dataset1.json','r') as r:
        data=json.load(r)

    start_time=time.time()
    for type in data.keys():
        
        features = []
        labels = []
        ids = []
        for conversation in data[type]:
            
            tree_feature_dict = extract_tree_features(conversation)
            tree_features_array = convert_to_array_of_branches(tree_feature_dict, conversation)
            features.extend(tree_features_array)
            for i in range(len(tree_features_array)):
                labels.append(label_veracity[conversation['veracity']])
                ids.append(conversation['id'])
        
        if features!=[]:

            features = pad_sequences(features, maxlen=None,dtype='float32',padding='post',truncating='post', value=0.)
            labels = np.asarray(labels)
            path_type = os.path.join(path, type)
            if not os.path.exists(path_type):
                os.makedirs(path_type)
            np.save(os.path.join(path_type, 'feat_array'), features)
            np.save(os.path.join(path_type, 'labels'), labels)
            np.save(os.path.join(path_type, 'ids'), ids)

    print (time.time()-start_time)



if __name__ == '__main__':
    main()

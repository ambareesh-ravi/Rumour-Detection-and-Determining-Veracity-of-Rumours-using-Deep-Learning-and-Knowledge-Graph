import json
import pprint as pp
from collections import OrderedDict
import pickle
import numpy as np
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

ans=[]
source_i=[]
glob_variable = 0
label_files=['1-ebola-essien_labels.pkl','2-putinmissing_labels.pkl','3-germanwings-crash_labels.pkl','4-prince-toronto_labels.pkl','5-ferguson_labels.pkl','6-ottawashooting_labels.pkl','7-charliehebdo_labels.pkl','8-sydneysiege_labels.pkl']
labels_dict={"support":0,"deny":1,"query":2,"comment":3}
file_list=["1-ebola-essien.pkl","2-putinmissing.pkl","3-germanwings-crash.pkl","4-prince-toronto.pkl","5-ferguson.pkl","6-ottawashooting.pkl","7-charliehebdo.pkl","8-sydneysiege.pkl"]
#file_list=["3-germanwings-crash.pkl"]
def label_tweets(filename,lab,label_file):

    #0- support     1-deny   2-query     3-comment
    with open("Events\\"+filename,'rb') as r:
        idd=pickle.load(r)
    labels=[]
    t=0
    l=0
    subA=lab["subtaska"]
    for i in range(0,len(idd)):
        labels_inner=[]
        for j in range(0,len(idd[i])):
            t+=1
            tweet_id=idd[i][j][0]
            #print tweet_id
            flag=0
            for key,value in subA.items():
                if key==tweet_id:
                    lab_int=labels_dict[subA[key]]
                    labels_inner.append(lab_int)
                    l+=1
                    flag=1
            if flag==0:
                print tweet_id
        if not labels_inner:
            print idd[i]
        labels_inner_arr=np.array(labels_inner)
        labels.append(labels_inner_arr)
    final_labels=np.array(labels)
    final_labels.dump("Labels\\"+label_file)
    final_labels=np.load("Labels\\"+label_file)
    #pp.pprint(final_labels)
    print t,l
    if t==l:
        print "All tweets are labelled !!!!"

    #lab["subtaska"]["580355750775656448"]="support"     #Support(0) is present as a label in labelled.json but not in train-key.json
	    
#struct = data["ebola-essien"]["521346721226711040"]["structure"]

#Label the tweets
with open('train-key.json','r') as d:
    label=json.load(d)

for i in range(0,len(file_list)):
    label_tweets(file_list[i],label,label_files[i])
#label_tweets('7-charliehebdo.pkl',label)

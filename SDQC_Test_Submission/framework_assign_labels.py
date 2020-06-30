import json
import pprint as pp
from collections import OrderedDict
import pickle
import numpy as np
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

with open("data - Copy.json",'r') as read_file:
	data = json.load(read_file, object_pairs_hook=OrderedDict)

ans=[]
source_i=[]
glob_variable = 0
eventt="charliehebdo"
label_file="7-charliehebdo_labels.pkl"
def label_tweets(filename,lab):

    #0- support     1-deny   2-query     3-comment
    with open("Events\\"+filename,'rb') as r:
        idd=pickle.load(r)
    labels=[]
    t=0
    l=0
    for i in range(0,len(idd)):
        labels_inner=[]
        for j in range(0,len(idd[i])):
            t+=1
            tweet_id=idd[i][j][0]
            #print tweet_id
            for key,value in lab.items():
                if lab[key]['event']==eventt and lab[key]['tweetid']==tweet_id:
                    #print "Hello"
                    lab_int=0
                    if 'responsetype-vs-source' in lab[key]:
                        stance=lab[key]['responsetype-vs-source']
                        del lab[key]['responsetype-vs-source']
                        if stance == "agreed":
                            lab_int=0
                        elif stance == "disagreed":
                            lab_int=1
                        elif stance == "appeal-for-more-information":
                            lab_int=2
                        elif stance == "comment":
                            lab_int=3
                        labels_inner.append(lab_int)
                        if 'responsetype-vs-previous' in lab[key]:
                            del lab[key]['responsetype-vs-previous']
                    elif 'responsetype-vs-previous' in lab[key]:
                        stance=lab[key]['responsetype-vs-previous']
                        del lab[key]['responsetype-vs-previous']
                        if stance == "agreed":
                            lab_int=0
                        elif stance == "disagreed":
                            lab_int=1
                        elif stance == "appeal-for-more-information":
                            lab_int=2
                        elif stance == "comment":
                            lab_int=3
                        labels_inner.append(lab_int)
                    elif 'support' in lab[key]:
                        stance=lab[key]['support']
                        del lab[key]['support']
                        if stance == "supporting":
                            lab_int=0
                        elif stance == "denying":
                            lab_int=1
                        elif stance == "underspecified":
                            lab_int=3
                        labels_inner.append(lab_int)
                    l+=1
        labels_inner_arr=np.array(labels_inner)
        labels.append(labels_inner_arr)
    final_labels=np.array(labels)
    """with open('labels/lol.pkl','wb') as wp:
        pickle.dump(final_labels,wp)
    with open('labels/lol.pkl','rb') as rp:
        final_labels=pickle.load(rp)"""
    final_labels.dump("Labels\\"+label_file)
    final_labels=np.load("Labels\\"+label_file)
    pp.pprint(final_labels)
    print t,l
    if t==l:
        print "All tweets are labelled !!!!"
	    
#struct = data["ebola-essien"]["521346721226711040"]["structure"]

#Label the tweets
with open('labelled.json','r') as d:
    label=json.load(d)
label_tweets('7-charliehebdo.pkl',label)

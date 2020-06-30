import json
import pprint as pp
from collections import OrderedDict
import pickle
import numpy as np
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")


#--------Generates SNO-event.pkl file -------#

with open("data.json",'r') as read_file:
	data = json.load(read_file, object_pairs_hook=OrderedDict)

ans=[]
source_i=[]
glob_variable = 0
tweet_id_list=[]
count=0
eventt='putinmissing'
dest_file='2-putinmissing.pkl'
id1="576321856166346752"
id2="576352246394421249"
def recurse_fun( dict_obj, index ):
	

	#algorithm:
	'''
		we iterate over key and values:

		1. get key og dict_obj and extract tweet of key and append to list as tweet,index
		2. then check if the value is iterable and if value is not []
		3. if value is not [] then we recurse with value, last global value

	'''
	for key,value in dict_obj.items():
		if key in tweet_id_list and (key!=id1 and key!=id2):
			#count=count+1
			source_i.append([key,index])
			glob_variable = len(source_i)-1
			if value:
				recurse_fun(value,glob_variable)
				
#struct = data["ebola-essien"]["521346721226711040"]["structure"]
with open('labelled.json','r') as d:
    lab=json.load(d)
for key,value in lab.items():
  if lab[key]['event']==eventt:
    tweet_id_list.append(lab[key]['tweetid'])
for key,value in data.items():
    if key==eventt:
        for key1,value1 in value.items():
            source_i=[]
            glob_variable = 0
            struct=value[key1]["structure"]
            recurse_fun(struct,0)
            ans.append(source_i)
#pp.pprint(ans)


c=0
for a in ans:
        for b in a:
                c+=1
with open(dest_file,'wb') as w:
    pickle.dump(ans,w)
with open(dest_file,'rb') as r:
    ans=pickle.load(r)
pp.pprint(ans)
pp.pprint(count)
print(c)
"""
#Label the tweets
with open('labelled.json','r') as d:
    label=json.load(d)
label_tweets('putinmissing.pkl',label)
"""

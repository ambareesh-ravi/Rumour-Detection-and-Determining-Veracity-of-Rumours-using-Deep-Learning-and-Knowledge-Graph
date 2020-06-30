import pickle
import numpy

files=['1-tweets.pkl','2-tweets.pkl','3-tweets.pkl','4-tweets.pkl','5-tweets.pkl','6-tweets.pkl','7-tweets.pkl','8-tweets.pkl']  #train
final_list=[]

for i in range(0,len(files)):
	with open('Tweets\\'+files[i],'rb') as r:
		tweets=pickle.load(r)
	for j in range(0,len(tweets)):
		final_list.append(tweets[j])
		
with open('Full/all_triplets.pkl','wb') as w:
	pickle.dump(final_list,w)

with open('Full/all_triplets.pkl','rb') as r:
	final_list=pickle.load(r)
	
print (len(final_list))

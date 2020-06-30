import json
import os

#Total Test Data : 1066 Tweets
def main():
	path_to_folds = 'rumoureval-2019-test-data/twitter-en-test-data'
	folds = sorted(os.listdir(path_to_folds))
	newfolds = [i for i in folds if i[0] != '.']
	folds = newfolds
	dataset = {}
	count=0
	for nfold, fold in enumerate(folds):
		path_to_tweets = os.path.join(path_to_folds, fold)
		source_tweets = sorted(os.listdir(path_to_tweets))
		newfolds = [i for i in source_tweets if i[0] != '.']
		source_tweets = newfolds
		count=count+len(source_tweets)
		dataset[fold]={}
		for source_foldr in source_tweets:
			dataset[fold][source_foldr]={}
			source_foldr_path=os.path.join(path_to_tweets,source_foldr)
			source_tweet=os.listdir(source_foldr_path+'/source-tweet')
			#print (source_tweet_path)
			#print (source_foldr_path+'/source-tweet/'+source_tweet[0])
			with open(source_foldr_path+'/source-tweet/'+source_tweet[0],'r') as r:
				src=json.load(r)
			with open(source_foldr_path+'/structure.json','r') as r:
				structure=json.load(r)
			dataset[fold][source_foldr]['source']=src
			dataset[fold][source_foldr]['structure']=structure
			dataset[fold][source_foldr]['replies']={}
			#print (source_foldr_path+'/replies/')
			replies_list=os.listdir(os.path.join(source_foldr_path,'replies'))
			count=count+len(replies_list)
			for reply in replies_list:
				with open(source_foldr_path+'/replies/'+reply,'r') as r:
					reply_tweet=json.load(r)
				reply_key=os.path.splitext(reply)[0]
				dataset[fold][source_foldr]['replies'][reply_key]=reply_tweet
			
	
	
	with open('test_dataset.json','w') as w:
		json.dump(dataset,w)
	print (count)		
	
	
	
	
	
	
if __name__=="__main__":
	main()

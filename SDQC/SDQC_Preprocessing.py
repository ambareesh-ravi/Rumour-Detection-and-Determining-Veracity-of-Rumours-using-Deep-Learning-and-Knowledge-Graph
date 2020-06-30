import json
import re
#from collections import OrderedDict

"""for key,value in lab.items():
  if 'evidentiality' in data[key]:
    del data[key]['evidentiality']


#with open('try.json','w') as d:
#  json.dump(lab,d,sort_keys=True)
"""
class Preprocessing:
	def label_tweets(self,lab):
		for key,value in lab.items():
			if 'responsetype-vs-source' in lab[key]:
				stance=lab[key]['responsetype-vs-source']
				del lab[key]['responsetype-vs-source']
				if stance == "agreed":
					stance="support"
				elif stance == "disagreed":
					stance="deny"
				elif stance == "appeal-for-more-information":
					stance="query"
				lab[key]['label']=stance
				if 'responsetype-vs-previous' in lab[key]:
					del lab[key]['responsetype-vs-previous']
			elif 'responsetype-vs-previous' in lab[key]:
				stance=lab[key]['responsetype-vs-previous']
				del lab[key]['responsetype-vs-previous']
				if stance == "agreed":
					stance="support"
				elif stance == "disagreed":
					stance="deny"
				elif stance == "appeal-for-more-information":
					stance="query"
				lab[key]['label']=stance
				#if 'responsetype-vs-source' in lab[key]:
				#	del lab[key]['responsetype-vs-source']
					
			
			
			elif 'support' in lab[key]:
				stance=lab[key]['support']
				del lab[key]['support']
				if stance == "supporting":
					stance="support"
				elif stance == "denying":
					stance="deny"
				elif stance == "underspecified":
					stance="comment"
				lab[key]['label']=stance
				
		return lab
	
	
	
	def	strip_tags_urls(self,ds,data):
		for key,value in ds.items():
			idd=ds[key]['tweetid']
			for key1,value1 in data.items():
				if key1==ds[key]['event']:
					for key2,value2 in value1.items(): #value 1 is inside the rumour
						if key2==idd: #if it is a source tweet
							s=value1[idd]['source']
							ds[key]['retweet_count']=s['retweet_count']
							ds[key]['entities']=s['entities']
							text=s['text']
							url_stripped_tweet = re.sub(r"http\S+", "", text)
							hastag_stripped_tweet = re.sub(r"#\S+", "", url_stripped_tweet)
							ds[key]['tweet']=hastag_stripped_tweet
					if 'tweet' not in ds[key]:
						for k,v in value1.items():
							for k1,v1 in v['replies'].items(): #v is inside the key and replies
								if k1==idd:
									s=v['replies'][k1]
									ds[key]['retweet_count']=s['retweet_count']
									ds[key]['entities']=s['entities']
									text=s['text']
									url_stripped_tweet = re.sub(r"http\S+", "", text)
									hastag_stripped_tweet = re.sub(r"#\S+", "", url_stripped_tweet)
									ds[key]['tweet']=hastag_stripped_tweet
									ds['reply_to_id']=s['in_reply_to_status_id_str']
					
					#ds[key]['certainity']=data[key1]['certainity']
					#ds[key]['evidentiality']=data[key1]['evidentiality']
									
		return ds
    
      

#OrderedDict(sorted(lab.items(),key=lambda t: t[0]))
              
          
            
        

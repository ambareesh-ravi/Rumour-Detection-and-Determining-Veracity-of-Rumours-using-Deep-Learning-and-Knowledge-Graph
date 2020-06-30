import json
import pickle
import re
import pprint as pp

with open("test_dataset.json","r") as json_data:
	data = json.load(json_data)
	
with open("edge_output.pkl","r") as r:
      ans = pickle.load(r)

count = 0

def extract_tweet(tweet_id):
	for key,value in data.items():
		for key1,value1 in value.items():
			if key1 == tweet_id: #sourceTweetNa
				return value[key1]["source"]["text"]
			else:
				replies = value[key1]["replies"]
				for key2,value2 in replies.items():
					if key2 == tweet_id: #ReplyTweetNa
						return replies[key2]["text"]

				  
for list_index in range(len(ans)):
	for id in range(len(ans[list_index])):
		v = ans[list_index][id][0]
		w = ans[list_index][id][1]
		tweet = extract_tweet(v)
		tweet = 	tweet.encode('ascii','ignore')
		url_stripped_tweet = re.sub(r"http\S+", "", tweet)
		hastag_stripped_tweet = re.sub("#", "", url_stripped_tweet)
		um_stripped_tweet=re.sub(r"@\S+","",hastag_stripped_tweet)
		final_tweet=re.sub(r"\\\w+","",um_stripped_tweet)
		count+=1
		ans[list_index][id]= [v,final_tweet,w]

with open("test_tweet_text.pkl","w") as tweet_text:
	pickle.dump(ans,tweet_text)
	
print(count)
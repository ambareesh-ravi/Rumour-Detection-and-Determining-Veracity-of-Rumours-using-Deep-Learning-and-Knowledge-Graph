import json
import pickle
import re
import pprint as pp

def extract_tweet(event_dict,tweet_id):
  for key,value in event_dict.items():
    if key==tweet_id: #source tweet
      
      id_dict=event_dict[key]
      return id_dict['source']['text']

    
    else :  #Reply_tweet
      
      replies_dict=event_dict[key]['replies']
      for key1,value1 in replies_dict.items():
        if key1==tweet_id:
          return replies_dict[key1]['text']



if __name__=="__main__":

  with open('Data\data.json','r') as d:
    data=json.load(d)
  event=["ebola-essien","putinmissing","germanwings-crash","prince-toronto","ferguson","ottawashooting","charliehebdo","sydneysiege"]
  events='Events\\'
  tweets='Tweets\\'
  src_file=["1-ebola-essien.pkl","2-putinmissing.pkl","3-germanwings-crash.pkl","4-prince-toronto.pkl","5-ferguson.pkl","6-ottawashooting.pkl","7-charliehebdo.pkl","8-sydneysiege.pkl"]
  dest_file=["1-tweets.pkl","2-tweets.pkl","3-tweets.pkl","4-tweets.pkl","5-tweets.pkl","6-tweets.pkl","7-tweets.pkl","8-tweets.pkl"]
  flag=1
  for i in range(0,len(src_file)):
    with open(events+src_file[i],'rb') as r:
      ans=pickle.load(r)
      l1=[]
      count=0
      for a in ans:
        l2=[]
        for b in a:
          tweet=extract_tweet(data[event[i]],b[0])
          if not tweet:
            print b[0]
          else:
            tweet=tweet.encode('ascii','ignore')
            url_stripped_tweet = re.sub(r"http\S+", "", tweet)
            #hastag_stripped_tweet = re.sub(r"#\S+", "", url_stripped_tweet)
            hastag_stripped_tweet = re.sub("#", "", url_stripped_tweet)
            um_stripped_tweet=re.sub(r"@\S+","",hastag_stripped_tweet)
            final_tweet=re.sub(r"\\\w+","",um_stripped_tweet)
            count+=1
            l2.append([b[0],final_tweet,b[1]])
        l1.append(l2)
      with open(tweets+dest_file[i],'wb') as w:
        pickle.dump(l1,w)
      with open(tweets+dest_file[i],'rb') as r:
        l1=pickle.load(r)
    #pp.pprint (l1)
      print count,len(l1)
    #tweet_id="521360486387175424
    #print extract_tweet(data[event],tweet_id)

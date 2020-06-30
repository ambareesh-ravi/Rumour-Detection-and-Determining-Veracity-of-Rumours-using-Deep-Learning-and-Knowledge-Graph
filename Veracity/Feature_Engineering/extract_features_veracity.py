
import numpy as np

import nltk
import re,pickle
from word2vec import sumw2v
from textblob import TextBlob
from Lexicons import false_antonyms,false_synonyms,negation_words,SpeechAct
from extract_sdqc import extract_sdqc_feature
with open('vocab2vec.pkl','rb') as r:
  d=pickle.load(r)
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}
wh_words = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why','how']
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

def content_formatting(text,content):
	if text.find(content)>=0:
		return 1
	else:
		return 0

def count_lexicon(tokens,lexicon_list):
	count=0
	for token in tokens:
		if token in lexicon_list:
			count += 1
	return count

def get_speechAct_vector(text):
	sv=[]
	for key in SpeechAct.keys():
		count=0
		for verb in SpeechAct[key]:
			if verb in text.lower():
				count+=1
		sv.append(count)
	return sv
	
def get_features(tweet,tokens,otherthreadtokens):
	feat=[]
	feat.extend(sumw2v(tweet))
	feat.append(content_formatting(tweet,'?'))
	feat.append(content_formatting(tweet,'!'))
	feat.append(content_formatting(tweet,'.'))
	feat.append(content_formatting(tweet,'#'))
	hasurl=0
	if (content_formatting(tweet,'urlurlurl')>=0) or (content_formatting(tweet,'http') >= 0):
		hasurl = 1
	feat.append(hasurl)
	haspic=0
	if (content_formatting(tweet,'picpicpic') >= 0) or (content_formatting(tweet,'pic.twitter.com') >= 0) or (content_formatting(tweet,'instagr.am') >= 0):
		haspic=1
	feat.append(haspic)
	negation_count= 0
	for negationword in negation_words:
		if negationword in tokens:
			negation_count += 1
	feat.append(negation_count)
	feat.append(len(tweet))	#Chararcter count
	feat.append(len(nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','', tweet.lower()))))	#Word count
	swearwords = []
	with open('data/badwords.txt', 'r') as f:
		for line in f:
			swearwords.append(line.strip().lower())
	hasswearwords = 0
	for token in tokens:
		if token in swearwords:
			hasswearwords += 1
	feat.append(hasswearwords)
	uppers = [l for l in tweet if l.isupper()]
	feat.append(float(len(uppers))/len(tweet))	#Capital Ratio
	feat.append(check_pos_tag(tweet, 'noun'))
	feat.append(check_pos_tag(tweet, 'verb'))
	feat.append(check_pos_tag(tweet, 'adj'))
	feat.append(check_pos_tag(tweet, 'adv'))
	feat.append(check_pos_tag(tweet, 'pron'))
	feat.append(count_lexicon(tokens,false_synonyms)) #tweet
	feat.append(count_lexicon(tokens,false_antonyms))#tweet
	feat.append(count_lexicon(otherthreadtokens,false_synonyms)) #Thread
	feat.append(count_lexicon(otherthreadtokens,false_antonyms)) #Thread
	feat.append(count_lexicon(tokens,wh_words))#tweet
	feat.append(count_lexicon(otherthreadtokens,wh_words)) #Thread
	return feat
def extract_source_features(conversation):
    features=[]

    tw = conversation['source']
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',tw['text'].lower()))

    otherthreadtweets = ''
    for response in conversation['replies']:
        otherthreadtweets += ' ' + response['text']
            
    otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',otherthreadtweets.lower()))

    source_tweet = tw['text']
    features=get_features(source_tweet,tokens,otherthreadtokens)
    features.append(tw['favorite_count'])
    features.append(tw['retweet_count'])
    features.append(tw['user']['followers_count'])
    features.append(tw['user']['listed_count'])
    features.append(tw['user']['statuses_count'])
    features.append(tw['user']['friends_count'])
    features.append(tw['user']['favourites_count'])
    verified={ True:1,False:0}
    features.append(verified[tw['user']['verified']])
    features.extend(get_speechAct_vector(source_tweet))
    features.extend(get_speechAct_vector(otherthreadtweets))
    sdqc_features_dict=extract_sdqc_feature(conversation['source']['id_str'])
    if conversation['source']['id_str'] in sdqc_features_dict.keys():
        features.append(sdqc_features_dict[conversation['source']['id_str']])
    else:
        features.append(3)
    features.append(1)	#Source_tweet Flag
    src_features_arr=np.asarray(features)
    return src_features_arr,sdqc_features_dict


def extract_tree_features(conversation):
    tree_features={}
    source_features_arr,sdqc_features_dict = extract_source_features(conversation)
    
    srctokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',conversation['source']['text'].lower()))
    tree_features[conversation['source']['id_str']] = source_features_arr
    
	#For Replies
    for tw in conversation['replies']:
        features=[]
        tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',tw['text'].lower()))
        otherthreadtweets = ''
        otherthreadtweets += conversation['source']['text'] 
   
            
        for response in conversation['replies']:
            otherthreadtweets += ' ' + response['text']
                
        otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',otherthreadtweets.lower()))
        branches = conversation['branches']
        for branch in branches:
            if tw['id_str'] in branch:
                if branch.index(tw['id_str'])-1 == 0:
                    prevtokens = srctokens
                else:
                    prev_id = branch[branch.index(tw['id_str'])-1]
                    for ptw in conversation['replies']:
                        if ptw['id_str'] == prev_id:
                            prevtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','',ptw['text'].lower()))
                            break
            else:
                prevtokens = []
            break
        reply_tweet = tw['text']
        features=get_features(reply_tweet,tokens,otherthreadtokens)
        features.append(tw['favorite_count'])
        features.append(tw['retweet_count'])
        features.append(tw['user']['followers_count'])
        features.append(tw['user']['listed_count'])
        features.append(tw['user']['statuses_count'])
        features.append(tw['user']['friends_count'])
        features.append(tw['user']['favourites_count'])
        verified={ True:1,False:0}
        features.append(verified[tw['user']['verified']])
        features.extend(get_speechAct_vector(reply_tweet))
        features.extend(get_speechAct_vector(otherthreadtweets))	#Thread
        if tw['id_str'] in sdqc_features_dict.keys():
            features.append(sdqc_features_dict[tw['id_str']])
        else:
            features.append(3)
        features.append(0)  #Not a source
        repl_arr=np.asarray(features)
		
        tree_features[tw['id_str']] = repl_arr	
    return tree_features

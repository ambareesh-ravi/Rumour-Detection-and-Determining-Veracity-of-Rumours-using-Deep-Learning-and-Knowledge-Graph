import pickle
from statistics import mean



def search_google(query):
        from googlesearch import search
        for j in search(query, tld="com", num=10, stop=1, pause=2):
                print (j)



event_tweets=[]
with open('../SDQC/Tweets/5-tweets.pkl','rb') as r:
	tweets=pickle.load(r)

for t in tweets:
	for tweet in t:
		event_tweets.append(tweet[1])

#print (len(event_tweets))

#print (event_tweets)

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(tweets).split() for tweets in event_tweets] 

#print (doc_clean)

for a in doc_clean:
        if not a:
                doc_clean.remove(a)

#print (doc_clean)

import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import gensim
from gensim import corpora


dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

#print (doc_term_matrix)

Lda = gensim.models.ldamodel.LdaModel

ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=100)

topics=ldamodel.print_topics(num_topics=3, num_words=3)
print(topics)
print (type(topics))
print (type(topics[0][1]))
t1=topics[0][1].split('+')

print(t1)
probs=[]
words=[]
for i in range(0,len(topics)):
	t=topics[i][1].split('+')
	for i in range(0,len(t)):
		each_topic=t[i].split('*')
		num=float(each_topic[0])
		probs.append(num)
		if i<=len(t)-2:
			words.append(each_topic[1][1:-2])
		else:
			words.append(each_topic[1][1:-1])

print (probs)
print (words)
search_query=""
mean_weight=mean(probs)
for i in range(0,len(words)):
        if probs[i]>mean_weight:
                search_query=search_query+words[i]+" "

print (search_query)
search_google(search_query)

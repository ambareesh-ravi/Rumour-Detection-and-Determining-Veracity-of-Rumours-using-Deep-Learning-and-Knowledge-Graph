
import numpy as np
import nltk
from nltk.corpus import stopwords
import re,pickle
from copy import deepcopy
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import gensim

with open('vocab2vec.pkl','rb') as r:
  model_WV=pickle.load(r)

def str_to_wordlist(text,remove_stopwords=False):
	text=re.sub("-"," - ",text)
	text=re.sub(r"\.+"," . ",text)
	text=re.sub("/"," or ",text)
	text=re.sub(","," , ",text)
	text=re.sub("\*"," * ",text)
	text=re.sub("'"," ' ",text)
	text=re.sub("="," = ",text)
	text=re.sub("srry","sorry",text)
	text=re.sub("fkn","fucking",text)
	text=re.sub("f\*\*\*ing","fucking",text)
	text=re.sub("somthng","something",text)
	text=re.sub("s\*\*t","shit",text)
	text=re.sub("&amp;","and",text)
	text=text.replace("Germanwings","german  wings")
	text=text.replace("kouachi","X")
	text=text.replace("sydneysiders","Sydney siders")
	text=text.replace("airtravel","air travel")
	str_text = re.sub("[^a-zA-Z]", " ", text)
	words = nltk.word_tokenize(str_text.lower())
	if remove_stopwords:
                stops = set(stopwords.words("english"))
                words = [w for w in words if w not in stops]
	return(words)

def sumw2v(tweet, avg=True):
    global model_WV
    model = model_WV
    temp_rep = np.zeros(300)
    wordlist = str_to_wordlist(tweet, remove_stopwords=False)
    for w in range(len(wordlist)):
        if wordlist[w] in model:
            temp_rep += model[wordlist[w]]
    if avg and len(wordlist) != 0:
        sumw2v = temp_rep/len(wordlist)
    else:
        sumw2v = temp_rep
    return sumw2v

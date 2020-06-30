import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import numpy as np

dict_sad={":-(":"SAD", ":(":"SAD", ":-|":"SAD",  ";-(":"SAD", ";-<":"SAD", "|-{":"SAD"}
dict_happy={":-)":"HAPPY",":)":"HAPPY", ":o)":"HAPPY",":-}":"HAPPY",";-}":"HAPPY",":->":"HAPPY",";-)":"HAPPY"}

wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def replace_emotion(sentence):
    for i in dict_happy:
        sentence = sentence.replace(i,dict_happy[i])
    for i in dict_sad:
        sentence = sentence.replace(i,dict_sad[i])
    return sentence


def bigram_feat(feat,sentence):
    tokens = tokenizer.tokenize(sentence)
    lemmas = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
    bigrams = nltk.bigrams(lemmas)
    bigrams = [part[0]+' '+part[1] for part in bigrams]
    bigramfeat = lemmas + bigrams
    
    for f in bigramfeat:
        feat['contains(%s)' % f] = 1.0
        
def one_half_sentiment(feat,sentence):
    tokens = tokenizer.tokenize(sentence)
    
    if len(tokens)==1:
        tokens+=['.']
    #print (len(tokens)/2)
    f_half = tokens[0:int(len(tokens)/2)]
    s_half = tokens[int(len(tokens)/2):]
    
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in f_half]).strip())
        feat['sentiment fhalf'] = blob.sentiment.polarity
        feat['subjective fhalf'] = blob.sentiment.subjectivity
    except:
        feat['sentiment fhalf'] = 0.0
        feat['subjective fhalf'] = 0.0
        
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in s_half]).strip())
        feat['sentiment shalf'] = blob.sentiment.polarity
        feat['subjective shalf'] = blob.sentiment.subjectivity
    except:
        feat['sentiment shalf'] = 0.0
        feat['subjective shalf'] = 0.0
        
    feat['one_half_sentiment'] = np.abs(feat['sentiment fhalf'] - feat['sentiment shalf'])


def one_third_sentiment(feat,sentence):
    tokens = tokenizer.tokenize(sentence)
    if len(tokens)==2:
        tokens+=['.']
    f_half = tokens[0:int(len(tokens)/3)]
    s_half = tokens[int(len(tokens)/3):int(2*len(tokens)/3)]
    t_half = tokens[int(2*len(tokens)/3):]
    
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in f_half]).strip())
        feat['sentiment fthird'] = blob.sentiment.polarity
        feat['subjective fthird'] = blob.sentiment.subjectivity
    except:
        feat['sentiment fthird'] = 0.0
        feat['subjective fthird'] = 0.0
        
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in s_half]).strip())
        feat['sentiment sthird'] = blob.sentiment.polarity
        feat['subjective sthird'] = blob.sentiment.subjectivity
    except:
        feat['sentiment sthird'] = 0.0
        feat['subjective sthird'] = 0.0
        
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in t_half]).strip())
        feat['sentiment tthird'] = blob.sentiment.polarity
        feat['subjective tthird'] = blob.sentiment.subjectivity
    except:
        feat['sentiment tthird'] = 0.0
        feat['subjective tthird'] = 0.0
        
    feat['sentiment 1n2'] = np.abs(feat['sentiment fthird'] - feat['sentiment sthird'])
    feat['sentiment 1n3'] = np.abs(feat['sentiment fthird'] - feat['sentiment tthird'])
    feat['sentiment 2n3'] = np.abs(feat['sentiment sthird'] - feat['sentiment tthird'])


def pos_feat(feat, sentence):
    tokens = tokenizer.tokenize(sentence)

    tokens = [tok.lower() for tok in tokens]
    pos_vector = nltk.pos_tag(tokens)
    vector = np.zeros(4)

    for j in range(len(pos_vector)):
        pos=pos_vector[j][1]
        if pos[0:2] == 'NN':
            vector[0]+=1
        elif pos[0:2] == 'JJ':
            vector[1]+=1
        elif pos[0:2] == 'VB':
            vector[2]+=1
        elif pos[0:2] == 'RB':
            vector[3]+=1
      
    for j in range(len(vector)):
        feat['pos' + str(j+1)] = vector[j]


def content_formatting(feat,text,type):
    text=str(text)
    if type=='capital':
        count = 0
        threshold = 4
        for j in range(len(text)):
            if str(text[j]).isupper():
                count=count+1
        feat['capital']= int(count>=threshold)
    if type=='!':
        if type in text:
            feat['!']=1
        else:
            feat['!']= 0    
def count_emotion_happy_sad(feat,sentence):
    happy = 0;
    sad = 0    
    for i in dict_happy:
        happy += sentence.count(i)
    for i in dict_sad:
        sad += sentence.count(i)
    feat['happy']=happy
    feat['sad']=sad

def get_features(sent):
    feat = {}
    content_formatting(feat,sent,'capital')
    content_formatting(feat,sent,'!')
    count_emotion_happy_sad(feat,sent)
    sent = replace_emotion(sent)
    bigram_feat(feat,sent)
    one_half_sentiment(feat,sent)
    one_third_sentiment(feat,sent)
    pos_feat(feat,sent)
    return feat

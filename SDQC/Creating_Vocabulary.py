import pickle,re,nltk

with open('Full/all-tweets.pkl','rb') as t:
  dat=pickle.load(t)

print (len(dat))
print (type(dat[0]))


files=['1-tweets.pkl','2-tweets.pkl','3-tweets.pkl','4-tweets.pkl','5-tweets.pkl','6-tweets.pkl','7-tweets.pkl','8-tweets.pkl']  #train
labels_dict={0:"support",1:"deny",2:"query",3:"comment"}
data=[]
for i in range(0,len(files)):
	with open('Tweets//'+files[i],'rb') as r:
		t=pickle.load(r)
	
	for tt in t:
		for a in tt:
			data.append(a[1])
			
print (len(data))
print (type(data[0]))

def str_to_wordlist(text, remove_stopwords=False):
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


words=[]
for tweet in data:
	words.extend(str_to_wordlist(tweet))
print (len(words))
vocab=set(words)
print (data[0])
print (data[2])

print (len(vocab))

with open('Full/vocabpy2.pkl','wb') as w:
        pickle.dump(vocab,w)

with open('Full/vocabpy2.pkl','rb') as r:
        voc=pickle.load(r)

print (len(voc))

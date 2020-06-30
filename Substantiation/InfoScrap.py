import requests
import pprint as pp
import re
import nltk
import time
import pickle
from SVO import findSVOs,nlp
def remove_tags(text):
	TAG_RE = re.compile(r'<[^>]+>')
	return TAG_RE.sub("",text)

def webscrapping(j):
	s ="" 
	wiki = requests.get(j).text
	from bs4 import BeautifulSoup
	soup = BeautifulSoup(wiki,features="html.parser")
	soup.encode('utf-8')
	page_content = soup.find_all('p')
	for content in page_content:
		s += remove_tags(str(content))
	return s

def google_search():
	try: 
		from googlesearch import search 
	except ImportError: 
		print("No module named 'Google' found")
	

	from googlesearch import search
	trusted_source = ["https://www.indiatoday.in/","https://en.wikipedia.org/wiki/","https://www.ndtv.com/","https://economictimes.indiatimes.com","https://timesofindia.indiatimes.com","https://news18.com"]
	query=""
	while(query!="exit"):
		query = input("Type the search Query ")
		
		s1 =""
		for j in search(query, tld="com", num=10, stop=1, pause=2):
			print (j)
			for t in trusted_source:
				if t in j:
					s1 += webscrapping(j)
		#print s1		
		sentences = nltk.sent_tokenize(s1) 
		#print (sentences)
		
		from pycorenlp import StanfordCoreNLP
		coreNLPInstance=StanfordCoreNLP('http://localhost:9000')
		print('[INFO] Connected to Running Instance')
		print ('[INFO] Possible Knowledge triples')
		triples_list=[]
		start_time=time.time()
		
		for sentence in sentences:
			output=coreNLPInstance.annotate(sentence,properties={'annotators':'openie','outputFormat':'json'})
			openIeObject=output['sentences'][0]['openie']
			#print (openIeObject)
			for rel in openIeObject:
				subject=rel['subject'].lower()
				relation=rel['relation'].lower()
				obj=rel['object'].lower()
				knowledgeTripletTuple=(subject,relation,obj)
				print(knowledgeTripletTuple)
				triples_list.append(knowledgeTripletTuple)
			tokens=nlp(sentence)
			triplet_unicode=findSVOs(tokens)
			for t in triplet_unicode:
				subject=t[0].lower()
				relation=t[1].lower()
				obj=t[2].lower()
				knowledgeTriplet=(subject,relation,obj)
				print (knowledgeTriplet)
				triples_list.append(knowledgeTriplet)
		print (len(triples_list))
		pickle.dump(triples_list,open('test_triples.pkl','wb'))
		
		triples_list=pickle.load(open('test_triples.pkl','rb'))
		print (time.time()-start_time)
		print ('[INFO] Knowledge triplets extracted')

google_search()

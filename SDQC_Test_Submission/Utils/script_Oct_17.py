import csv
from datetime import datetime
from math import floor
import copy
from os.path import isfile
import random
import sys
from pprint import PrettyPrinter
from urllib2 import urlopen
import numpy as np
from sklearn.model_selection import KFold
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.collocations import *	
# from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.classify import SklearnClassifier
#from textblob.classifiers import NaiveBayesClassifier
from sklearn.svm import LinearSVC
import re
import nltk     

import nltk.classify

reload(sys)
sys.setdefaultencoding("utf-8")

p = PrettyPrinter(indent=4)


def feature_and_classify(train,test):
    Noun_pronoun_adj = pos_tag_train_data(train)
    temp_train = copy.deepcopy(train)
    temp_test = copy.deepcopy(test)

    for i in temp_train:
        i[1] = gen_features(i[1],Noun_pronoun_adj)

    for i in temp_test:
        i[1] = gen_features(i[1],Noun_pronoun_adj)

    #print(temp_test[1])
    #print("hello")
    train_lis_temp = feat(temp_train)
    test_lis_temp = feat_test(temp_test)

#    print(test_lis_temp[0])  
 
    print(train_lis_temp)

    # classifier = nltk.classify.SklearnClassifier(LinearSVC())

    # nbc = classifier.train(train_lis_temp)

    # CM(nbc,test_lis_temp)
    # print("Classificaton over for one fold!#######################################################################\n")


def feat(obj):
    temp=[]

    for i in obj:
        temp.append([i[1],i[2]])

    return temp


def feat_test(obj):
    temp=[]

    for i in obj:
        temp.append([ i[0],i[1],i[2] ])

    return temp


def CM(classifier,testing_set):
    gold_result=[]
    test_result=[]

    #print(testing_set)

    prab_mam = []

    print("lidu ")
    print(testing_set[1][0])

    for i in range(len(testing_set)):

        #p.pprint(i)

        id_prab = testing_set[i][0]
        predict_prab =  classifier.classify(testing_set[i][1])
        original_prab = testing_set[i][2]

        prab_mam.append([id_prab,predict_prab,original_prab])

        test_result.append(  classifier.classify( testing_set[i][1] )  )
        gold_result.append( testing_set[i][2] )

    print(test_result)
    print(gold_result)

    Conf = nltk.ConfusionMatrix(gold_result,test_result)
    print(Conf)

    #print(testing_set)

    test_acc = []

    for i in testing_set:
        test_acc.append( [ i[1], i[2] ] )
    
    #print(nltk.classify.accuracy(classifier,testing_set))

    print(nltk.classify.accuracy(classifier,test_acc))

    # print (classification_report(gold_result,test_result))

    p.pprint(prab_mam)

def gen_features(doc, lis_of_Nouns_pronoun_adj):
    splitted_doc = doc.split();

    global lis_of_NT_noun
    global lis_of_NT_pronoun
    global lis_of_NT_adjective
    global lis_of_NT_adverb
    global lis_of_T_noun
    global lis_of_T_pronoun
    global lis_of_T_adjective
    global lis_of_T_adverb

    lis_of_NT_noun = lis_of_Nouns_pronoun_adj[0]
    lis_of_NT_pronoun = lis_of_Nouns_pronoun_adj[1]
    lis_of_NT_adjective = lis_of_Nouns_pronoun_adj[2]
    lis_of_NT_adverb = lis_of_Nouns_pronoun_adj[3]

    lis_of_T_noun = lis_of_Nouns_pronoun_adj[4]
    lis_of_T_pronoun = lis_of_Nouns_pronoun_adj[5]
    lis_of_T_adjective = lis_of_Nouns_pronoun_adj[6]
    lis_of_T_adverb = lis_of_Nouns_pronoun_adj[7]

    fdict={}

    fdict["NT_words"]=[]
    fdict["T_words"] =[]

    for word in  splitted_doc:
        if word in lis_of_NT_noun or word in lis_of_NT_adjective or word in lis_of_NT_pronoun or word in lis_of_NT_adverb:
            fdict["NT_words"].append(word)
        if word in lis_of_T_noun or word in lis_of_T_adjective or word in lis_of_T_pronoun or word in lis_of_NT_adverb:
            fdict["T_words"].append(word)

    fdict["T_words"] = tuple(fdict["T_words"])
    fdict["NT_words"] = tuple(fdict["NT_words"])
    return fdict


def pos_tag_train_data(train_lis):
    lis_of_T_train=[]

    lis_of_NT_train=[]

    for doc in train_lis:
        if doc[2] == 'T':
            lis_of_T_train.append(doc)
        elif doc[2] == 'NT':
            lis_of_NT_train.append(doc)
    #print("lenght of NT's in training list: ", (lis_of_NT_train))
    #print("length of T's in training set: ",(lis_of_T_train))
    #print("Start time for pos_tags: "+str ( datetime.now() ))
    ##4. now lets get pos_tags for all unique words in NT and T list
    pos_tag_for_words_NT_doc=[]

    for doc in lis_of_NT_train:
        pos_tag_for_words_NT_doc.append(  [  doc[0],pos_tag(doc[1].split())   ]   ) 

    pos_tag_for_words_T_doc=[]

    for doc in lis_of_T_train:
        pos_tag_for_words_T_doc.append(  [  doc[0],pos_tag(doc[1].split())   ]   )  
    #print("End time for pos_tags: "+str ( datetime.now() ))
    ##5. Now take one doc and split and form a set : do this for all docs in NT and T thus we will get 500 sets (ofourse the set is converted back to list )
    #print("Start time of finding unique ones for pos_tags: "+str ( datetime.now() ))
    from collections import OrderedDict as od 

    unique_pos_tags_of_T_doc=[]

    for i in pos_tag_for_words_T_doc:
        unique_pos_tags_of_T_doc.append([i[0], list(od.fromkeys(i[1]))])

    unique_pos_tags_of_NT_doc=[]

    for i in pos_tag_for_words_NT_doc:
        unique_pos_tags_of_NT_doc.append([i[0], list(od.fromkeys(i[1]))])
    #print("End time of finding unique ones for pos_tags: "+str ( datetime.now() ))
    # 6. Now we have pos tags for each and every document so now let us join all of them to form a document for all postags of NT and T document
    ##print("Start time of finding unique ones from all: "+str ( datetime.now() ))

    All_pos_tags_T = []

    for each_doc in unique_pos_tags_of_T_doc:
            All_pos_tags_T = All_pos_tags_T + each_doc[1]

    All_pos_tags_NT=[]

    for each_doc in unique_pos_tags_of_NT_doc:
            All_pos_tags_NT = All_pos_tags_NT + each_doc[1]
    # 7. Let us now remove the common postags using od.fromkeys
    All_pos_tags_NT_uniq = od.fromkeys(All_pos_tags_NT)

    All_pos_tags_T_uniq = od.fromkeys(All_pos_tags_T)
    #print("End time of finding unique ones for all: "+str ( datetime.now() ))
    # 8. Now let us separate all postags in T as Nouns, Verbs, Pronouns, Adjectives and Adverbs.
    #print("Start time  for separating pos_tags: "+str ( datetime.now() ))

    lis_of_T_noun=[]
    lis_of_T_adjective=[]
    lis_of_T_pronoun=[]
    lis_of_T_adverb=[]

    for each_word in All_pos_tags_T_uniq:
        if each_word[1].startswith('N'):
            lis_of_T_noun.append(each_word[0])

        elif each_word[1].startswith('J'):
            lis_of_T_adjective.append(each_word[0])

        elif each_word[1].startswith('P'):
            lis_of_T_pronoun.append(each_word[0])

        elif each_word[1].startswith('R'):
            lis_of_T_adverb.append(each_word[0])

    lis_of_NT_noun=[]
    lis_of_NT_adjective=[]
    lis_of_NT_pronoun=[]
    lis_of_NT_adverb=[]

    for each_word in All_pos_tags_NT_uniq:
        if each_word[1].startswith('N'):
            lis_of_NT_noun.append(each_word[0])

        elif each_word[1].startswith('J'):
            lis_of_NT_adjective.append(each_word[0])

        elif each_word[1].startswith('P'):
            lis_of_NT_pronoun.append(each_word[0])
        
        elif each_word[1].startswith('R'):
            lis_of_NT_adverb.append(each_word[0])

    #print("End time  for separating pos_tags: "+str ( datetime.now() ))
    return[lis_of_NT_noun,lis_of_NT_pronoun,lis_of_NT_adjective,lis_of_NT_adverb,lis_of_T_noun,lis_of_T_pronoun,lis_of_T_adjective,lis_of_T_adverb]


def read_dataset_from_csv(path):
    temp = list()
    with open(path,"rt") as csvfile:
        reader = csv.reader(csvfile,delimiter=",")

        for row in reader:
            temp.append(row)
    return temp

def preprocessing(filepath, outputfilepath):

    #PI:ADD
    # cleaned = list()
    # if isfile("processed.csv"):
        # with open("processed.csv") as csvfile:
            # reader = csv.reader(csvfile,delimiter=",")

            # for row in reader:
                # cleaned.append([row[0],row[1],row[2]])

        # return cleaned

    #PI:ADDOVER

    cleaned = list()
    # url = re.compile(r"((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)")
    # url = re.compile(r"\b\S*\:\S*")
    url = re.compile(r"(?:[a-z]+:\/\/|www\.)\S+")
    whitespace = re.compile(r"\s{2,}")

    stop_words = set(word for word in stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    with open(filepath, "rt") as csvfile:
        reader = csv.reader(csvfile,delimiter=",")

        for row in reader:
            temp = " ".join([word.lower() for word in row[1].split() if word.lower() not in stop_words])
            temp = re.sub(url, "", temp)
            temp = u"".join([char for char in temp if (ord(char) > 64 and ord(char) < 91) or (ord(char) > 96 and ord(char) < 123) or ord(char)==32])
            temp = re.sub(whitespace, " ", temp)
            temp = u" ".join([lemmatizer.lemmatize(word) for word in temp.split()])
            cleaned.append([row[0], temp, row[2]])

    writer = csv.writer(open(outputfilepath,"w"))
    writer.writerows(cleaned)
    print("Preprocessing done")
    return cleaned

def split_into_T_NT(filepath):
    T_list = list()
    NT_list = list()

    #PI:ADD

    if isfile("T.csv") and isfile("NT.csv"):
        return

    #PI:ADDOVER

    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            if row[2] == "T":
                T_list.append(row)
            else:
                NT_list.append(row)

    writer = csv.writer(open("T.csv", "w"))
    writer.writerows(T_list)

    writer = csv.writer(open("NT.csv", "w"))
    writer.writerows(NT_list)

def gen_permutations(n, dataset_length):
    permutations = list()
    for i in range(n):
        permutations.append(random.sample(range(dataset_length), dataset_length)) #Lidwin change removed the [] inside random.sample([bla,bla]) to be random.sample(bla,bla)
    return permutations

def scrape(fileinput, fileoutput):
    if not isfile(fileinput):
        print("Could not find " + fileinput)
        exit()

    link="https://bugs.chromium.org/p/chromium/issues/detail?id="
    rows_of_id_label=[]

    with open(fileinput) as f:
        reader = csv.reader(f,delimiter=',')
        next(reader)
        for row in reader:
            rows_of_id_label.append( [ row[0],row[1] ] )

    Total_Descriptions_List=[]
    dataset_length = len(rows_of_id_label)

    ##loop through each row get the id append with the url of issue tracker then do the CRAWL LOGIC
    for ind, ele in enumerate(rows_of_id_label):
        link = link + ele[0]
        page_html = urlopen(link)
        soup = bs(page_html,'html.parser')
        '''Crawling for the Title of the Issue:'''
        ##All the titles of the Issue are found within a TD tag with class issueheader and also there are many more occurances where td along with issueheader is used
        ##but mostly I am sure the First tag with TD and class as issueheader has the title thus taking [0] index of the all TD tags found
        Titles_parent_tag = soup.find_all('td',class_='issueheader')
        Title_text = Titles_parent_tag[0].text.strip()
        '''
        For the title the crawling is over from below the crawling for Issue Description Begins
        Crawling for the Issue description:
        '''
        Issue_Description_parent_tag = soup.find_all('pre',class_='issue_text')
        ##in this the Issue_Description_parent_tag[0] contains the issue description and the other things contain the Comment description
        number_of_hyperlinks = len( Issue_Description_parent_tag[0].find_all('a') )

        ##find the number of Hyperlinks within the description and remove all these anchor tags using a loop and decompose function
        for i in range(number_of_hyperlinks):
            Issue_Description_parent_tag[0].a.decompose()

        Issue_Description_text = " ".join( Issue_Description_parent_tag[0].text.split() )
        Total_description = Title_text+". "+Issue_Description_text
        Total_description = Total_description.replace(';',' ')
        Total_Descriptions_List.append(Total_description)
        #Scraper ends here
        link = "https://bugs.chromium.org/p/chromium/issues/detail?id="
        print "%f percent scraped" % (((ind+1) / float(dataset_length) * 100))

    for ind, el in enumerate(rows_of_id_label):
        el.insert(1, Total_Descriptions_List[ind])

    file_ptr = open(fileoutput,"w")
    writer = csv.writer(file_ptr)
    writer.writerows(rows_of_id_label)
    file_ptr.close()

def sequential_fold(percent_of_train, dataset):
    count_combos=0
    i=0
    percent_of_train
    train_count = int(floor(len(dataset)*(percent_of_train/100.0)))
    test_count = int(floor(len(dataset) - train_count))
    while count_combos < len(dataset):
        train_lis = []
        test_lis  = []
        count_train = 0;
        count_test  = 0;
        j = i
        while count_train < train_count :
            train_lis.append(dataset[j])
            j += 1
            count_train += 1
            if j == len(dataset):
                j = 0
        while count_test < test_count :
            test_lis.append(dataset[j])
            j += 1
            count_test += 1
            if j == len(dataset):
                j = 0
        count_combos += 1
        i += 1 
        yield [train_lis, test_lis]

def kfold(splits, data):
    kf = KFold(n_splits = splits)
    dataset_numpy  = np.array(data) #no need to comment this lidwin

    for traini,testi in kf.split(data):
        train_lis = dataset_numpy[traini]
        test_lis  = dataset_numpy[testi]

        yield [train_lis, test_lis]


def collocation_gen_ngram(files, debt_type):

    #PI:ADD

    if isfile("T_bigrams.csv") and isfile("NT_bigrams.csv"):
        return

    #PI:ADDOVER

    bigrams = list()
    T_features_bigram = list()
    NT_features_bigram = list()

    for file, debt_type in zip(files, debt_type):
        corpus = list()
        with open(file) as sentences_file:
            reader= csv.reader(sentences_file,delimiter=",")
            for row in reader:
                corpus.append([row[1]])
            
        bgm    = nltk.collocations.BigramAssocMeasures()
        finder = list()
        for index, element in enumerate(corpus):
            finder.append([BigramCollocationFinder.from_words(element[0].split())])
        scored = []
        for index, element in enumerate(finder):
            scored.append([element[0].score_ngrams(bgm.likelihood_ratio)])
        # p.pprint(scored[0])

        for i in scored:
            for j in i[0]:
                if file == "T.csv":
                    T_features_bigram.append(j)
                else:
                    NT_features_bigram.append(j)

    sorted_T_bigrams = sorted(T_features_bigram, key=lambda x:x[1], reverse=True)
    sorted_NT_bigrams = sorted(NT_features_bigram, key=lambda x:x[1], reverse=True)
    writer = csv.writer(open("T_bigrams.csv", "w"))
    writer.writerows(sorted_T_bigrams)
    writer = csv.writer(open("NT_bigrams.csv", "w"))
    writer.writerows(sorted_NT_bigrams)

def flagging(lis_of_pos_tags):#PI: added a new param
    # global lis_of_NT_noun
    # global lis_of_NT_pronoun
    # global lis_of_NT_adjective
    # global lis_of_NT_adverb
    # global lis_of_T_noun
    # global lis_of_T_pronoun
    # global lis_of_T_adjective
    # global lis_of_T_adverb

    #PI:ADD

    lis_of_NT_noun = lis_of_pos_tags[0]
    lis_of_NT_pronoun = lis_of_pos_tags[1]
    lis_of_NT_adjective = lis_of_pos_tags[2]
    lis_of_NT_adverb = lis_of_pos_tags[3]

    lis_of_T_noun = lis_of_pos_tags[4]
    lis_of_T_pronoun = lis_of_pos_tags[5]
    lis_of_T_adjective = lis_of_pos_tags[6]
    lis_of_T_adverb = lis_of_pos_tags[7]


    #PI:ADDOVER


    if len(lis_of_T_noun) == len(lis_of_NT_noun) == len(lis_of_T_pronoun) == len(lis_of_NT_pronoun) == len(lis_of_T_adjective) == len(lis_of_NT_adjective) == len(lis_of_T_adverb) == len(lis_of_NT_adverb) == 0:
        print("POS Tagging has to be performed first!")
        return

    lis_of_T_noun_pro = lis_of_T_noun + lis_of_T_pronoun
    lis_of_T_noun_pro_adj = lis_of_T_noun_pro + lis_of_T_adjective
    lis_of_T_noun_pro_adj_adv = lis_of_T_noun_pro_adj + lis_of_T_adverb
    lis_of_NT_noun_pro = lis_of_NT_noun +lis_of_NT_pronoun
    lis_of_NT_noun_pro_adj = lis_of_NT_noun_pro + lis_of_NT_adjective
    lis_of_NT_noun_pro_adj_adv = lis_of_NT_noun_pro_adj + lis_of_NT_adverb

    set_of_T_noun_pro = set(lis_of_T_noun_pro)
    set_of_T_noun_pro_adj = set(lis_of_T_noun_pro_adj)
    set_of_T_noun_pro_adj_adv = set(lis_of_T_noun_pro_adj_adv)
    set_of_NT_noun_pro = set(lis_of_NT_noun_pro)
    set_of_NT_noun_pro_adj = set(lis_of_NT_noun_pro_adj)
    set_of_NT_noun_pro_adj_adv = set(lis_of_NT_noun_pro_adj_adv)

    T_bigrams = read_dataset_from_csv("T_bigrams.csv")[:1000]
    NT_bigrams = read_dataset_from_csv("NT_bigrams.csv")[:1000]

    T_temp = list()
    NT_temp = list()

    #tells how many words in the bigram are in the variable 'set_of_T_noun_pro' and then sort based of flagging this is the use of flagging  

    for row in T_bigrams:
        bg = eval(row[0])
        count = 0
        if bg[0] in set_of_T_noun_pro:
            count += 1
        if bg[1] in set_of_T_noun_pro:
            count += 1
        T_temp.append([(bg[0], bg[1]), count])

    for row in NT_bigrams:
        bg = eval(row[0])
        count = 0
        if bg[0] in set_of_NT_noun_pro:
            count += 1
        if bg[1] in set_of_NT_noun_pro:
            count += 1
        NT_temp.append([(bg[0], bg[1]), count])

    sorted_T_temp = sorted(T_temp, key=lambda x:x[1], reverse=True)
    sorted_NT_temp = sorted(NT_temp, key=lambda x:x[1], reverse=True)

    writer = csv.writer(open("T_flag.csv","w"))
    writer.writerows(sorted_T_temp)
    writer = csv.writer(open("NT_flag.csv","w"))
    writer.writerows(sorted_NT_temp)

def strong_and_weak_bigrams(): #this separates the T_flag and NT_flag into 2 categories namely Strong and Weak based on the flag count ie: 2 or 1
    if not isfile("T_flag.csv") or not isfile("NT_flag.csv"):
        print("Please flag data first")
        return

    T_data = read_dataset_from_csv("T_flag.csv")
    NT_data = read_dataset_from_csv("NT_flag.csv")

    strong_T, weak_T, strong_NT, weak_NT = [], [], [], []

    for i in T_data:
        if eval(i[1]) == 2:
            strong_T.append(eval(i[0]))
        if eval(i[1]) == 1:
            weak_T.append(eval(i[0]))

    for i in NT_data:
        if eval(i[1]) == 2:
            strong_NT.append(eval(i[0]))
        if eval(i[1]) == 1:
            weak_NT.append(eval(i[0]))
    #PI: p.pprint(strong_T)
    return strong_T, weak_T, strong_NT, weak_NT

def choose_feat_and_classifier(train, test, i_feat_sel, i_classifier):
    classifiers = [nltk.NaiveBayesClassifier, SklearnClassifier(LinearSVC()), DecisionTreeClassifier]
    classifier = classifiers[i_classifier-1]
    #PI:global strong_T, weak_T, strong_NT, weak_NT
    #PI:DOWN strong_T, weak_T, strong_NT, weak_NT = strong_and_weak_bigrams()

    if i_feat_sel == 1:
        #postagging
        feature_and_classify(train,test)

    elif i_feat_sel == 2:
        #postagging first
        #PI:DEL feature_and_classify(train,test) reason: this must not be called as it also does the 1st feature selection by itself
        #PI:ADD

        lis_of_pos_tags = pos_tag_train_data(train) 
        
        flagging(lis_of_pos_tags)
        #PI:ADDOVER

        strong_T, weak_T, strong_NT, weak_NT = strong_and_weak_bigrams()

        #PI: p.pprint(strong_T)
        #Strong collocation
        list_of_train_set_bigrams_T = []
        list_of_test_set_bigrams_T = []
        list_of_train_set_bigrams_NT = []
        list_of_test_set_bigrams_NT = []

        master_training=[]
        master_test=[]

        for doc in train:
            temp = {}
            bigrams_T = []
            bigrams_NT = []
            # p.pprint(doc[1])
            for bigram in nltk.bigrams(doc[1].split()):
                if bigram in strong_T:
                    bigrams_T.append(bigram)
                elif bigram in strong_NT:
                    bigrams_NT.append(bigram)
            #PI:if bigrams_T:
            temp["Strong_T"] = bigrams_T
            #PI: list_of_train_set_bigrams_T.append([temp, doc[2] ])
            #PI: if bigrams_NT:
            temp["Strong_NT"] = bigrams_NT
            #PI: list_of_train_set_bigrams_NT.append([temp, doc[2] ])
            master_training.append([temp,doc[2]])


        print("hello david ",len(master_training))

        p.pprint(master_training)

        for i in master_training :
            try:
                i[0]['Strong_T'] = tuple(i[0]['Strong_T'])
                i[0]['Strong_NT'] = tuple(i[0]['Strong_NT'])
            except:
                pass    

            #print(i)
            #break

        var = classifier.train(master_training)   

        #print(master_training[-1][0])

        #print(    var.classify(   master_training[-1][0]     )     )     


        for doc in test:
            temp = {}
            bigrams_T = []
            bigrams_NT = []
            for bigram in nltk.bigrams(doc[1].split()):
                if bigram in strong_T:
                    bigrams_T.append(bigram)
                elif bigram in strong_NT:
                    bigrams_NT.append(bigram)
            #PI:if bigrams_T:
            temp["Strong_T"] = bigrams_T
            #list_of_test_set_bigrams_T.append([temp])#PI:, "T"])
            #PI:if bigrams_NT:
            temp["Strong_NT"] = bigrams_NT
            #list_of_test_set_bigrams_NT.append([temp])#PI:, "NT"])
            master_test.append([temp,doc[2]])#master_test.append([doc[0],temp,doc[2]])
        
        #training_set = list_of_train_set_bigrams_T + list_of_test_set_bigrams_NT


#    PI:    p.pprint(training_set[0][0]["Strong"])



        #PI: testing_set = list_of_test_set_bigrams_T + list_of_test_set_bigrams_NT

        print(len(master_test))

        for i in master_test:
            try:
                i[0]['Strong_T'] = tuple(i[0]['Strong_T'])
                i[0]['Strong_NT'] = tuple(i[0]['Strong_NT'])
            except:
                pass    

        result, output = [], []

        #print("GOKUL SP SIR")

        

        praba_mam = []


        for i in master_test:
            #print(i[0])
            output = var.classify(i[0])
            #print("david")
            #print( var.classify( {'Strong_T': ( u'chrome', u'version', u'step', u'reproduce', u'reproduce', u'problem' ), 'Strong_NT': ( u'attach', u'screenshot' ) }   ) )

            #working feature type 
            #print(var.classify( {'Strong_T': ( (u'chrome', u'version'), (u'step', u'reproduce'), (u'reproduce', u'problem') ), 'Strong_NT': ( (u'attach', u'screenshot'), ) } ) )

            #praba_mam.append(i[0], var.classify(i[1]), i[2])
            result.append(output)
        CM(var, master_test)



        print(praba_mam)

    elif i_feat_sel == 3:
        #PI:ADD


        lis_of_pos_tags = pos_tag_train_data(train) 
        
        flagging(lis_of_pos_tags)
        #PI:ADDOVER

        strong_T, weak_T, strong_NT, weak_NT = strong_and_weak_bigrams()

        #PI: p.pprint(strong_T)
        #Strong collocation
        list_of_train_set_bigrams_T = []
        list_of_test_set_bigrams_T = []
        list_of_train_set_bigrams_NT = []
        list_of_test_set_bigrams_NT = []

        master_training=[]
        master_test=[]

        for doc in train:
            temp = {}
            bigrams_T = []
            bigrams_NT = []
            # p.pprint(doc[1])
            for bigram in nltk.bigrams(doc[1].split()):
                if bigram in weak_T:
                    bigrams_T.append(bigram)
                elif bigram in weak_NT:
                    bigrams_NT.append(bigram)
            #PI:if bigrams_T:
            temp["Weak_T"] = bigrams_T
            #PI: list_of_train_set_bigrams_T.append([temp, doc[2] ])
            #PI: if bigrams_NT:
            temp["Weak_NT"] = bigrams_NT
            #PI: list_of_train_set_bigrams_NT.append([temp, doc[2] ])
            master_training.append([temp,doc[2]])


        print("hello sp sir ",len(master_training))

        #p.pprint(master_training)

        for i in master_training :
            try:
                i[0]['Weak_T'] = tuple(i[0]['Weak_T'])
                i[0]['Weak_NT'] = tuple(i[0]['Weak_NT'])
            except:
                pass    

            #print(i)
            #break

        var = classifier.train(master_training)   

        #print(master_training[-1][0])

        #print(    var.classify(   master_training[-1][0]     )     )     


        for doc in test:
            temp = {}
            bigrams_T = []
            bigrams_NT = []
            for bigram in nltk.bigrams(doc[1].split()):
                if bigram in strong_T:
                    bigrams_T.append(bigram)
                elif bigram in strong_NT:
                    bigrams_NT.append(bigram)
            #PI:if bigrams_T:
            temp["Weak_T"] = bigrams_T
            #list_of_test_set_bigrams_T.append([temp])#PI:, "T"])
            #PI:if bigrams_NT:
            temp["Weak_NT"] = bigrams_NT
            #list_of_test_set_bigrams_NT.append([temp])#PI:, "NT"])
            master_test.append([temp,doc[2]])
        
        #training_set = list_of_train_set_bigrams_T + list_of_test_set_bigrams_NT


#    PI:    p.pprint(training_set[0][0]["Strong"])



        #PI: testing_set = list_of_test_set_bigrams_T + list_of_test_set_bigrams_NT

        #print(len(master_test))

        for i in master_test:
            try:
                i[0]['Weak_T'] = tuple(i[0]['Weak_T'])
                i[0]['Weak_NT'] = tuple(i[0]['Weak_NT'])
            except:
                pass   


        print(master_test[0]) 

        result, output = [], []
        for i in master_test:
            #print(i[0])
            output = var.classify(i[0])
            result.append(output)
        CM(var, master_test)



        #PI:ADDOVER


#PI:DEL Below
'''
        #postagging first
        feature_and_classify(train,test)
        #Weak collocation
        list_of_train_set_bigrams_T = []
        list_of_test_set_bigrams_T = []
        list_of_train_set_bigrams_NT = []
        list_of_test_set_bigrams_NT = []

        for doc in train:
            temp = {}
            bigrams_T = []
            bigrams_NT = []
            for bigram in nltk.bigrams(doc[1].split()):
                if bigram in weak_T:
                    bigrams_T.append(bigram)
                elif bigram in weak_NT:
                    bigrams_NT.append(bigram)
            if bigrams_T:
                temp["weak"] = bigrams_T
                list_of_train_set_bigrams_T.append([temp, "T"])
            if bigrams_NT:
                temp["weak"] = bigrams_NT
                list_of_train_set_bigrams_NT.append([temp, "NT"])

        for doc in test:
            temp = {}
            bigrams_T = []
            bigrams_NT = []
            for bigram in nltk.bigrams(doc[1].split()):
                if bigram in weak_T:
                    bigrams_T.append(bigram)
                elif bigram in weak_NT:
                    bigrams_NT.append(bigram)
            if bigrams_T:
                temp["weak"] = bigrams_T
                list_of_test_set_bigrams_T.append([temp, "T"])
            if bigrams_NT:
                temp["weak"] = bigrams_NT
                list_of_test_set_bigrams_NT.append([temp, "NT"])
'''

if __name__ == "__main__":

    #PI: lis_of_NT_noun = list()
    #PI:lis_of_NT_pronoun = list()
    #PI:lis_of_NT_adjective = list()
    #PI:lis_of_NT_adverb = list()

    #PI:lis_of_T_noun  = list()
    #PI:lis_of_T_pronoun = list()
    #PI:lis_of_T_adjective = list()
    #PI:lis_of_T_adverb  = list()

    #strong_T, weak_T, strong_NT, weak_NT = [], [], [], []

    i_dataset = 2#PI:Change:int(raw_input("scrape(1) or use existing file(2)?\n\n"))

    scrape_input = "only_id_label.csv"
    scrape_output = "id_desc_label_crawl.csv"
    preprocess_output = "processed.csv"

    if i_dataset == 1:
        scrape(fileinput=scrape_input, fileoutput=scrape_output)
        print("Scraping successful!")
        dataset = preprocessing(scrape_output, preprocess_output)
    else:
        while True:
            # i_filepath = "p.csv"#PI:raw_input("Please enter the file path:\n\n")
            i_filepath = "p.csv"#raw_input("Please enter the file path:\n\n")
            if isfile(i_filepath):
                dataset = preprocessing(i_filepath, preprocess_output)
                print("\nDataset loaded successfully!\n")
                break
            print("\nNot a valid file path\n")
    split_into_T_NT(preprocess_output)

    collocation_gen_ngram(files = ["T.csv", "NT.csv"], debt_type = ["T", "NT"])

    #PI:flagging("flag.csv")

    i_foldtype = 1#PI:int(raw_input("\nkfold(1) or sequential folds(2)?\n\n"))

    if i_foldtype==1:
        i_folds = 8#PI:int(raw_input("\nPlease enter the number of folds:\n\n"))
    else:
        i_trainpercent = int(raw_input("\nPlease enter the percent of dataset to use for training:\n\n"))
    # i_perm = 0#PI:int(raw_input("\nPlease enter the number of additional randomized dataset permutations to use:\n\n"))
    # i_feat_sel = 2#PI:int(raw_input("Please select a feature selection method\n1.POS Tagging\n2.Strongly collocated bigrams\n3.Weakly collocated bigrams\n4.Word sense disambiguation (Lesk)\n\n"))
    # i_classifier = 1#PI:int(raw_input("Please select a classifier\n1.Naive Bayes\n2.SVM\n3.Decision Tree\n4.Random Forest\n5.J48\n\n"))
    i_perm = 0#int(raw_input("\nPlease enter the number of additional randomized dataset permutations to use:\n\n"))
    i_feat_sel = int(raw_input("Please select a feature selection method\n1.POS Tagging\n2.Strongly collocated bigrams\n3.Weakly collocated bigrams\n4.Word sense disambiguation (Lesk)\n\n"))
    i_classifier = int(raw_input("Please select a classifier\n1.Naive Bayes\n2.SVM\n3.Decision Tree\n4.Random Forest\n5.J48\n\n"))

    # permutations = gen_permutations(i_perm, len(dataset))
    # permutations_dataset = list()

    # for ind in permutations:
    #     temp = list()
    #     for i in ind:
    #         temp.append(dataset[i])
    #     permutations_dataset.append(temp)

    if i_foldtype == 1:
        for train, test in kfold(i_folds, dataset):
            train = train.tolist()
            test = test.tolist()
            choose_feat_and_classifier(train, test, i_feat_sel, i_classifier)

        # for ele in permutations_dataset:
        #     for train, test in kfold(i_folds, ele):
        #         train = train.tolist()
        #         test = test.tolist()
        #         choose_feat_and_classifier(train, test, i_feat_sel, i_classifier)
    else:
        for train, test in sequential_fold(i_trainpercent, dataset):
            choose_feat_and_classifier(train, test, i_feat_sel, i_classifier)

        for ele in permutations_dataset:
            for train, test in sequential_fold(i_trainpercent, ele):
                choose_feat_and_classifier(train, test, i_feat_sel, i_classifier)

#PI:UP    collocation_gen_ngram(files = ["T.csv", "NT.csv"], debt_type = ["T", "NT"])
#PI:UP    flagging("flag.csv")


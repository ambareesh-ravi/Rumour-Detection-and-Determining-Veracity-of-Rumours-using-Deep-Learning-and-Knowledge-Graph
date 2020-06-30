import pickle
import pprint as pp
import numpy

#files=['1-tweets.pkl','2-tweets.pkl','4-tweets.pkl','5-tweets.pkl','6-tweets.pkl','8-tweets.pkl']  #train
files=['3-tweets.pkl','7-tweets.pkl']  #test
#files=['1-tweets.pkl','2-tweets.pkl','3-tweets.pkl','4-tweets.pkl','5-tweets.pkl','6-tweets.pkl','8-tweets.pkl']  #train
#files=['1-tweets.pkl']

#labels=['1-ebola-essien_labels.pkl','2-putinmissing_labels.pkl','4-prince-toronto_labels.pkl','5-ferguson_labels.pkl','6-ottawashooting_labels.pkl','8-sydneysiege_labels.pkl'] #train
#labels=['1-ebola-essien_labels.pkl']#test
#labels=['1-ebola-essien_labels.pkl','2-putinmissing_labels.pkl','3-germanwings-crash_labels.pkl','4-prince-toronto_labels.pkl','5-ferguson_labels.pkl','6-ottawashooting_labels.pkl','8-sydneysiege_labels.pkl'] #train
#labels=['3-germanwings-crash_labels.pkl']
labels=['3-germanwings-crash_labels.pkl','7-charliehebdo_labels.pkl']
final_list=[]
final_list_labels=[]
final_edges=[]

for i in range(0,len(files)):
  with open('Tweets\\'+files[i],'rb') as r:
    tweets=pickle.load(r)
  with open('Labels\\'+labels[i],'rb') as r1:
    label=pickle.load(r1)
    final_arr=[]
    two_arr=[]
    arr1=[]
    arr2=[]
    c=0
    for j in range(0,len(tweets)):
      #final_list_labels.append(label[i])
      arr1=[]
      arr2=[]
      #two_arr=[]
      labels_list=[]
      if len(tweets[j])>=3:
        final_list.append(tweets[j])
        for k in range(0,len(tweets[j])):
          labels_list.append(label[j][k])
          if k>=1:
            arr1.append(k)
            arr2.append(tweets[j][k][2])
        arr1=numpy.array(arr1)
        arr2=numpy.array(arr2)
        two_arr=numpy.vstack((arr1,arr2))
      #two_arr=numpy.zeros(arr1.shape[1])
      #two_arr.append(arr1)
      #two_arr.append(arr2)
      #two_arr=numpy.array(two_arr)
      #final_arr.append(two_arr)
      #if two_arr.shape[1]>=3:
        final_edges.append(two_arr)
        if arr1.shape==0 and arr2.shape==0:
          print tweets[j][k]

        labels_list=numpy.array(labels_list)
        final_list_labels.append(labels_list)
    #print type(final_arr)
    #pp.pprint (final_arr)
  

print(len(final_list))
print (len(final_list_labels))
print final_list_labels[0]
print len(final_edges),final_edges[0].shape
with open('Test\dataset_test.pkl','wb') as w:
  pickle.dump(final_list,w)
with open('Test\dataset_test.pkl','rb') as r:
  sai=pickle.load(r)
print (len(sai))
with open('Test\labels_test.pkl','wb') as w1:
  pickle.dump(final_list_labels,w1)
with open('Test\edges_test.pkl','wb') as w2:
  pickle.dump(final_edges,w2)
with open('Test\edges_test.pkl','rb') as r1:
  saii=pickle.load(r1)
print (len(saii))

import json
import pickle
from pprint import pprint

with open('dev-key.json') as f:
    data = json.load(f)
#for i in data['subtaskaenglish']:
with open('Train\dataset_train.pkl','rb') as dtrain:
	dtr = pickle.load(dtrain)

print len(dtr)

with open('Test\dataset_test.pkl','rb') as dtest:
	dts = pickle.load(dtest)

print len(dts)
	
dtr.extend(dts)
finalList = dtr

print len(finalList),finalList[0][0]

final = []

for i,j in data['subtaskaenglish'].items():	
	for k in range(len(finalList)):
		for t in range(len(finalList[k]):
			if(i in finalList[k][t][0]):
				final.append(finalList[k][t])
			

#pprint(data)
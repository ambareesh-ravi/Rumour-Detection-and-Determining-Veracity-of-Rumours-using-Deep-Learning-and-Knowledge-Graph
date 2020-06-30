import json
import pprint as pp
import pickle
import numpy as np

with open("test_dataset.json","r") as rd_json:
	data = json.load(rd_json)
	
source = []	
ans = []
count = 0 
glob = 0

def edgefunc(struct,index):
	for key,value in struct.items():
		source.append([key,index])
		global count
		count = count + 1
		glob=len(source)-1
		if value:
			edgefunc(value,glob)

		
for key,value in data.items():
	for key1,value1 in value.items():
		struct = value[key1]["structure"]
		source = []
		edgefunc(struct,glob)
		ans.append(source)
	
with open("edge_output.pkl","w") as edge:
	pickle.dump(ans,edge)
	
print(count)
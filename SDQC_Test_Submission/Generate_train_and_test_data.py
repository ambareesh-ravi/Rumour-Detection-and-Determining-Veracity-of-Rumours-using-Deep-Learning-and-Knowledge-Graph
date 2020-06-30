import pickle
import pprint as pp
import numpy

with open("test_tweet_text.pkl","r") as file:
	data = pickle.load(file)

id_list = []
tweet_list = []
edge_list = []
count = 0

for i in range(len(data)):
	temp_id = []
	temp_tweet = []
	edge_array = []
	edge_index = []
	for j in range(len(data[i])):
		temp_id.append(data[i][j][0])
		temp_tweet.append(data[i][j][1])
		count = count + 1
		if j>0 :
			edge_index.append(j)
			edge_array.append(data[i][j][2])
	id_list.append(temp_id)
	tweet_list.append(temp_tweet)
	a1 = numpy.array(edge_index)
	a2 = numpy.array(edge_array)
	final_stacked_edges = numpy.vstack((a1,a2))
	edge_list.append(final_stacked_edges)

with open("Test\ids_separated.pkl","w") as id:
	pickle.dump(id_list,id)
with open("Test\\tweets_separated.pkl","w") as tweets:
	pickle.dump(tweet_list,tweets)
with open("Test\\edges_separated.pkl","w") as edges:
	pickle.dump(edge_list,edges)
	
print(count)	
		
		


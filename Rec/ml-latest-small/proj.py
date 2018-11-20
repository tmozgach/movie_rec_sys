import pandas as pd
from surprise import Reader, Dataset
from surprise import SVD, evaluate
from surprise import NMF
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import accuracy
import numpy as np


#def get_top_n(predictions,n=10):
	#top_n=defaultdict(list)
	#for uid,iid,true_r,est,


ratings=pd.read_csv('ratings.csv')
movies=pd.read_csv('movies.csv')

print("Movies dictionary: ",movies.keys())
print("Type of ratings is: ",type(ratings))

print(ratings.keys())

dictratings=ratings.keys()

ratings_dict={'itemID': list(ratings.movieId),
	      'userID': list(ratings.userId),
	      'rating': list(ratings.rating)}

#print("ratings.movieId")
#print(list(ratings.movieId))

movies_dict={
		'itemID': list(movies.movieId),
		'title': list(movies.title),
		'genres': list(movies.genres)
	    }
df=pd.DataFrame(ratings_dict)
df_z=pd.DataFrame(movies_dict)

print("df_Z")
print(df_z.head(12))
reader=Reader(rating_scale=(0.5,5.0))

#print("df['userID']: ",df['userID'])

data=Dataset.load_from_df(df[['userID','itemID','rating']],reader)
#columns=['itemID','title','genres']
#df_z=df_z.reindex(columns=columns)

#reader_z=Reader(line_format='itemID title genres',sep=',')
#data_2=Dataset.load_from_df(df[['itemID','title','genres']],reader)



keyslist=[]

for key in ratings.keys():
	keyslist.append(key)


print("First 10 ratings: ");
print(ratings.head(10));


#data.split(n_folds=5)
bsl_options={'method':'sgd','learning_rate':0.0005}
#algo=NMF()
kf=KFold(n_splits=3)
algo=NMF()

algo_list=[]



epoch_list=[10,20,30,40,50,60]

#for epoch in epoch_list:
	#algo_list.append(NMF(n_epochs=epoch))

#print(algo_list)

for epoch in range(90):
	print("epoch: ",epoch)
	algo=NMF(n_epochs=epoch)
	for trainset,testset in kf.split(data):
		algo.fit(trainset)
		predictions=algo.test(testset)
		acc=accuracy.rmse(predictions,verbose=True)
		print("acc: ",acc)



#for trainset,testset in kf.split(data):
	#algo.fit(trainset)
	#predictions=algo.test(testset)
	#accuracy.rmse(predictions,verbose=True)
#print(algo)



#cross_validate(algo,data,measures=['RMSE'],cv=2,verbose=True)
#evaluate(algo,ratings,measures=['RMSE'])
#baseln=algo.compute_baselines()

#print("baseln: ",baseln)



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import pdb
## Load ratings into a dataframe
#df_ratings=pd.read_csv("ml-latest-small/ratings.csv")
#print(df_ratings.keys())

## Load users into another dataframe
#df_users=pd.read_csv("ml-latest-small/users.csv")
#print(df_users.keys())

## Load movies into another dataframe
#df_movies=pd.read_csv("ml-latest-small/movies.csv")
#print(df_movies.keys())

movies=pd.read_csv('ml-1m/movies.dat',sep="::",header=None,engine='python',
encoding='latin-1')

users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

print(movies.head())

print(ratings.head())

#pdb.set_trace()

training_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set=np.array(training_set,dtype='int')

test_set=pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set=np.array(test_set,dtype='int')

print("Dimensions of training set ", training_set.shape)
print("Dimensions of testing set ",test_set.shape)

#pdb.set_trace()

nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))


print(nb_users)
print(nb_movies)

def convert(data):
	#pdb.set_trace()
	new_data=[]
	for id_users in range(1,nb_users+1):
		id_movies=data[:,1][data[:,0]==id_users]
		id_ratings=data[:,2][data[:,0]==id_users]
		ratings=np.zeros(nb_movies)
		ratings[id_movies-1]=id_ratings
		new_data.append(list(ratings))
	return new_data

#pdb.set_trace()
training_set=convert(training_set)

test_set=convert(test_set)

#print(training_set[0])

np.save('training_set_proc.npy',training_set)
np.save('test_set_proc.npy',test_set)
training_set = np.load('training_set_proc.npy')
test_set = np.load('test_set_proc.npy')

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

class SAE(nn.Module):
    def __init__(self, ):
        # super function is used to use classes of parent class
        super(SAE,self).__init__()
        # by this we can get all the inherited classes of nn.Module
        # first argument is the features, second is the the number of units
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20,10) #second layer has 10 neurons
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,nb_movies)
        self.activation = nn.Sigmoid()
        self.activation_t = nn.Tanh()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation_t(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sae = SAE()
sae=sae.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(sae.parameters(),lr=0.01,weight_decay=0.5)

nb_epoch = 20

for epoch in range(1,nb_epoch+1):
    train_loss = 0
    pdb.set_trace()
    #number of users who at least rated one movie to reduce the computation
    s = 0. #RMSE needs a float
    for id_user in range(nb_users):
        pdb.set_trace()
        input = Variable(training_set[id_user,:]).unsqueeze(0) #dimension for a batch is also added here on the 0th axis
        input=input.to(device)
        #weights in this code are updated after each i/p vector
        target = input.clone() #target is the input
        target=target.to(device)
        if torch.sum(target.data > 0) > 0:
            # if the user has rated atleast one movie
            output = sae(input)
            output=output.to(device) #predicted ratings are returned for this particular user
            target.require_grad = False #gradient is a clone of i/p so now its gradient won't be calculated
            output[target == 0] = 0 #These are zeros so they don't need to be included in the computation
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #denominator should not be zero
            # it is the average of the error with non-zero ratings
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector) #loss.data[0] is the loss value
            s+=1
            optimizer.step()
            # backward decides the direction(increased or decreased) of the weights and optimizer decides their intensities
    print('epoch: '+str(epoch)+ 'loss: '+ str(train_loss/s)) #train loss divided by number of users with atleast one rating


test_loss = 0
s = 0.

for id_user in range(nb_users):
    pdb.set_trace()
    input = Variable(training_set[id_user,:]).unsqueeze(0) #because we are predicting ratings for that user
    input=input.to(device)
    target = Variable(test_set[id_user,:]).unsqueeze(0)
    target=target.to(device)
    if torch.sum(target.data > 0) > 0:
        s+=1.
        output = sae(input)
        output=output.to(device)
        pdb.set_trace()
        target.require_grad = False
        output[target == 0] = 0
        pred_loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(pred_loss.data[0]*mean_corrector)
print('test loss: '+ str(test_loss/s))



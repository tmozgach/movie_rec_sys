import torch
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm
import pandas as pd
from sklearn import model_selection
from statistics import mean
import pdb
class NeuralMatrixFactorization(nn.Module):
	def __init__(self,max_user,max_item,user_k=10,item_k=10,hidden_dim=50):
		super().__init__()
		self.user_emb=nn.Embedding(max_user,user_k,0)
		self.item_emb=nn.Embedding(max_item,item_k,0)
		self.mlp=nn.Sequential(
			nn.Linear(user_k+item_k,hidden_dim),
			nn.ReLU(),
			nn.BatchNorm1d(hidden_dim),
			nn.Linear(hidden_dim,hidden_dim),
			nn.ReLU(),
			nn.BatchNorm1d(hidden_dim)
		)
	def forward(self,x):
		user_idx=x[:,0]
		item_idx=x[:,1]
		user_feature=self.user_emb(user_idx)
		item_feature=self.item_emb(item_idx)
		out=torch.cat([user_feature,item_feature],1)
		out=self.mlp(out)
		out=nn.functional.sigmoid(out)*5
		return out.squeeze()



class MatrixFactorization(nn.Module):
	def __init__(self,max_user,max_item,k=20):
		super().__init__()
		self.max_user=max_user
		self.max_item=max_item
		self.user_emb=nn.Embedding(max_user,k,0)
		#self.
		self.item_emb=nn.Embedding(max_item,k,0)

	def forward(self,x):
		user_idx=x[:,0]
		item_idx=x[:,1]
		user_feature=self.user_emb(user_idx)
		item_feature=self.item_emb(item_idx)

		out=torch.sum(user_feature*item_feature,1)
		out-nn.functional.sigmoid(out)*5
		return out

def eval_net(net,loader,score_fn=nn.functional.l1_loss,device="cpu"):
	ys=[]
	ypreds=[]
	for x,y in loader:
		x=x.to(device)
		ys.append(y)
		with torch.no_grad():
			y_pred=net(x).to("cpu").view(-1)
		ypreds.append(y_pred)
	score=score_fn(torch.cat(ys).squeeze(),torch.cat(ypreds))
	return score.item()



df=pd.read_csv("ml-latest-small/ratings.csv")
#df=pd.read_csv("ml-latest/ratings.csv")
print("dataframe keys: ",df.keys())
X=df[["userId","movieId"]].values
print(X.shape)
Y=df[["rating"]].values
print(Y.shape)

pdb.set_trace()
train_X,test_X,train_Y,test_Y=model_selection.train_test_split(X,Y,test_size=0.1)

print("Type of our training and testing variables are",type(train_X),type(test_X),type(train_Y),type(test_Y))

train_dataset=TensorDataset(
	torch.tensor(train_X,dtype=torch.int64),torch.tensor(train_Y,dtype=torch.float32))

#print(train_dataset.shape)
print(type(train_dataset))

test_dataset=TensorDataset(
	torch.tensor(test_X,dtype=torch.int64),torch.tensor(test_Y,dtype=torch.float32)
)

train_loader=DataLoader(train_dataset,batch_size=1024,num_workers=4,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=1024,num_workers=4)

print("type of X",type(X))
max_user,max_item = X.max(0)

print("max_user: ",max_user,"max item: ",max_item)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net=MatrixFactorization(max_user+1,max_item+1)
net=net.to(device)
print(net)

#print(X[:,1].shape)
opt=optim.Adam(net.parameters(),lr=0.01)
loss_f=nn.MSELoss()


for epoch in range(5):
	print(epoch)
	loss_log=[]
	for x,y in tqdm.tqdm(train_loader):
		x=x.to(device)
		y=y.to(device)
		o=net(x)
		loss=loss_f(o,y.view(-1))
		pdb.set_trace()
		net.zero_grad()
		loss.backward()
		opt.step()
		loss_log.append(loss.item())
	test_score=eval_net(net,test_loader,device=device)
	print("epoch test: ",epoch,"mean test: ",mean(loss_log),"test score: ",test_score,flush=True)

#net.to("cpu")

pdb.set_trace()
#query=torch.stack([
	#torch.zeros(max_item).fill_(1),
	#torch.arange(1, int(max_item+1) )
	#],1)
query=torch.stack([torch.zeros(max_item).fill_(1),torch.arange(1,int(max_item+1)).float()],1).long()
#pdb.set_trace()
query=query.to(device)
#pdb.set_trace()
scores,indices=torch.topk(net(query),5)
print(scores,indices)

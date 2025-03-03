import torch
import random

datas=[]
labels=[]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(10000):
	t=random.randint(0,1)
	for _ in range(2):
		data=torch.zeros(9,9,3)
		data[:,0,0]=data[:,8,0]=data[0,:,0]=data[8,:,0]=1
		data[8,8,2]=1
		x,y=random.randint(1,7),random.randint(1,7)
		data[x,y,1]=1
		act=1+(x+y+t)%2*2
		data=torch.concat((data.view(-1),torch.tensor([act])))
		datas.append(data)

datas=torch.stack(datas).to(device)
datas=datas.view(-1,2*(9*9*3+1))
labels=datas[:,-1].view(-1)
datas=datas[:,:-1]

class Net(torch.nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.fc1=torch.nn.Linear(9*9*3*2+1,128)
		self.fc2=torch.nn.Linear(128,16)
		self.fc3=torch.nn.Linear(16,1)
		self.relu=torch.nn.ReLU()
	def forward(self,x):
		x=self.relu(self.fc1(x))
		x=self.relu(self.fc2(x))
		x=self.fc3(x)
		return x
	
net=Net().to(device)
optimizer=torch.optim.Adam(net.parameters(),lr=0.001)

for epoch in range(10000):
	optimizer.zero_grad()
	output=net(datas)
	# print(output.shape,labels.shape)
	loss=((labels-output.view(-1))**2).mean()
	loss.backward()
	optimizer.step()
	print(epoch,loss.item())

# output=net(datas).view(-1)
# print((torch.abs(output-loss)<0.2).float().mean())
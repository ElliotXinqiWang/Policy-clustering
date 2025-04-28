import torch
import math
import random
import numpy as np
from torch import nn

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def e1(i,j):
	alpha=1
	return alpha/(abs(i-j)+1)
def e2(i,j):
	k=3
	param=[[2,2,1],[3,8,2],[9,1,3]]
	res=math.inf
	for k in param:
		res=min(res,((i-k[0])**2+(j-k[1])**2)/k[2]**2)
	return res
def f1(x): #TV(p||q)
	return abs(x-1), -1 if x<1 else 1
def f2(x): #KL(p||q)
	return x*torch.log(x), torch.log(x)+1
def f3(x): #KL(q||p)
	return -torch.log(x), -1/x
def f4(x): #JS(p||q)
	return (x*torch.log(2*x/(x+1))+torch.log(2/(x+1)))/2, torch.log(2*x/(x+1))/2

def train(n,r,algo,_energe,f,normalized=False,iter=1000000):
	def energe(i,j):
		return _energe(i+1,j+1)
	def kl_div():
		W=torch.softmax(w,0)
		q=torch.zeros((n,n),device=device)
		for i in range(r):
			q+=W[i]*torch.softmax(t[0][i],0)[:,None]*torch.softmax(t[1][i],0)[None,:]
		return torch.sum(p*torch.log(p/q)).item()
	
	p=torch.zeros((n,n))
	for i in range(n):
		for j in range(n):
			p[i,j]=math.exp(-energe(i,j))
	p=p/p.sum()

	w=torch.randn(r,device=device)
	t=torch.randn(2,r,n,device=device)
	if normalized:
		s=0
		for i in range(n):
			for j in range(n):
				s+=math.exp(-energe(i,j))
		z=torch.tensor(math.log(s),requires_grad=False,device=device)
		opt=torch.optim.Adam([w,t],lr=0.01)
	else:
		z=torch.tensor(0.,requires_grad=True,device=device)
		opt=torch.optim.Adam([w,t,z],lr=0.01)
	_loss=[]
	if algo=='gd':
		for it in range(iter):
			loss=0

			W=torch.softmax(w,0)
			q=torch.zeros((n,n))
			for i in range(r):
				q+=W[i]*torch.softmax(t[0][i],0)[:,None]*torch.softmax(t[1][i],0)[None,:]
			for i in range(n):
				for j in range(n):
					loss+=q[i,j]*f(p[i,j]/q[i,j])[0]

			opt.zero_grad()
			loss.backward()
			opt.step()
			if it%10==0 and LOG:
				print(f"it:{it}, loss:{loss.item()}, kl_div:{kl_div()}")
			if it>10 and abs(sum(_loss[-10:])/10-loss.item())<1e-3:
				break
			_loss.append(loss.item())
	elif algo=='sgd':
		batch_size=1
		for it in range(iter):
			loss=0
			loss_no_grad=0

			for _ in range(batch_size):
				W=torch.softmax(w,0)
				k=torch.multinomial(W,1)[0].item()
				i=torch.multinomial(torch.softmax(t[0][k],0),1)[0].item()
				j=torch.multinomial(torch.softmax(t[1][k],0),1)[0].item()
				q=0
				for _ in range(r):
					q+=W[_]*torch.softmax(t[0][_],0)[i]*torch.softmax(t[1][_],0)[j]
				P=torch.exp(-energe(i,j)-z)
				loss+=(f(P/q)[0]-P/q*f(P/q)[1]).detach()*torch.log(q)
				loss_no_grad+=f(P/q)[0]
				if not normalized:
					i=random.randint(0,n-1)
					j=random.randint(0,n-1)
					delta=-energe(i,j)+2*math.log(n)-z
					loss+=torch.exp(delta)-delta-1
					loss_no_grad+=torch.exp(delta)-delta-1

			opt.zero_grad()
			loss.backward()
			opt.step()
			if it%10==0 and LOG:
				print(f"it:{it}, loss:{loss_no_grad.item()}, kl_div:{kl_div()}")
			if it>10 and abs(sum(_loss[-10:])/10-loss_no_grad.item()/batch_size)<1e-3:
				break
			_loss.append(loss_no_grad.item()/batch_size)
	else:
		raise ValueError("Unknown algorithm")
	if LOG:
		print(f"iter:{it} final loss:{_loss[-1]}, kl_div:{kl_div()}")
	return np.array([it,_loss[-1],kl_div()])

LOG=False

n=16
r=4
algo='sgd'
energe=e1
normalized=False
for i in range(6):
	if i==1: r=10
	elif i==2: r=2
	else: r=4
	if i==3: algo='gd'
	else: algo='sgd'
	if i==4: normalized=True
	else: normalized=False
	if i==5: energe=e2
	else: energe=e1
	print("Experiment:",i+1)
	for f in [f1,f2,f3,f4]:
		res=np.zeros(3)
		for _ in range(16):
			res+=train(n,r,algo,energe,f,normalized=normalized,iter=1000000)
		print(res/16)
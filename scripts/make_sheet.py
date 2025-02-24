import sys
import re
ns=int(sys.argv[1])
nk=int(sys.argv[2])
s=[]
k=[]
I=3
for i in range(ns):
	s.append(int(sys.argv[I]))
	I+=1
for i in range(nk):
	k.append(int(sys.argv[I]))
	I+=1
print(ns,nk,s,k,file=sys.stderr)
ari=[[0]*ns]*nk
loss=[[0]*ns]*nk
nmi=[[0]*ns]*nk
for i in range(nk):
	for j in range(ns):
		pwd="logs/"+str(s[j])+"-"+str(k[i])+"-out.txt"
		data=open(pwd,"r").read()
		match = re.search(r"ARI: ([-\d.]+)", data)
		assert match, "ARI not found"
		ari[i][j]=float(match.group(1))
		match = re.findall(r"Loss: ([-\d.]+)", data)
		assert len(match), "Loss not found"
		loss[i][j]=float(match[-1])
		# print(match[-1],file=sys.stderr)
		match = re.search(r"NMI: ([-\d.]+)", data)
		assert match, "NMI not found"
		nmi[i][j]=float(match.group(1))
print("ARI")
for i in range(nk):
	for j in range(ns):
		print(ari[i][j],end=" ")
	print()
print("Loss")
for i in range(nk):
	for j in range(ns):
		print(loss[i][j],end=" ")
	print()
print("NMI")
for i in range(nk):
	for j in range(ns):
		print(nmi[i][j],end=" ")
	print()
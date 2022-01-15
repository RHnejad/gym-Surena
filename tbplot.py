import numpy as np 
import matplotlib.pyplot as plt

x=[]
y=[]

directory="xpos"

files=[]
import os
for file_name in os.listdir(directory):
    files.append(file_name)
files.sort()

import json
bias=0
for name in files:
    with open(directory+"/"+name) as json_file:
        print("*",name)
        data = json.load(json_file)
    
    n=len(data)
    print(n)

    for i in range(n):
        x.append(data[i][1]+bias)
        y.append(data[i][2])
    bias= x[-1]
    
N=len(y)

for j in range(0):
    Y=y.copy()
    for i in range( N-1):
        print("**********")
        print(Y[i+1])
        print(y[i+1])
        print(y[i])
        Y[i+1]=(y[i]+y[i+1])/2
        print(Y[i+1])
    y=Y.copy()
# print(x[-1])


fig = plt.figure(figsize=[14,5])
ax = fig.add_subplot(1, 1, 1)

major_ticks_x = np.arange(0, x[-1],int(x[-1]/20))
minor_ticks_x = np.arange(0, 4005888,100000)

major_ticks_y = np.arange(0, max(y)*1.02, int(max(y)*1.01/20))
minor_ticks_y = np.arange(0, 3000,10000)

ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
ax.set_xlim(0,x[-1])
ax.set_ylim(0,max(y)*1.01)
ax.set_xlabel("Time Step",fontsize = 17)
ax.set_ylabel("Episode Reward Mean",fontsize = 17)
plt.title("Position Control",fontsize = 14)

ax.grid(which='both')

# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)
from matplotlib import colors #as mcolors

ax.plot(x,y,linewidth=3,color="darkslateblue")#,".", markersize=8)#linewidths=10)

plt.savefig(directory+'.eps', format='eps')

plt.show()


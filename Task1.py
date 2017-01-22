import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *
from oslo import *

L = 32 
p = [0,1]

pile = oslo(L,p[0])
trans = int(1e3)
recur = int(1e3)

# def drop_grains(self,trans=0,recur=1000,from_zero=True):
pile.drop_grains(trans,recur,True)

h = pile.htotal 

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('time')
ax1.set_ylabel('total height')
ax1.plot(h)
print h[-1]

#######################################################################

pile = oslo(L,p[1])
trans = int(1e3)
recur = int(1e3)

# def drop_grains(self,trans=0,recur=1000,from_zero=True):
pile.drop_grains(trans,recur,True)

h = pile.htotal 

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('time')
ax2.set_ylabel('total height')
ax2.plot(h)
print h[-1]

plt.show()
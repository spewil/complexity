from __future__ import division
import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *
from oslo import *

#plot heights for p=0, p=0.5, and p=1

L = 32 
p = [0,.5,1]

trans = int(1e3)
recur = int(5e3)

pile0 = oslo(L,p[0])
# def drop_grains(self,trans=0,recur=1000,from_zero=True):
pile0.drop_grains(trans,recur,True)

pile5 = oslo(L,p[1])
# def drop_grains(self,trans=0,recur=1000,from_zero=True):
pile5.drop_grains(trans,recur,True)

pile1 = oslo(L,p[2])
# def drop_grains(self,trans=0,recur=1000,from_zero=True):
pile1.drop_grains(trans,recur,True)

h0 = pile0.htotal 
h1 = pile1.htotal 
h5 = pile5.htotal

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('grains dropped')
ax1.set_ylabel('total height')
ax1.plot(h0)
ax1.plot(h1)
ax1.plot(h5)
print 'final heights are: ' + str(h0[-1]) + ', ' + str(h5[-1]) + ', '  + 'and ' + str(h1[-1])

###################################

#histogram of avalanches for p=0.5

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('grains dropped')
ax2.set_ylabel('avalanche probability P(s)')
# xlim changes what the plot shows 
ax2.set_xlim([trans,trans+recur])
# change to only plot the recurrent sizes sizes[recur:-1]
ax2.plot(pile5.sizes)

#log_binned plot of sizes 

b, c = log_bin(pile5.sizes, a = 1.7)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_xlabel('grains dropped')
ax3.set_ylabel('avalanche size')
ax3.loglog(b, c, 'r-')

plt.show()
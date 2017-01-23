import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *
from oslo import *

## model params 
# L values 
L = [2**x for x in range(4,9)]
#binomial probability 
p = 0.5

# set up three figures 
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('time')
ax1.set_ylabel('total height')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('centres')
ax2.set_ylabel('counts')
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_xlabel('system size L')
ax3.set_ylabel('crossover time')

# list of piles 
piles = []
for i in range(len(L)):
	piles.append(oslo(L[i],p))

# time variables 
trans = 1e3
recur = 1e4

crossovers = []
for pile in piles:
# 	drop_grains(self,trans=500,recur=1000,from_zero=True,draw=False):
	pile.drop_grains(trans,recur,True,False)
	s = pile.sizes
	h = pile.htotal 
	if hasattr(pile,'crossover'):
		crossovers.append(pile.crossover)
	ax1.plot(h)	
	if pile == piles[0]:
		vals, counts = lin_bin(s, int(max(s)))
		b, c = log_bin(s, 1., 1.5, 2)
		ax2.loglog(vals, counts, 'bx')
		ax2.loglog(b, c, 'r-')

ax3.plot(L,crossovers)
plt.show()


# sort the avalanche list 
# plot and hope to find power law structure 
# s = np.sort(s)
# s = np.flipud(s) #only works for 1D arrays 
# do log binning
# b, c = log_bin(data, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):

# # fitting 
# coefficients = np.polyfit(b, c, 1)
# polynomial = np.poly1d(coefficients)
# ys = polynomial(b)
# ax2.plot(b, ys)


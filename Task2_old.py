from __future__ import division
import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *
from oslo import *
import json

def moving_mean(x,W):
    # x is 1D
    # window size W 
    for i in range(len(x)):
        try: 
            x[i] = sum(x[(i-W):(i+W)])/len(x[(i-W):(i+W)]) 
        except:
            if i-W < 0: 
                x[i] = sum(x[0:(i+W)])/len(x[0:(i+W)])
            elif i+W > len(x): 
                x[i] = sum(x[(i-W):len(x)])/(len(x[(i-W):len(x)]))
    return x 


###PLOTTING###
###########################

# set up three figures 

# heights, moving average 
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('time')
ax1.set_ylabel('total height')

# probability 
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('centres')
ax2.set_ylabel('counts')

# crossover time
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_xlabel('system size L')
ax3.set_ylabel('crossover time')

###########################

## model params 
# L values 
L = [2**x for x in range(4,9)]
#binomial probability 
p = 0.5

# list of piles, oslo objects 
piles = []
for i in range(len(L)):
	piles.append(oslo(L[i],p))

# time variables 
trans = 1e3
recur = 1e3

def plot_data():
	s_data = []
	h_data = []
	crossovers = []
	for pile in piles:
	# 	drop_grains(self,trans=500,recur=1000,from_zero=True,draw=False):
		pile.drop_grains(trans,recur,False,False)
		s = pile.sizes
		h = pile.htotal

		ax1.plot(h)	
		ax1.plot(moving_mean(pile.htotal,50),'r-')

		# vals, counts = lin_bin(s, int(max(s)))
		b, c = log_bin(s, a = 1.5)
		# ax2.loglog(vals, counts, 'bx')
		ax2.loglog(b, c, 'r-')

		# save data for later plotting, etc
		sj = json.dumps(s.tolist())
		hj = json.dumps(h.tolist()) 
		s_data = s_data.append(sj)
		h_data = h_data.append(hj)

		a = np.arange(10).reshape(2,5) # a 2 by 5 array
		b = a.tolist() # nested lists with same data, indices
		file_path = "/file_" + str(i) + ".json"
		file_path = "/file_%s.json" % i
		file_path = "/file_{}.json".format(i)
		file_path = "/path.json" ## your path variable
		### this saves the array in .json format
		json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) 

		if hasattr(pile,'crossover'):
			crossovers.append(pile.crossover)

	ax3.plot(L,crossovers)

	plt.show()

if __name__ == '__main__':
	plot_data()

# # load data 
# json.loads(filename)
# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)

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




from __future__ import division
import numpy as np  
from scipy.optimize import fmin
import json 
import matplotlib.pyplot as plt
from log_bin import *
from oslo import *


## global model params 
# L values 
L = [2**x for x in range(3,9)]
#binomial probability 
p = 0.5


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

def collect_data():
	# time variables 
	trans = 1e3	
	recur = 1e5

	crossovers = []

	for i in range(len(L)):
		
		# make a pile 
		pile = oslo(L[i],p)
		# drop_grains(self,trans=500,recur=1000,from_zero=True,draw=False):
		pile.drop_grains(trans,recur,True,False)
		
		# save data for later plotting, etc
		sj = pile.sizes.tolist()
		hj = pile.htotal.tolist()

		file_path_s = 'data/avalanche_' + str(L[i]) + '.json'
		file_path_h = 'data/height_total_' + str(L[i]) + '.json'

		with open(file_path_s, 'w') as fp:
			json.dump(sj, fp) 
		with open(file_path_h, 'w') as fp:
			json.dump(hj, fp) 

		if hasattr(pile,'crossover'):
			crossovers.append(pile.crossover)

	file_path_c = 'data/crossovers.json'

	with open(file_path_c, 'w') as fp:
			json.dump(crossovers, fp) 

def plot_data():

		### SET UP ###

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

		### IMPORT ###

	for i in range(len(L)):		
		file_path_s = 'data/avalanche_' + str(L[i]) + '.json'
		file_path_h = 'data/height_total_' + str(L[i]) + '.json'
		with open(file_path_s) as fp:
			s = np.array(json.load(fp))
		with open(file_path_h) as fp:
			h = np.array(json.load(fp))

		###PLOTTING###

		ax1.plot(h)	
		ax1.plot(moving_mean(h,50),'r-')

		# vals, counts = lin_bin(s, int(max(s)))
		b, c = log_bin(s, a = 1.5)
		# ax2.loglog(vals, counts, 'bx')
		ax2.loglog(b, c, 'r-')
	
	# fig1.savefig('figs/heights_raw.png')
	# fig2.savefig('figs/avalanches_raw.png')
	
	file_path_c = 'data/crossovers.json'
	with open(file_path_c) as fp:
		c = np.array(json.load(fp))
	
	print c

 	ax3.plot(L,c,'ro')

	# # fitting 
	print L 

	# def func(a, XX):
	#     yfit = [a*(x**2) for x in XX]
	#     print yfit
	#     return yfit
	# def objective(a):
	#     'function to minimize'
	#     SSE = np.sum((c - func(a, L))**2)
	#     print SSE
	#     return SSE

	# a_fit = fmin(objective, 1)
	# x = np.linspace(0, 128)
	# ax3.plot(x, func(a_fit, x))

	# fitting 
	coefficients = np.polyfit(L, c, 2)
	coefficients[-2:] = [0]*2 # set linear and constants to 0 
	polynomial = np.poly1d(coefficients)
	ys = polynomial(range(256))
	ax3.plot(range(256), ys)

	# print a_fit

	plt.show()

	# fig3.savefig('figs/crossoversvtime.png')

if __name__ == '__main__':
	collect_data()
	plot_data()

# sort the avalanche list 
# plot and hope to find power law structure 
# s = np.sort(s)
# s = np.flipud(s) #only works for 1D arrays 
# do log binning
# b, c = log_bin(data, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):





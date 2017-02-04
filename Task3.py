from __future__ import division
import numpy as np  
from scipy.optimize import fmin
import json 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from log_bin import *
from oslo import *

def plot_data(save=True):


######### SET UP FIGURES ##########

	# probability P(s,L)
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	ax2.set_xlabel('centres')
	ax2.set_ylabel('counts')	
	

########## IMPORT DATA ############

	file_path_s = 'data/avalanche_' + str(L) + '.json'
	file_path_h = 'data/height_total_' + str(L) + '.json'
	with open(file_path_s) as fp:
		s = np.array(json.load(fp))
	with open(file_path_h) as fp:
		h = np.array(json.load(fp))

	#### (a) 

	#### P(s,L)
	vals, counts = lin_bin(s, int(max(s)))
	ax2.loglog(vals, counts, 'bx')
	centres, counts = log_bin(s, a = 1.5)
	ax2.loglog(centres, counts, 'r-')

	plt.show()

if __name__ == '__main__':

	## global model params 
	# L values 
	L = 2**8 #[2**x for x in range(3,8)]
	#binomial probability 
	p = 0.5

	# time variables 
	trans = 1e3	
	recur = 1e5

	time = trans+recur
	# collect_data()
	plot_data(save=True)
from __future__ import division
import numpy as np  
from scipy.optimize import fmin
import json 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from log_bin import *
from oslo import *

def moving_mean(x,W):
    # x is 1D
    # window size W 
    y = np.array(x) # leave x untouched 
    for i in range(len(y)):
        try: 
            y[i] = sum(y[(i-W):(i+W)])/len(y[(i-W):(i+W)]) 
        except:
            if i-W < 0: 
                y[i] = sum(y[0:(i+W)])/len(y[0:(i+W)])
            elif i+W > len(y): 
                y[i] = sum(y[(i-W):len(y)])/(len(y[(i-W):len(y)]))
    return y 

def collect_data():

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

def plot_data(save=True):

	print L 
	print pows 

		### SET UP ###

	# heights, moving average 
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1.set_xlabel('time, grains dropped')
	ax1.set_ylabel('total height')

	# crossover time
	fig3 = plt.figure()
	ax3 = fig3.add_subplot(111)
	ax3.set_xlabel('system size L')
	ax3.set_ylabel('crossover time')

	# mean recurrent height
	fig4 = plt.figure()
	ax4 = fig4.add_subplot(111)
	ax4.set_xlabel('system size L')
	ax4.set_ylabel('mean recurrent height')

	# height scaling 
	fig5 = plt.figure()
	ax5 = fig5.add_subplot(111)
	ax5.set_xlabel('scaled time [t/L**2]')
	ax5.set_ylabel('scaled height')

	# mean recurrent height
	fig6 = plt.figure()
	ax6 = fig6.add_subplot(111)
	ax6.set_xlabel('system size L')
	ax6.set_ylabel('correction to scaling')

	# probability P(h,L)
	fig7 = plt.figure()
	ax7 = fig7.add_subplot(111)
	ax7.set_xlabel('bin')
	ax7.set_ylabel('density')

	# probability P(h,L)
	fig8 = plt.figure()
	ax8 = fig8.add_subplot(111)
	ax8.set_xlabel('scaled bins')
	ax8.set_ylabel('scaled density')

		### IMPORT ###

	# t_c scaling 
	file_path_c = 'data/crossovers.json'
	with open(file_path_c) as fp:
		c = np.array(json.load(fp))

	# update c for only the powers we want 
	cc = [c[pow-1] for pow in pows]
	
	# fitting 
	coeff_t = np.polynomial.polynomial.polyfit(L, cc, [2])
	coeff_t = coeff_t[::-1] 
	# print 't_c scaling coeffs: ' + str(coeff_t)
	polynomial = np.poly1d(coeff_t)
	ys = polynomial(range(256))
	# add plots 
	ax3.plot(L,cc,'ro',zorder=10)
	ax3.plot(range(256), ys)

	# will save the mean recurrent heights as a list
	h_means = []
	h_stddevs = []

	log_hL = [] 

	for i in range(len(L)):		
		file_path_s = 'data/avalanche_' + str(L[i]) + '.json'
		file_path_h = 'data/height_total_' + str(L[i]) + '.json'
		with open(file_path_s) as fp:
			s = np.array(json.load(fp))
		with open(file_path_h) as fp:
			h = np.array(json.load(fp))

		###PLOTTING###
		
		# ax1.plot(h)	
		# use a window that grows with system size (looks nicer)
		h_smooth = moving_mean(h,2*L[i])
		#plot the smoothed heights 
		ax1.plot(h_smooth)

		#grab and store the mean of the steady-state 
		#recurrent height average of t0:end steps 
		# t0 = tc + 200
		h_recur = np.array(h[cc[i]+200:]) 
		
		h_mean = np.sum(h_recur)/len(h_recur)
		h_sq_mean = np.sum(h_recur**2)/len(h_recur)
		# stddev for the height 
		h_stddev = np.sqrt(h_sq_mean - h_mean**2)

	 	#lists for each system
	 	h_means.append(h_mean)
	 	h_stddevs.append(h_stddev)

	 	# scale the height axis, normalize by L
	 	# take height 1k past the crossover time
	 	timespan = cc[i]+1000
	 	h_scale = h_smooth[0:timespan] / L[i] #multiply by 1/L to normalize height
	 	scaled_time = np.array(range(timespan)) / L[i]**2 #multiply by 1/L^2 to scale crossover time
	 	ax5.plot(scaled_time,h_scale)

	 	# ax5.set_xlim([0,3])

	 	# HEIGHT PROBABILITY  P(h,L)
	 	# using bincount 
		# freq = np.bincount(h_recur.astype(int))
		# prob = freq/float(len(h_recur))
		# print sum(prob)
		# ax7.plot(prob)

#######################################

		# bins 
		# go 10 above max for a nice gaussian
		bin_array=range(int(h_mean)-10, int(h_mean)+10)	 	
	 	P_of_hL = []
		for k in range(len(bin_array)):
			# True/False masking for the heights, if it is height n, make True
		    mask = (h_recur == bin_array[k]) 
		    # normalize by the number of discrete probabilities 
		    P_of_hL.append(len(h_recur[mask])/len(h_recur))

		# check normalization
		# print np.sum(P_of_hL)

		# P_of_hL is a list of probabilities 
		ax7.plot(bin_array,P_of_hL)

		hs = (h_recur - h_mean)/h_stddev 
		bin_array_s = (np.array(range(L[i],2*L[i]+1)) - h_mean)/h_stddev

		# using the hist function: 
		# increase the number of bins with system size 
		ax8.hist(hs, normed= True, bins=bin_array_s, histtype= 'step', range=[-6,6], align='left')
		x = np.linspace(-3, 3, 100)
		ax8.plot(x, mlab.normpdf(x, 0, 1))
		
		# if L[i] > 8:

		# 	scaled_P_of_hL = []
		# 	# scaling the bins 
		# 	# num bins ~ system size 
		# 	# bin_array_s = np.linspace(hs_mean-10,hs_mean+10,L[i])
		# 	#(bin_array - h_mean)/stddev 
		# 	for j in range(len(bin_array_s)-1):
		# 		# True/False masking for the heights, between
		# 	    mask = (hs >= bin_array_s[j])&(hs < bin_array_s[j+1])
		# 	    # normalize by the bin array
		# 	    scaled_P_of_hL.append(len(hs[mask])/len(hs))
		# 	# check normalization
		# 	print np.sum(scaled_P_of_hL)
		# 	# print h_mean
		# 	# print str(np.mean(hs)) + '\n'
		# 	ax8.plot(bin_array_s[:-1],scaled_P_of_hL)

		################## finding corrections 


	# fig1.savefig('writeup/figs/heights_raw.png')
	# fig2.savefig('writeup/figs/avalanches_raw.png'


	# first approximation a0 using 128 256
	# slope of height vs system size L 
	a0 = h_means[-1]/L[-1] # (h_means[-1] - h_means[-2])/(L[-1] - L[-2])
	a0_first = a0 
	print a0
	err = 1e5
	step = 1e-3
	LogL = np.log(L)

	# figuring out a0 iteratively by minimizing the second order fit parameter 
	
	# make a list of a0's we want to look at
	# start at one or the last value for the first iteration will be zero 
	# (it's exactly our guess) 
	a_trials = [x*step + a0 for x in range(1,51)]


	for a in a_trials:	

		Log_hOverLa0 = np.log(1 - (np.array(h_means)/(np.array(L)*a)))
		print 'current a0 iteration: ' + str(a) 
		# print 1 - (np.array(h_means)/(np.array(L)*a))

		# fit it quadratically  
		coeff_a = np.polyfit(LogL,Log_hOverLa0, 2)
		# print coeff_a

		# plot it 
		if a == a0_first + step: 
			ax4.plot(LogL,Log_hOverLa0, 'r')
		else:
			ax4.plot(LogL,Log_hOverLa0, 'b')

		# take the magnitude of the second derivative
		# we want to minimize this! 
		check = np.absolute(coeff_a[0])
		# highest power first 
		if check < err:
			# update quadratic "error"
			err = check 
			# update our choice of a0 with the quadratically-minimized version 
			# (the current iteration)
			a0 = a
			print 'current error: ' + str(err)

	ax4.set_xlabel('log(L)')
	ax4.set_ylabel('log(1 - h/La0')

	# add plots 
	# ax4.plot(L, h_means,'bo',zorder=10)

	# height scaling 
	# for t<<tc, scales ~ sqrt(t)! 
	# up to some constant 
	t = np.linspace(0,2.5,100) 
	h_scaling_less_tc = 1.8*np.sqrt(t)
	ax5.plot(t,h_scaling_less_tc,'r--')
	ax5.set_xlim([0,2.5])
	ax5.set_ylim([0,2.5])
	# add the constant 
	ax5.plot(t,np.ones(len(t))*a0,'r--')

	ax6.plot(L,[h_means[i]/L[i] for i in range(len(L))]) 

	# scale transformation for data collapse 
	# handles,labels = ax5.get_legend_handles_labels()
	
	labels = [str(x) for x in L]
	ax5.legend(labels,loc=4)

	# plot height standard deviation vs L 
	fig9 = plt.figure()
	ax9 = fig9.add_subplot(111)
	ax9.plot(np.log(L),np.log(h_stddevs),'ro')
	# fitting 
	coeff_std = np.polyfit(np.log(L), np.log(h_stddevs), 1)
	# coeff_std = coeff_std[::-1] 
	# print 'height scaling coeffs: ' + str(coeff_h)
	polynomial = np.poly1d(coeff_std)
	ystd = polynomial(np.log(L))
	# add plots 
	ax9.plot(np.log(L), ystd,'b--')
	ax9.plot(np.log(L),0.241+np.log(L)*0.5411)
	ax9.set_xlabel('log(system size L)')
	ax9.set_ylabel('log(height standard deviation)')
	# how does this scale? 


	# plot height standard deviation vs L 
	fig10 = plt.figure()
	ax10 = fig10.add_subplot(111)
	ax10.plot(np.log(L),np.log(1 - (np.array(h_means)/(np.array(L)*(a0)))))
	ax10.set_xlabel('log(system size L)')
	ax10.set_ylabel('log(1 - <h>/a0*L)')

	if save == True:
		fig1.savefig('writeup/figs/heightsvtime.png')

		fig3.savefig('writeup/figs/crossovertime.png')
		fig4.savefig('writeup/figs/meanheights.png')
		fig5.savefig('writeup/figs/heightcollapse.png')
		fig6.savefig('writeup/figs/correctiontoscaling.png')
		fig7.savefig('writeup/figs/heightprob.png')
		fig8.savefig('writeup/figs/heightprobscaled.png')
		fig9.savefig('writeup/figs/heightstandarddev.png')

	plt.show()

if __name__ == '__main__':

	## global model params 
	# L values 
	pows = range(3,9)
	L = [2**x for x in pows]
	#binomial probability 
	p = 0.5

	# time variables 
	trans = 0	
	recur = 1e5

	time = trans+recur
	# collect_data()
	plot_data(save=True)

# sort the avalanche list 
# plot and hope to find power law structure 
# s = np.sort(s)
# s = np.flipud(s) #only works for 1D arrays 
# do log binning
# b, c = log_bin(data, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):





#Complexity Project 1/17 - 2/17 
#usr/bin/python 

import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *

class oslo(): 
    def __init__(self, size):
        self.size = size 
        self.heights = np.zeros(size)
        self.threshes = np.array([np.random.randint(1,3) for x in self.heights]) 
    
    def drive(self): 
        # add rice grain to left hand side 
        self.heights[0] += 1 
    
    def relax(self):
        # shift the heights over by 1 
        shifted = np.roll(self.heights,-1) 
        # off the cliff is 0 
        shifted[-1] = 0 
        # slope is difference between left and right 
        slopes = self.heights - shifted
        # true if above threshold height 
        overthresh = slopes > self.threshes
        # while any are true, should be updated 
        s = 0 
        while np.sum(overthresh) > 0:  
            # move through each element of slopes
            for i in np.arange(self.size):    
                # if greater than threshold 
                if slopes[i] > self.threshes[i]:
                    # change the height of that one 
                    self.heights[i] -= 1
                    # if it's not the last element 
                    if i != self.size - 1:
                        self.heights[i+1] += 1
                    # update the threshold value of ith position 
                    self.threshes[i] = np.random.randint(1,3)
                    s += 1
            # add moves from for loop to total avalanche size for relaxation
            #update slopes and booleans 
            slopes = self.heights - shifted
            overthresh = slopes > self.threshes
        return s 

    def draw(self):
        for height in self.heights:
            print '#' * np.int(height) 

L = 30
pile = oslo(L)
# while True:
#     pile.drive()
#     pile.relax()
#     pile.draw()
#     raw_input()

transients = 1025
recurrent = int(1e5)
total = transients + recurrent
s_list = np.array([])
for i in range(1,transients):
    pile.drive()
    pile.relax()
for k in range(1,recurrent):
    pile.drive()
    s_curr = pile.relax()
    s_list = np.append(s_list, s_curr)

# print s_list

# sort the avalanche list 
# plot and hope to find power law structure 
s_sort = np.sort(s_list)
s_sort = np.flipud(s_sort) #only works for 1D arrays 
print np.max(s_sort)
print len(s_sort)

# do log binning
b, c = log_bin(s_sort) #, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):
# b, c = log_bin(x, 1., 1.5, a, debug_mode=True)
# plt.loglog(vals, counts, 'bx')
plt.loglog(b, c, 'r-')
plt.show() 
    
# plt.xlabel("Log (Rank of frequency)")
# plt.ylabel("Log (Frequency)")
# plt.title("Frequency vs frequency rank for words")
# plt.plot(s_sort)
# plt.ylabel('avalanche size')

# coefficients = numpy.polyfit(logB, logA, 1)
# polynomial = numpy.poly1d(coefficients)
# ys = polynomial(b)
# plt.plot(b, ys)

# plt.show()






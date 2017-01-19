#Complexity Project 1/17 - 2/17 
#usr/bin/python 

import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt

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
            s += s
            #update slopes and booleans 
            slopes = self.heights - shifted
            overthresh = slopes > self.threshes
        return s 

    def draw(self):
        for height in self.heights:
            print '#' * np.int(height) 

L = 128
pile = oslo(L)
# while True:
#     pile.drive()
#     pile.relax()
#     pile.draw()
#     raw_input()

transients = 1025
recurrent = 257
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
s_sort = np.flipud(s_sort)
# print s_sort
plt.loglog(s_sort)
plt.ylabel('avalanche size')


# a = numpy.asarray(a, dtype=float)
# b = numpy.asarray(b, dtype=float)
# Now you can perform operations on them. What the loglog-plot does, is to take the logarithm to base 10 of both a and b. You can do the same by

# logA = numpy.log10(a)
# logB = numpy.log10(b)
# This is what the loglog plot visualizes. Check this by ploting both logA and logB as a regular plot. Repeat the linear fit on the log data and plot your line in the same plot as the logA, logB data.

# coefficients = numpy.polyfit(logB, logA, 1)
# polynomial = numpy.poly1d(coefficients)
# ys = polynomial(b)
# plt.plot(logB, logA)
# plt.plot(b, ys)

# plt.loglog(b,a,'ro')
# plt.plot(b,ys)
# plt.xlabel("Log (Rank of frequency)")
# plt.ylabel("Log (Frequency)")
# plt.title("Frequency vs frequency rank for words")
# plt.show()

print np.max(s_sort)
plt.show()






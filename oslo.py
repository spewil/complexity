#Complexity Project 1/17 - 2/17 
#usr/bin/python 

import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *

class oslo(): 
    def __init__(self, size, p):
        self.p = p
        self.size = size 
        self.heights = np.zeros(size)
        self.threshes = np.array([np.random.binomial(1,p,None) + 1 for x in self.heights]) 
    
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
        return s, self.heights

    def draw(self):
        for height in self.heights:
            print '#' * np.int(height)






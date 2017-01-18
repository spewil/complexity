#Complexity Project 1/17 - 2/17 
#usr/bin/python 

import numpy as np  
import scipy as sp
import random as rd 

class oslo(): 
    def __init__(self, size):
        self.size = size 
        self.heights = np.zeros(size)
        self.threshes = np.array([rd.randint(1,2) for x in self.heights]) 
    
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
        while np.sum(overthresh) > 0:  
            # move through each element of slopes
            i = 0
            for slope in slopes:
                # if greater than threshold 
                if slope > self.threshes[i]:
                    # change the height of that one 
                    self.heights[i] -= 1
                    # if it's not the last element 
                    if i != self.heights.size:
                        print self.heights.size 
                        self.heights[i+1] += 1
                i += 1
            #update truefalse 
            slopes = self.heights - shifted
            overthresh = slopes > self.threshes
            return self.heights

L = 8
pile = oslo(L)
pile.drive()
pile.relax()
raw_input()




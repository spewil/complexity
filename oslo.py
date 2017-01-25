#Complexity Project 1/17 - 2/17 
#usr/bin/python 
from __future__ import division
import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *

class oslo(): 
    def __init__(self, size, p):
        self.p = p
        self.size = size 
        self.heights = np.zeros(size)
        self.threshes = np.array([np.random.binomial(1,self.p,None) + 1 for x in self.heights]) 
        self.slopes = np.zeros(size)
        self.crossover = None # temp value for crossover time 

    def drive(self): 
        # add rice grain to left hand side 
        self.heights[0] += 1 
    
    def relax(self):
        # shift the heights over by 1 
        shifted = np.roll(self.heights,-1) 
        # off the cliff is 0 
        shifted[-1] = 0 
        # slope is difference between left and right 
        self.slopes = self.heights - shifted
        # true if above threshold height 
        overthresh = self.slopes > self.threshes
        # while any are true, should be updated 
        s = 0 
        while np.sum(overthresh) != 0:  
            # move through each element of slopes
            for i in np.arange(self.size):    
                # if greater than threshold 
                if self.slopes[i] > self.threshes[i]:
                    # size of the avalanche goes up 1
                    s += 1
                    
                    # always decrement the height of current 
                    self.heights[i] -= 1
                    
                    # if it's not the last element 
                    if i != self.size - 1: 
                        # topple to the next 
                        self.heights[i+1] += 1

                    # update the threshold value of ith position 
                    self.threshes[i] = np.random.binomial(1,self.p,None) + 1
            
            # add moves from for loop to total avalanche size for relaxation
            #update slopes and booleans 
            shifted = np.roll(self.heights,-1)
            shifted[-1] = 0 
            self.slopes = self.heights - shifted
            overthresh = self.slopes > self.threshes
        return s, self.heights[0]

    def draw(self):
        for height in self.heights:
            print '#' * np.int(height)

    def drop_grains(self,trans=500,recur=1000,from_zero=True,draw=False):
        time = int(trans + recur)
        self.sizes = np.array([])
        self.htotal = np.array([])
        if from_zero:
            for i in range(1,time+1):
                self.drive()
                s_curr, h0 = self.relax()
                self.htotal = np.append(self.htotal, h0)
                self.sizes = np.append(self.sizes, s_curr)

                if draw:
                    self.draw()
                    raw_input()
        else:
            # get the transients out of the way 
            for i in range(1,int(trans)+1):
                self.drive()
                self.relax()
            # store the total heights and list of avalanches 
            for k in range(1,int(recur)+1):
                self.drive()
                s_curr, h0 = self.relax()
                self.htotal = np.append(self.htotal, h0)
                self.sizes = np.append(self.sizes, s_curr)

                if draw:
                    self.draw()
                    raw_input()









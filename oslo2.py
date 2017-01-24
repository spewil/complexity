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
            self.heights -= overthresh
            overthresh_shift = np.roll(overthresh,1)
            overthresh_shift[0] = 0 
            self.heights += overthresh_shift

            for i in np.arange(self.size):    
                if overthresh[i] == 1:
                    self.threshes[i] = np.random.binomial(1,self.p,None) + 1 
            s += np.sum(overthresh)

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
        self.crossover = None # temp value for crossover time 
        if from_zero:
            for i in range(1,time+1):
                self.drive()
                s_curr, h0 = self.relax()
                self.htotal = np.append(self.htotal, h0)
                self.sizes = np.append(self.sizes, s_curr)
                #when the last slot fills for the first time
                if self.heights[-1] != 0 and self.crossover == None:
                    self.crossover = i 
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
                #when the last slot fills for the first time 
                if self.heights[-1] != 0 and self.crossover == None:
                    self.crossover = i
                if draw:
                    self.draw()
                    raw_input()       








import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *
from oslo import *

L = 32 
p = [0,1]

pile = oslo(L,0.5)
trans = int(1e3)
recur = int(1e3)

# def drop_grains(self,trans=500,recur=1000,from_zero=True,draw=False):
pile.drop_grains(trans,recur,True,True)


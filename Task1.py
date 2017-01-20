import numpy as np  
import scipy as sp
import matplotlib.pyplot as plt
from log_bin import *
from Oslo import *

h= 100

main()

L = 30
pile = oslo(L,0.3)
# while True:
#     pile.drive()
#     pile.relax()
#     pile.draw()
#     raw_input()

transients = 1025
recurrent = int(1e4)
total = transients + recurrent
s_list = np.array([])
for i in range(1,transients):
    pile.drive()
    pile.relax()
for k in range(1,recurrent):
    pile.drive()
    s_curr, heights = pile.relax()
    s_list = np.append(s_list, s_curr)

# print s_list

# sort the avalanche list 
# plot and hope to find power law structure 
s_sort = np.sort(s_list)
s_sort = np.flipud(s_sort) #only works for 1D arrays 
print np.max(s_sort)
print len(s_sort)

# do log binning
b, c = log_bin(s_list,a=1.2) #, bin_start=1., first_bin_width=1., a=2., datatype='float', drop_zeros=True, debug_mode=False):
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
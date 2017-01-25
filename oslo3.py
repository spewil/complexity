"""
Creator: Angel Joaniquet Tukiainen
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit


class Model:
    """ Frame of the Oslo model. It consists of one integer of the height
        of the first position of the model, and 2 arrays, with the slope and slope
        threshold of the position. The height can be reconstructed from
        the height of the first position and the consecutive slopes
    """

    def __init__(self, ll, p):
        """
        Initializates an instace of the class Model
        """
        self.P = p          # probability of threshold 1 (from the initialization)
        self.L = ll         # length of the array (from the initialization)

        self.T = 0          # Time of the system is stored here. (Evolves)
        self.height = 0     # height at point i = 0 (Evolves)
        self.slope = np.array([0]*ll)             # array with slopes at every point (Evolves)
        self.thresh = np.random.binomial(1, p, ll) + 1  # array with the slope thresholds (Evolves)
        #self.heights = [0]  # List with the heights over time (Deprecated)

    def display_height(self):
        """ Generates the list of heights from the heights at i=0 and the slopes
            returns: a list of length l withtthe values of the heights
        """

        heigs = [self.height]*self.L
        diff = self.slope[0]        # Accumulates the slope until i to substract
        for k in range(1, self.L):  # to the height=height(i=0) at every point y
            heigs[k] -= diff
            diff += self.slope[k]
        return heigs

    """
    (DEPRECATED)
    def smooth_heights(self, w):
         Smooths the heights of the model with the formula

                sh(i,t) = sum_{j=-w}^{w} h(j,t) for i in [w,L-w]  [[1]]

        :param w:
        :return: list of size self.L - 2W: [W,L-W] with the smoothed heigns sh(i,t)

        smo_hei = []
        heis = self.display_height()    # Stores the heights for use
        for l in range(w, self.L - w):  # Application of the smoother formula [[1]]
            heig = 0
            for k in range(k - w, k + w + 1):
                heig += heis[k]
            smo_hei.append(hei)
        return smo_hei
    """

    def drive(self):
        """ Updates the slope and the height of the first point, as well as incrementing
            the time.
        """

        self.slope[0] += 1
        self.height += 1
        self.T += 1

    def relax(self):
        """ Aplies one time step of the algorithm of the Oslo method, cheking if the pile is stable, ie.
            there is no slope bigger that the threshold, and if there is, it aplies the adequate
            transition rule for the position. Updates the slopes and the height of the first point.
            Also ads the rest height of the pile to the height list.

        :return: quantity of rice that exits in this relaxation
        """

        output = 0  # Number of rices that exits.
        while (self.slope > self.thresh).any():  # Checks if the pile is at rest to kill the loop
            for k in range(self.L):
                if self.slope[k] > self.thresh[k]:  # checks threshold
                    self.thresh[k] = np.random.binomial(1, self.P, 1) + 1  # Resetels the threshold after fall
                    if k == 0:              # Step for the 0th position, we check the
                        self.slope[0] -= 2  # qualitative position  to apply the correct rule
                        self.height -= 1
                        self.slope[1] += 1

                    elif k == (self.L-1):   # Step for the  L position
                        self.slope[k] -= 1
                        self.slope[k-1] += 1
                        output += 1

                    else:
                        self.slope[k] -= 2  # 0 < k-th < L position
                        self.slope[k-1] += 1
                        self.slope[k+1] += 1
        return output

    def draw(self):
        """ Draws the pile in the terminal at the curent state. Also returns information of the points:
            XX Y Z,
            with XX : height(i), Y : slope(i), Z : slope threshold(i)
        """

        heigs = self.display_height()
        k = 0
        for heig in heigs:
            print "{}".format(heig).zfill(2) + ", {}, {} :".format(self.slope[j], self.thresh[j]) + "o"*heig
            k += 1

"""
    To the main program
"""

if __name__ == "__main__":

    if sys.argv[1] == "1a":
        """ Watch the system evolve after every drive and relaxation
            1b L p
            prints an ascii graph of the pile.
        """
        print "press n to quit, any other to drive."
        oslo = Model(int(sys.argv[2]), float(sys.argv[3]))
        print oslo.display_height()
        print oslo.slope
        print oslo.thresh
        if raw_input("Start? [Y]es/[n]o") == "n":
            quit()
        while True:
            oslo.draw()
            oslo.drive()
            print oslo.relax()
            if raw_input() == 'n':
                break

    if sys.argv[1] == "1b":
        """ Test the speed of execution until the steady state one model
             1b L p
             prints execution time
        """
        start = time.time()
        oslo = Model(int(sys.argv[2]), float(sys.argv[3]))
        drops = [0]*11
        while oslo.slope[-1] == 0:
            oslo.drive()
            oslo.relax()

        for i in range(oslo.L*3):
            oslo.drive()
            drops = drops[1:10]
            drops.append(oslo.relax())

        while True:
            oslo.drive()
            drops = drops[1:10]
            drops.append(oslo.relax())
            if sum(drops[1:11])/10 == 1:
                break
        end = time.time()
        print(end - start)
        oslo.draw()
        print drops, oslo.T, oslo.height

    if sys.argv[1] == "2a":
        """
        Plot of the height vs time of L in a,
        and records the crossing time and crossing/ average stationary height of different systems and plots them
        """
        a = [16, 32, 64, 128, 256]
        HEI = []
        TIM = []
        for i in a:
            if raw_input("Next? [n]o to quit") == "n":
                quit()
            print "Plotting L = {}...".format(i)
            oslo = Model(i, float(sys.argv[2]))
            heights = []
            drops = [0]*11
            while oslo.slope[-1] == 0:
                oslo.drive()
                oslo.relax()
                heights.append(oslo.height)
            TIM.append(oslo.T)  # We take t_c to be when the particles arrive to the las position.
                                # TRUE that it will be too low
            for j in range(oslo.L * 3):
                oslo.drive()
                drops = drops[1:10]
                drops.append(oslo.relax())
                heights.append(oslo.height)

            while True:
                oslo.drive()
                drops = drops[1:10]
                drops.append(oslo.relax())
                heights.append(oslo.height)
                if sum(drops[1:11]) / 10 == 1:
                    break
            HEI.append(sum(heights[-10:])/10.)

            plt.plot(heights, '*')
            plt.show()
        b = TIM
        c = HEI
        fig, ax1 = plt.subplots()
        ax1.plot(a, b, 'b*')
        ax1.set_xlabel('Size (L) of the model')
        ax1.set_ylabel('Time', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(a, c, 'ro', color='r')
        ax2.set_ylabel('Height', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        plt.show()
        quit()

    if sys.argv[1] == "2b":
        """
          Plots of the smoothed height vs time of L in a,
        """
        a = [64, 128, 256]
        w = int(sys.argv[3])
        for i in a:
            if raw_input("Next? [n]o to quit") == "n":
                quit()
            print "Plotting L = {}...".format(i)
            oslo = Model(i, float(sys.argv[2]))
            heights = []
            drops = [0] * 11
            while oslo.slope[-1] == 0:
                oslo.drive()
                oslo.relax()
                heights.append(oslo.height)

            for j in range(oslo.L * 3):
                oslo.drive()
                drops = drops[1:10]
                drops.append(oslo.relax())
                heights.append(oslo.height)

            while True:
                oslo.drive()
                drops = drops[1:10]
                drops.append(oslo.relax())
                heights.append(oslo.height)
                if sum(drops[1:11]) / 10 == 1:
                    break

            smo_hei = []
            for l in range(w, oslo.L - w):  # Application of the smoother formula [[1]]
                heig = 0
                for k in range(l - w, l + w + 1):
                    heig += heights[k]
                smo_hei.append(heig)

            plt.plot(smo_hei, '*')
            plt.show()


    def func(L,a,b,w):
        return a*L*(1-b*L**(-w))

    if sys.argv[1] == "2c":
        """
        """
        TT = int(sys.argv[3])
        a = range(20, 200, 20)   # Size of the tested models

        Heights = []
        Devs = []
        for i in a:     # Cicles throought the different sizes
            oslo = Model(i, float(sys.argv[2]))

            hei = 0
            hei_dev2 = 0
            j = -1
            while j != 0:  # Drives the model to the stable state
                j -= 1
                oslo.drive()
                oslo.relax()
                if oslo.slope[-1] == 1 and j < 0:   # Initializes the stopping condition
                    j = TT
                if j > 0 :     # sums the heights of the system
                    hei += oslo.height
                    hei_dev2 += (float(oslo.height))**2
            print (1.*hei)/TT
            print  np.sqrt( hei_dev2/TT - (float(hei)/TT)**2)
            Heights.append(float(hei)/TT)
            Devs.append(np.sqrt( hei_dev2/TT - (float(hei)/TT)**2))
        b = Heights
        c = Devs
        fig, ax1 = plt.subplots()
        ax1.plot(a, b, 'b*')
        ax1.set_xlabel('Size (L) of the model')
        ax1.set_ylabel('Height', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(a, c, 'ro', color='r')
        ax2.set_ylabel('Deviations', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        plt.show()

        popt, pcov = curve_fit(func,a,b)
        print popt, pcov
        quit()



    if sys.argv[1] == "2d0":
        """  Does nothing yet
        """
        quit()
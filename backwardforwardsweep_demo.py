import numpy as np
import matplotlib.pyplot as plt

from networks import Network
from solvers import backwardforwardsweep

""" 
    Demonstrates code for backwardforwardsweep solver
    Assumes LV network, PQ loads only, and slack bus is the first bus (i.e. bus 0)
"""


network37 = Network('network37')

V_all, line_currents, V_mag, V_ang, S_line, max_diff, diff_save = backwardforwardsweep(network37)

plt.semilogy(np.array(diff_save),'-sk')
plt.xlabel('Iteration')
plt.ylabel('Max value change')
plt.title("backward forward sweep solver convergence")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.linalg import solve_triangular
import time
from pyvis.network import Network
import networkx as nx
import json

###########################################################
# Network characteristics
###########################################################
busNo = 37
Vbase = 4.8 / np.sqrt(3)  # kV Base voltage for normalization
Sbase = 100  # kVA Base apparent power for normalization
Zbase = 1000 * Vbase ** 2 / Sbase  # ohm

# this has put everything into some weird per unit scale??

BusNum = np.arange(busNo)
Slack_Bus_Num = [
    0]  # slack bus, Note that ideal secondary of the transformer is considered as the slack bus, i.e., V = 1<0 pu

Sbase_old_t = 2500 / 3  # kVA Transformer rated apparent power. Note that transforer impedance is given in pu and it is based on transformer rated values (S and V) as the base values. We should update this value based on new base values, i.e., Sbase and Vbase
Zbase_t_old = 1000 * Vbase ** 2 / Sbase_old_t  # Vbase is the same, however, Sbase_old_t and Sbase_new_t (Sbase) are different.
Zbase_t_new = 1000 * Vbase ** 2 / Sbase

# Transformers impedances
###########################################################
Zt_old_pu = 0.02 + 0.08j

Zt_new_pu = Zt_old_pu * Zbase_t_old / Zbase_t_new  # transformer impedance in the new per unit system

###########################################################
# Line impedances (R +jX) in ohms per mile
###########################################################
Z1 = 0.2926 + 0.1973j  # Z1 to Z4 are chosen from IEEE 37-node datasheet. Note that only info of phase A is being used because here the assumption is that the system is single phase.
Z2 = 0.4751 + 0.2973j
Z3 = 1.2936 + 0.6713j
Z4 = 2.0952 + 0.7758j

# Line lengths in ft, The original lengths are divided by 5280 to change to mile
length = [1.0, 0.3504, 0.1818, 0.2500, 0.1136, 0.0379, 0.0606, 0.0606, 0.1061, 0.1212, 0.0758, 0.0758, 0.0379, 0.0758,
          0.0985, 0.0379, 0.2424, 0.0606, 0.1136, 0.0455, 0.0530, 0.0379, 0.0530, 0.0758, 0.0455, 0.0606, 0.0682,
          0.0985, 0.1515, 0.1136, 0.0530, 0.1742, 0.0227, 0.1439, 0.0152, 0.0985]

# line information
node_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 9, 15, 15,
                   7, 6, 4, 20, 21, 21, 3, 24, 24, 3, 27, 28, 29, 30, 29, 32, 32,
                   28, 35])
node_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
          35, 36]

line_z_pu = np.array([Zt_new_pu * length[0], Z1 * length[1] / Zbase, Z2 * length[2] / Zbase, Z2 * length[3] / Zbase,
                      Z3 * length[4] / Zbase, Z3 * length[5] / Zbase, Z3 * length[6] / Zbase, Z3 * length[7] / Zbase,
                      Z3 * length[8] / Zbase, Z3 * length[9] / Zbase, Z3 * length[10] / Zbase, Z3 * length[11] / Zbase,
                      Z4 * length[12] / Zbase, Z3 * length[13] / Zbase, Z4 * length[14] / Zbase,
                      Z4 * length[15] / Zbase, Z4 * length[16] / Zbase, Z4 * length[17] / Zbase,
                      Z3 * length[18] / Zbase,
                      Z4 * length[19] / Zbase, Z3 * length[20] / Zbase, Z4 * length[21] / Zbase,
                      Z4 * length[22] / Zbase, Z4 * length[23] / Zbase, Z4 * length[24] / Zbase,
                      Z4 * length[25] / Zbase, Z3 * length[26] / Zbase, Z3 * length[27] / Zbase,
                      Z3 * length[28] / Zbase,
                      Z3 * length[29] / Zbase, Z4 * length[30] / Zbase, Z4 * length[31] / Zbase,
                      Z4 * length[32] / Zbase, Z4 * length[33] / Zbase, Z4 * length[34] / Zbase,
                      Z4 * length[35] / Zbase])

##########################################################
# Load in kW and kVAR at each node
##########################################################
P_load = np.array(
    [0, 0, 140, 0, 0, 0, 0, 0, 85, 0, 140, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 42, 42, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0,
     17, 85], dtype=np.double)
Q_load = np.array(
    [0, 0, 70, 0, 0, 0, 0, 0, 40, 0, 70, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 21, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 8,
     40], np.double())

# craete one normalised load array
load_powers = np.expand_dims(P_load/Sbase + 1j*Q_load/Sbase,1)



# # create connectivity graph
# connection_graph = np.zeros((busNo,busNo))
# for i in range(len(node_a)):
#     connection_graph[node_a[i],node_b[i]] = 1
#     connection_graph[node_b[i], node_a[i]] = 1

# create current_connection graph
# i.e. this will be linear system of equations for sum of currents at each node = 0
current_graph = np.zeros((busNo,len(node_a)))
for i in range(len(node_a)):
    a = node_a[i]
    b = node_b[i]
    current_graph[a, i] = -1.
    current_graph[b, i] = 1.
current_graph = current_graph[1:,:]     # remove slack node as we assume it can provide requried current

voltage_graph = current_graph.T

# todo: Use sparse matrices when the size gets large
# initialise voltages? what should the voltage at slack bus be?
V_slack = 1. + 0.j
node_voltages = np.ones((busNo-1,1)) * V_slack      # remove slack
line_currents = np.zeros((len(node_a),1))

max_iter = 100

diff_save = []

ts = time.time()

for iter in range(max_iter):
    ## backward sweep (calculate currents for fixed voltages)
    load_currents = np.conj(load_powers[1:])/np.conj(node_voltages)     # ignore load_current at slack node
    new_currents = solve_triangular(current_graph, load_currents,
                                                  unit_diagonal=True,
                                                  check_finite=False)
    current_diff = line_currents - new_currents
    line_currents = 1. * new_currents

    ## forward sweep
    # line voltage drops
    line_voltages = np.expand_dims(line_z_pu,1) * line_currents
    btmp = -1.*line_voltages
    btmp[0] = btmp[0] + V_slack
    new_voltages = solve_triangular(voltage_graph,btmp,lower=True,unit_diagonal=True,check_finite=False)
    voltage_diff = node_voltages - new_voltages
    node_voltages = 1. * new_voltages
    max_diff = np.maximum(np.max(np.abs(voltage_diff)),np.max(np.abs(current_diff)))
    # max_diff = np.max(np.abs(voltage_diff))
    diff_save.append(max_diff)
    if max_diff < 1e-10:
        break

tf = time.time()

V_all = np.vstack((np.atleast_2d(V_slack), node_voltages))

V_mag = np.absolute(V_all)
V_ang = np.angle(V_all)*(180/np.pi)
S_line = V_all[:-1] * np.conj(line_currents)

print('Time to run = ', tf-ts)

plt.plot(np.array(diff_save),'-sk')
plt.xlabel('Iteration')
plt.ylabel('Max value change')
plt.title("backward forward sweep solver convergence")
plt.show()

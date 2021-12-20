import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import spsolve_triangular

# implementation of backwardforwardsweepsolver
# assumes LV network, slack bus is first node, and PQ loads only

def backwardforwardsweep(network, max_iter=100, tolerance=1e-10):
    busNo = network.busNo
    V_slack = network.V_slack
    node_a = network.node_a
    sparse = network.sparse
    # node_b = network.node_b
    line_z_pu = network.line_z_pu
    current_graph = network.current_graph
    voltage_graph = network.voltage_graph
    load_powers = network.load_powers

    node_voltages = np.ones((busNo - 1, 1)) * V_slack  # remove slack
    line_currents = np.zeros((len(node_a), 1))

    diff_save = []

    for iter in range(max_iter):
        ## backward sweep (calculate currents for fixed voltages)
        load_currents = np.conj(load_powers[1:]) / np.conj(node_voltages)  # ignore load_current at slack node
        if sparse:
            new_currents = spsolve_triangular(current_graph, load_currents,
                                            unit_diagonal=True,
                                              lower=False)
        else:
            new_currents = solve_triangular(current_graph, load_currents,
                                            unit_diagonal=True,
                                            check_finite=False)
        current_diff = line_currents - new_currents
        line_currents = 1. * new_currents

        ## forward sweep
        # line voltage drops
        line_voltages = np.expand_dims(line_z_pu, 1) * line_currents
        btmp = -1. * line_voltages
        btmp[0] = btmp[0] + V_slack
        if sparse:
            new_voltages = spsolve_triangular(voltage_graph, btmp, lower=True, unit_diagonal=True)
        else:
            new_voltages = solve_triangular(voltage_graph, btmp, lower=True, unit_diagonal=True, check_finite=False)
        voltage_diff = node_voltages - new_voltages
        node_voltages = 1. * new_voltages
        max_diff = np.maximum(np.max(np.abs(voltage_diff)), np.max(np.abs(current_diff)))
        diff_save.append(max_diff)
        if max_diff < tolerance:
            break

    V_all = np.vstack((np.atleast_2d(V_slack), node_voltages))

    V_mag = np.absolute(V_all)
    V_ang = np.angle(V_all) * (180 / np.pi)
    S_line = V_all[:-1] * np.conj(line_currents)

    return V_all, line_currents, V_mag, V_ang, S_line, max_diff, diff_save
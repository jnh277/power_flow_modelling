import numpy as np
import scipy

# network Class description


class Network:
    def __init__(self, network=None, sparse=False):
        self.busNo = None            # number of busses
        self.vbase = None            # kV base voltage for normalisation
        self.sbase = None            # kVA apparent power for normalisation
        self.node_a = None           # node a for each line
        self.node_b = None           # node b for each line
        self.current_graph = None    # connection matrix for KCL
        self.voltage_graph = None    # connection matrix for KVL
        self.line_z_pu = None        # impedance of each line
        self.slack_bus_num = 0       # //todo: this is currently not used and is assumed zero
        self.load_powers = None      # power load connected to each node (including slack)
        self.V_slack = None           # voltage of slack node (normalised if loads are normalised)
        self.node_voltages = None
        self.line_currents = None
        self.sparse = sparse        # whether to store graphs in sparse format or not
        if network=="network37":
            self.load_network37()
        elif network=="test5":
            self.load_test_5()
        elif network=='single branch':
            self.load_single_branch()

    def load_network37(self):
        busNo = 37
        vbase = 4.8 / np.sqrt(3)  # kV Base voltage for normalization
        sbase = 100  # kVA Base apparent power for normalization
        zbase = 1000 * vbase ** 2 / sbase  # ohm
        slack_bus_num = [0]  # slack bus, Note that ideal secondary of the transformer is considered as the slack bus, i.e., V = 1<0 pu
        sbase_old_t = 2500 / 3  # kVA Transformer rated apparent power. Note that transforer impedance is given in pu and it is based on transformer rated values (S and V) as the base values. We should update this value based on new base values, i.e., Sbase and Vbase
        zbase_t_old = 1000 * vbase ** 2 / sbase_old_t  # Vbase is the same, however, Sbase_old_t and Sbase_new_t (Sbase) are different.
        zbase_t_new = 1000 * vbase ** 2 / sbase
        zt_old_pu = 0.02 + 0.08j
        zt_new_pu = zt_old_pu * zbase_t_old / zbase_t_new  # transformer impedance in the new per unit system
        z1 = 0.2926 + 0.1973j  # Z1 to Z4 are chosen from IEEE 37-node datasheet. Note that only info of phase A is being used because here the assumption is that the system is single phase.
        z2 = 0.4751 + 0.2973j
        z3 = 1.2936 + 0.6713j
        z4 = 2.0952 + 0.7758j
        length = [1.0, 0.3504, 0.1818, 0.2500, 0.1136, 0.0379, 0.0606, 0.0606, 0.1061, 0.1212, 0.0758, 0.0758, 0.0379,
                  0.0758,
                  0.0985, 0.0379, 0.2424, 0.0606, 0.1136, 0.0455, 0.0530, 0.0379, 0.0530, 0.0758, 0.0455, 0.0606,
                  0.0682,
                  0.0985, 0.1515, 0.1136, 0.0530, 0.1742, 0.0227, 0.1439, 0.0152, 0.0985]
        node_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 9, 15, 15,
                           7, 6, 4, 20, 21, 21, 3, 24, 24, 3, 27, 28, 29, 30, 29, 32, 32,
                           28, 35])
        node_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                  35, 36]
        line_z_pu = np.array(
            [zt_new_pu * length[0], z1 * length[1] / zbase, z2 * length[2] / zbase, z2 * length[3] / zbase,
             z3 * length[4] / zbase, z3 * length[5] / zbase, z3 * length[6] / zbase, z3 * length[7] / zbase,
             z3 * length[8] / zbase, z3 * length[9] / zbase, z3 * length[10] / zbase, z3 * length[11] / zbase,
             z4 * length[12] / zbase, z3 * length[13] / zbase, z4 * length[14] / zbase,
             z4 * length[15] / zbase, z4 * length[16] / zbase, z4 * length[17] / zbase,
             z3 * length[18] / zbase,
             z4 * length[19] / zbase, z3 * length[20] / zbase, z4 * length[21] / zbase,
             z4 * length[22] / zbase, z4 * length[23] / zbase, z4 * length[24] / zbase,
             z4 * length[25] / zbase, z3 * length[26] / zbase, z3 * length[27] / zbase,
             z3 * length[28] / zbase,
             z3 * length[29] / zbase, z4 * length[30] / zbase, z4 * length[31] / zbase,
             z4 * length[32] / zbase, z4 * length[33] / zbase, z4 * length[34] / zbase,
             z4 * length[35] / zbase])
        P_load = np.array(
            [0, 0, 140, 0, 0, 0, 0, 0, 85, 0, 140, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 42, 42, 0, 0, 8, 0, 0, 0, 0, 0,
             0, 0, 0,
             17, 85], dtype=np.double)
        Q_load = np.array(
            [0, 0, 70, 0, 0, 0, 0, 0, 40, 0, 70, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 21, 0, 0, 4, 0, 0, 0, 0, 0, 0,
             0, 0, 8,
             40], np.double())
        load_powers = np.expand_dims(P_load / sbase + 1j * Q_load / sbase, 1) # craete one normalised load array
        current_graph = np.zeros((busNo, len(node_a)))
        for i in range(len(node_a)):
            a = node_a[i]
            b = node_b[i]
            current_graph[a, i] = -1.
            current_graph[b, i] = 1.
        current_graph = current_graph[1:, :]  # remove slack node as we assume it can provide requried current // todo: reform this so nodes other than 0 can be slack
        voltage_graph = current_graph.T
        self.busNo = 37
        self.vbase = vbase
        self.sbase = sbase
        self.node_a = node_a
        self.node_b = node_b
        self.line_z_pu = line_z_pu
        self.load_powers = load_powers
        if self.sparse:
            self.current_graph = scipy.sparse.csr_matrix(current_graph)
            self.voltage_graph = scipy.sparse.csr_matrix(voltage_graph)
        else:
            self.current_graph = current_graph
            self.voltage_graph = voltage_graph
        self.V_slack = 1. + 0.j

    def load_test_5(self):
        busNo = 5       # 5 busses including slack bus
        vbase = 4.8 / np.sqrt(3)  # kV Base voltage for normalization
        sbase = 100  # kVA Base apparent power for normalization
        zbase = 1000 * vbase ** 2 / sbase  # ohm
        z3 = 1.2936 + 0.6713j
        length = [0.2, 0.3504, 0.1818, 0.2500]      # line lengths
        node_a = np.array([0, 1, 1, 3])
        node_b = [1, 2, 3, 4]
        line_z_pu = np.array([z3 * length[0] / zbase, z3 * length[1] / zbase,
                              z3 * length[2] / zbase, z3 * length[3] / zbase])
        P_load = np.array([0, 0, 140, 85, 140], dtype=np.double)
        Q_load = np.array([0, 0, 70, 40, 70], np.double())
        load_powers = np.expand_dims(P_load / sbase + 1j * Q_load / sbase, 1) # craete one normalised load array
        current_graph = np.zeros((busNo, len(node_a)))
        for i in range(len(node_a)):
            a = node_a[i]
            b = node_b[i]
            current_graph[a, i] = -1.
            current_graph[b, i] = 1.
        current_graph = current_graph[1:,:]  # remove slack node as we assume it can provide requried current // todo: reform this so nodes other than 0 can be slack
        voltage_graph = current_graph.T
        self.busNo = busNo
        self.vbase = vbase
        self.sbase = sbase
        self.node_a = node_a
        self.node_b = node_b
        self.line_z_pu = line_z_pu
        self.load_powers = load_powers
        self.current_graph = current_graph
        self.voltage_graph = voltage_graph
        self.V_slack = 1. + 0.j
        self.zbase = zbase
        self.length = length
        self.zpk = z3/ zbase

    def load_single_branch(self):
        busNo = 10       # 5 busses including slack bus
        vbase = 4.8 / np.sqrt(3)  # kV Base voltage for normalization
        sbase = 100  # kVA Base apparent power for normalization
        zbase = 1000 * vbase ** 2 / sbase  # ohm
        z3 = 1.2936 + 0.6713j
        # length = [0.2, 0.3504, 0.1818, 0.2500]      # line lengths
        length = [0.2] * (busNo-1)
        # node_a = np.array([0, 1, 2, 3])
        # node_b = [1, 2, 3, 4]
        node_a = np.arange(busNo-1)
        node_b = np.arange(1, busNo)
        # line_z_pu = np.array([z3 * length[0] / zbase, z3 * length[1] / zbase,
        #                       z3 * length[2] / zbase, z3 * length[3] / zbase])
        line_z_pu = np.array(length) * z3/zbase
        P_load = np.array([0, 0, 140, 85, 140], dtype=np.double)
        Q_load = np.array([0, 0, 70, 40, 70], np.double())
        load_powers = np.expand_dims(P_load / sbase + 1j * Q_load / sbase, 1) # craete one normalised load array
        current_graph = np.zeros((busNo, len(node_a)))
        for i in range(len(node_a)):
            a = node_a[i]
            b = node_b[i]
            current_graph[a, i] = -1.
            current_graph[b, i] = 1.
        current_graph = current_graph[1:,:]  # remove slack node as we assume it can provide requried current // todo: reform this so nodes other than 0 can be slack
        voltage_graph = current_graph.T
        self.busNo = busNo
        self.vbase = vbase
        self.sbase = sbase
        self.node_a = node_a
        self.node_b = node_b
        self.line_z_pu = line_z_pu
        self.load_powers = load_powers
        self.current_graph = current_graph
        self.voltage_graph = voltage_graph
        self.V_slack = 1. + 0.j
        self.zbase = zbase
        self.length = length
        self.zpk = z3/ zbase
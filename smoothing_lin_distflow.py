import numpy as np
from networks import Network
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from numpy import ma

""" 
    First we simulate a single branch network using lindistflow
"""

# get network parameters
T = 1       # one timestep
noise_std = 0.01
network = Network('single branch', sparse=False)
N = network.busNo
length_true = network.length
z= network.line_z_pu
R = np.real(z)
X = np.imag(z)

inv_current_graph = np.linalg.inv(network.current_graph)
inv_voltage_graph = np.linalg.inv(network.voltage_graph)

def linearApproximateDistflow(p_loads, q_loads, R, X, inv_voltage_graph, inv_current_graph, T):
    """

    :param p_loads: shape (N-1, T)
    :param q_loads: shape (N-1, T)
    :param R: shape (N-1,)
    :param X: shape (N-1,)
    :param inv_voltage_graph: shape (N-1,N-1)
    :param inv_current_graph: shape (N-1,N-1)
    :return:
    """
    p_line = np.matmul(inv_current_graph, p_loads)
    q_line = np.matmul(inv_current_graph, q_loads)

    U0 = np.vstack([np.ones((1, T)), np.zeros((N - 2, T))])
    B = U0-2 * np.expand_dims(R,1) * p_line + -2 * np.expand_dims(X, 1) * q_line


    U = np.matmul(inv_voltage_graph, B)
    return U, p_line, q_line

p = np.random.rand(N-1, T)
q = 0.5*np.random.rand(N-1, T)

U, p_line, q_line = linearApproximateDistflow(p, q, R, X, inv_voltage_graph, inv_current_graph, T)

y = np.vstack([U, p, q, p_line[[0],:], q_line[[0],:]]) + noise_std * np.random.randn((N-1)*3+2, T)


"""
    use kalman smoothing over the network dimension
"""

 # states are p_in, q_in, p_l, q_l, U


# slack bus values
U0 = 1.0
pl0 = 0.0
ql0 = 0.0
pin0 = p_line[0,0]
qin0 = q_line[0,0]

z0 = np.array([pin0, qin0, pl0, ql0, U0])
P0 = 1e-4 * np.eye(5)

r = R[0]
x = X[0]
A = np.array([[1, 0, -1, 0, 0],
              [0, 1, 0, -1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [-2*r, -2*x, 2*r, 2*x, 1]])
# B = np.array([[0], [0], [0], [0], [0]])
C = np.array([[0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])
# D = np.zeros((3, 1))
R = noise_std**2 * np.eye(3)
Q = np.zeros((5,5))
Q = np.fill_diagonal(Q, [1e-1, 1e-1, 1.0, 1.0, 1e-1])

kf = KalmanFilter(transition_matrices=A, transition_offsets=[0, 0, 0, 0, 0], observation_offsets=[0, 0, 0],
                  initial_state_mean=z0, initial_state_covariance=P0, observation_matrices=C, observation_covariance=R,
                  transition_covariance=Q, n_dim_state=5, n_dim_obs=3, em_vars=['transition_covariance'])



measurements = np.vstack([[0, 0, 1.0],np.hstack([p, q, U])])
measurements = measurements + noise_std * np.random.randn(measurements.shape[0],measurements.shape[1])
mask = np.zeros(measurements.shape)
mask[2] = 1
# mask[9] = 1
measurements = ma.masked_array(measurements, mask=mask)

(z_smooth, P_smooth) = kf.em(measurements).smooth(measurements)
# (z_smooth, P_smooth) = kf.smooth(measurements)

# plot voltages
plt.subplot(5,1,1)
plt.plot(U, label='True')
plt.plot(z_smooth[1:, -1], label='est', linestyle='--')
plt.legend()
plt.title('voltage mags')

plt.subplot(5,1,2)
plt.plot(p_line, label='True')
plt.plot(z_smooth[1:, 0], label='est', linestyle='--')
plt.legend()
plt.title('pline')

plt.subplot(5,1,3)
plt.plot(q_line, label='True')
plt.plot(z_smooth[1:, 1], label='est', linestyle='--')
plt.legend()
plt.title('qline')

plt.subplot(5,1,4)
plt.plot(p, label='True')
plt.plot(z_smooth[1:, 2], label='est', linestyle='--')
plt.legend()
plt.title('pL')

plt.subplot(5,1,5)
plt.plot(q, label='True')
plt.plot(z_smooth[1:, 3], label='est', linestyle='--')
plt.legend()
plt.title('qL')

plt.tight_layout()
plt.show()





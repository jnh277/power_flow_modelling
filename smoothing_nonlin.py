import numpy as np
from networks import Network
from solvers import backwardforwardsweep
from pykalman import AdditiveUnscentedKalmanFilter
import matplotlib.pyplot as plt

""" 
    First we simulate a single branch network using nonlin power flows
"""

# get network parameters
T = 1       # one timestep
noise_std = 0.01
network = Network('single branch', sparse=False)
N = network.busNo
length_true = network.length
z= network.line_z_pu

p = np.zeros((N, T))
p[1:] = np.random.rand(N-1, T)
q = np.zeros((N, T))
q[1:] = 0.5*np.random.rand(N-1, T)

network.load_powers = p + 1j * q

V_all, line_currents, V_mag, V_ang, S_line, max_diff, diff_save = backwardforwardsweep(network)


"""
    Estimate using a UKF smoother
"""

x0 = np.array([np.real(line_currents[0,0]), np.imag(line_currents[0,0]),
               0., 0., 1, 0])
P0 = 1e-4 * np.eye(6)

R = noise_std**2 * np.eye(3)
Q = np.zeros((6,6))
Q = np.fill_diagonal(Q, [1e-3, 1e-3, 1.0, 1.0, 1e-3, 1e-3])


def process(current_state):
    current_in = current_state[0] + 1j*current_state[1]
    current_load = current_state[2] + 1j*current_state[3]
    voltage = current_state[4] + 1j*current_state[5]

    next_current_in = current_in - current_load
    next_voltage = voltage - z[0] * (current_in - current_load)
    next_state = np.zeros((6,))
    next_state[0] = np.real(next_current_in)
    next_state[1] = np.imag(next_current_in)
    next_state[4] = np.real(next_voltage)
    next_state[5] = np.imag(next_voltage)
    return next_state

def measurement(current_state):
    current_load = current_state[2] + 1j*current_state[3]
    voltage = current_state[4] + 1j*current_state[5]

    load_power = voltage * np.conj(current_load)

    meas = np.zeros((3,))
    meas[0] = np.real(load_power)
    meas[1] = np.imag(load_power)
    meas[2] = np.abs(voltage)
    return meas


ukf = AdditiveUnscentedKalmanFilter(transition_functions=process, observation_functions=measurement,
                                    transition_covariance=Q, observation_covariance=R,
                                    initial_state_mean=x0, initial_state_covariance=P0,
                                    n_dim_state=6, n_dim_obs=3)


# measurements = np.vstack([[0, 0, 1.0],np.hstack([p, q, U])])
measurements = np.hstack([p, q, V_mag])
measurements = measurements + noise_std * np.random.randn(measurements.shape[0],measurements.shape[1])
# mask = np.zeros(measurements.shape)
# mask[2] = 1
# mask[9] = 1
# measurements = ma.masked_array(measurements, mask=mask)

(x_smooth, P_smooth) = ukf.smooth(measurements)


plt.subplot(3,2,1)
plt.plot(np.real(V_all), label='True')
plt.plot(x_smooth[:, -2], label='est', linestyle='--')
plt.legend()
plt.title('voltage real')

plt.subplot(3,2,2)
plt.plot(np.imag(V_all), label='True')
plt.plot(x_smooth[:, -1], label='est', linestyle='--')
plt.legend()
plt.title('voltage imag')


plt.subplot(3,2,3)
plt.plot(np.real(line_currents), label='True')
plt.plot(x_smooth[:, 0], label='est', linestyle='--')
plt.legend()
plt.title('line current real')

plt.subplot(3,2,4)
plt.plot(np.imag(line_currents), label='True')
plt.plot(x_smooth[:, 1], label='est', linestyle='--')
plt.legend()
plt.title('line current imag')

plt.subplot(3,2,5)
plt.plot(p, label='True')
plt.plot(x_smooth[1:, 2], label='est', linestyle='--')
plt.legend()
plt.title('pL')

plt.subplot(5,1,5)
plt.plot(q, label='True')
plt.plot(x_smooth[1:, 3], label='est', linestyle='--')
plt.legend()
plt.title('qL')

plt.tight_layout()
plt.show()

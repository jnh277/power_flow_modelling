import numpy as np
from networks import Network
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from numpy import ma
import scipy
from scipy.optimize import minimize

""" 
    First we simulate a single branch network using lindistflow
"""

# get network parameters
T = 300       # one timestep
noise_std = 0.0001
network = Network('test5', sparse=False)
N = network.busNo
length_true = network.length
z = network.line_z_pu
R = np.real(z)
X = np.imag(z)

Rpkm = R / length_true
Xpkm = X / length_true

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

# y = np.vstack([U, p, q, p_line[[0],:], q_line[[0],:]]) + noise_std * np.random.randn((N-1)*3+2, T)
yU = U + np.random.normal(0, noise_std, U.shape)
yPl = p + np.random.normal(0, noise_std, p.shape)
yQl = q + np.random.normal(0, noise_std, q.shape)
ypline = p_line[0,:] + np.random.normal(0, noise_std, p_line[0,:].shape)
yqline = q_line[0,:] + np.random.normal(0, noise_std, q_line[0,:].shape)
x_true = np.vstack([p, q])


# build C - matrix that takes p and q loads to the outputs
# build the C3 matrix, matrix that uses R and X as if we know R and X
A = inv_current_graph
B = inv_voltage_graph
u0 = np.vstack([np.ones((1, 1)), np.zeros((N - 2, 1))])
Abar = scipy.linalg.block_diag(A, A)
Rbar = np.diag(R)
Xbar = np.diag(X)
C3 = -2 * B @ np.hstack([Rbar, Xbar]) @ Abar

def build_C(inv_current_graph, inv_voltage_graph, C3):
    A = inv_current_graph
    B = inv_voltage_graph
    u0 = np.vstack([np.ones((1, 1)), np.zeros((N - 2, 1))])
    Abar = scipy.linalg.block_diag(A, A)
    C1 = scipy.linalg.block_diag(u0.T, u0.T) @ Abar
    C2 = np.eye(2*(N-1))
    C = np.vstack([C1, C2, C3])
    return C, u0, B

C, u0, B = build_C(inv_current_graph, inv_voltage_graph, C3)

y = np.vstack([ypline, yqline, yPl, yQl, yU - B @ u0])

prior_var = 10

# Use bayesian gaussian regression to find the states if we know the parameters
xhat = np.linalg.solve((C.T @ C / noise_std**2 + np.eye(2*(N-1))/prior_var), C.T @ y / noise_std**2)
varhat = np.linalg.inv((C.T @ C / noise_std**2 + np.eye(2*(N-1))/prior_var))


# now try EM to learn the C3 matrix


C3hat = np.random.rand(N-1, 2*(N-1)) # random initial guess
sigma = 1.0                     # estimate of scalar noise std
Sigma = np.eye(len(yU)+2*len(yPl)+2)
mse = []
sigma_list = [sigma]
for i in range(100):
    # E step to update estimates of states and the state covariances
    C, u0, B = build_C(inv_current_graph, inv_voltage_graph, C3hat)
    y = np.vstack([ypline, yqline, yPl, yQl, yU - B @ u0])
    prior_var = 10

    # xhat = np.linalg.solve((C.T @ np.linalg.solve(Sigma, C) + np.eye(2 * (N - 1)) / prior_var), C.T @ np.linalg.solve(Sigma,y))
    # varhat = np.linalg.inv((C.T @ np.linalg.solve(Sigma, C) + np.eye(2 * (N - 1)) / prior_var))
    xhat = np.linalg.solve((C.T @ C / sigma ** 2 + np.eye(2 * (N - 1)) / prior_var), C.T @ y / sigma ** 2)
    varhat = np.linalg.inv((C.T @ C / sigma ** 2 + np.eye(2 * (N - 1)) / prior_var))

    mse.append(np.mean((y - C@xhat)**2))

    # M step
    Exx = xhat @ xhat.T + varhat                    # the expected value of x*x^T
    # Eyx = y @ xhat.T                              # the expected value of y * x^T
    EyUx = (yU - B @ u0) @ xhat.T                   # the expected value of yU * x^T
    C3hat = np.linalg.solve(Exx.T, EyUx.T).T

    # update sigma
    C = build_C(inv_current_graph, inv_voltage_graph, C3hat)[0]
    Sigma = (1/T)*((y - C @ xhat) @ (y - C @ xhat).T + C @ varhat @ C.T)
    sigma = np.mean(np.sqrt(Sigma.diagonal()))  # hack to get scalar sigma
    sigma_list.append(sigma)


voltage_est = B @ u0 + C3hat @ xhat

plt.semilogy(sigma_list)
plt.ylabel('sigma')
plt.xlabel("EM iteration")
plt.show()

plt.semilogy(mse)
plt.show()


plt.subplot(2,1,1)
plt.imshow(C3)
plt.title('C3 True')

plt.subplot(2,1,2)
plt.imshow(C3hat)
plt.title('C3 est')

plt.tight_layout()
plt.show()
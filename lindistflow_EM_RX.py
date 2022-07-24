import numpy as np
from networks import Network
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from numpy import ma
import scipy
from scipy.optimize import minimize
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jit, grad
from tqdm import tqdm

""" 
    First we simulate a single branch network using lindistflow
"""

# get network parameters
T = 300       # one timestep
noise_std = 0.01
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

def build_C(inv_current_graph, inv_voltage_graph, R, X):
    A = inv_current_graph
    B = inv_voltage_graph
    u0 = np.vstack([np.ones((1, 1)), np.zeros((N - 2, 1))])


    Abar = scipy.linalg.block_diag(A, A)
    Rbar = np.diag(R)
    Xbar = np.diag(X)

    C1 = scipy.linalg.block_diag(u0.T, u0.T) @ Abar
    C2 = np.eye(2*(N-1))
    C3 = -2 * B @ np.hstack([Rbar, Xbar]) @ Abar

    C = np.vstack([C1, C2, C3])
    return C, u0, B

C, u0, B = build_C(inv_current_graph, inv_voltage_graph, R, X)

y = np.vstack([ypline, yqline, yPl, yQl, yU - B @ u0])

prior_var = 10

# Use bayesian gaussian regression to find the states
xhat = np.linalg.solve((C.T @ C / noise_std**2 + np.eye(2*(N-1))/prior_var), C.T @ y / noise_std**2)
varhat = np.linalg.inv((C.T @ C / noise_std**2 + np.eye(2*(N-1))/prior_var))


# now try EM
def E_step(R, X, sigma):
    C, u0, B = build_C(inv_current_graph, inv_voltage_graph, R, X)
    y = np.vstack([ypline, yqline, yPl, yQl, yU - B @ u0])

    prior_var = 10

    # Use bayesian gaussian regression to find the states
    xhat = np.linalg.solve((C.T @ C / sigma ** 2 + np.eye(2 * (N - 1)) / prior_var), C.T @ y / sigma ** 2)
    varhat = np.linalg.inv((C.T @ C / sigma ** 2 + np.eye(2 * (N - 1)) / prior_var))

    return xhat, varhat


# M step
# Exx = xhat @ xhat.T + varhat
# Eyx = y @ xhat.T
#
# EyUx = (yU - B @ u0) @ xhat.T
#
# C3hat = np.linalg.solve(Exx.T, EyUx.T).T

def Q_obj(theta, xhat, varhat):
    # R = Rpkm * theta
    # X = Xpkm * theta
    R = theta[:N-1]
    X = theta[N-1:]
    A = inv_current_graph
    B = inv_voltage_graph
    u0 = jnp.vstack([jnp.ones((1, 1)), jnp.zeros((N - 2, 1))])
    Abar = jscipy.linalg.block_diag(A, A)
    Rbar = jnp.diag(R)
    Xbar = jnp.diag(X)

    C3 = -2 * B @ jnp.hstack([Rbar, Xbar]) @ Abar
    EyUCx = jnp.trace((yU - B @ u0).T @ C3 @ xhat)

    ExCCx = 0
    for i in range(T):
        ExCCx += jnp.trace(C3 @ xhat[:, [i]] @ xhat[:,[i]].T @ C3.T + C3@varhat@C3.T)
    # ExCCx = jnp.trace(C3 @ xhat @ xhat.T @ C3.T + C3@varhat@C3.T)
    cost = - 2*EyUCx + ExCCx

    return cost

# cost_func = jit(Q_obj)
# grad_func = jit(grad(neglogl))
grad_func = grad(Q_obj)
def numpy_grad(theta, xhat, varhat):
    return np.array(grad_func(theta, xhat, varhat))

def numpy_cost(theta, xhat, varhat):
    return np.array(Q_obj(theta, xhat, varhat))


theta = 0.1 * np.ones(len(length_true) * 2)
# theta = np.array(length_true)
sigmahat = 1.
sigma_list = [sigmahat]
theta_list = [theta]
mse = []
M = 25
Rhat = theta[:N - 1]
Xhat = theta[N - 1:]
C = build_C(inv_current_graph, inv_voltage_graph, Rhat, Xhat)[0]
for i in tqdm(range(M), desc='Running EM'):

    xhat, varhat = E_step(Rhat, Xhat, sigmahat)

    mse.append(np.mean((y - C @ xhat) ** 2))

    res = minimize(lambda theta: numpy_cost(theta, xhat, varhat), theta,
                   method="SLSQP", jac=lambda theta: numpy_grad(theta, xhat, varhat), bounds=[(0, None)]*8)
    theta = res.x
    theta_list.append(theta)

    Rhat = theta[:N - 1]
    Xhat = theta[N - 1:]
    C = build_C(inv_current_graph, inv_voltage_graph, Rhat, Xhat)[0]
    # Sigma = (1/T)* ((y - C @ xhat) @ (y - C @ xhat).T + C @ varhat @ C.T)
    # Sigma = ((y - C @ xhat) @ (y - C @ xhat).T + C @ varhat @ C.T)
    Sigma = ((M-i)/T)*((y - C @ xhat) @ (y - C @ xhat).T + C @ varhat @ C.T)  # need to do this to slow down convergence of sigma
    sigmahat = np.mean(np.sqrt(Sigma.diagonal()))
    sigma_list.append(sigmahat)

theta_hist = np.vstack(theta_list)

print('rhat error ', Rhat - R)
print('xhat error ', Xhat - X)

plt.semilogy(sigma_list)
plt.ylabel('sigma')
plt.xlabel("EM iteration")
plt.show()

plt.semilogy(mse)
plt.ylabel('MSE')
plt.xlabel("EM iteration")
plt.show()

for i in range(len(R)):
    plt.subplot(len(R),2,(i+1)*2-1)
    plt.semilogy(theta_hist[:,i])
    plt.axhline(R[i], linestyle='--', color='k')
    plt.ylabel('R_{}'.format(i))
    plt.xlabel('EM iteration')

    plt.subplot(len(R),2,(i+1)*2)
    plt.semilogy(theta_hist[:,i+len(R)])
    plt.axhline(X[i], linestyle='--', color='k')
    plt.ylabel('X_{}'.format(i))
    plt.xlabel('EM iteration')

plt.tight_layout()
plt.show()


# a = np.array([[1, 2],[3, 4]])
# b = np.array([[0.25],[5]])
#
# c = np.kron(b, a)
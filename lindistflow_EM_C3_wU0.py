import numpy as np
from networks import Network
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as linalg

""" 
    First we simulate a single branch network using lindistflow
    states to estimate, U0, p_load, q_load
    parameters to estimate, the bottom right of the mapping matrix that goes from p_load, q_load to meas
"""

# get network parameters
T = 30000       # one timestep
noise_std = 0.01
network = Network('test5', sparse=False)
N = network.busNo
length_true = network.length
z = network.line_z_pu
R = np.real(z)
X = np.imag(z)
u0 = np.abs(network.V_slack)

inv_current_graph = np.linalg.inv(network.current_graph)
inv_voltage_graph = np.linalg.inv(network.voltage_graph)

def linearApproximateDistflow(p_loads, q_loads, R, X, inv_voltage_graph, inv_current_graph, T, u0):
    """

    :param p_loads: shape (N-1, T)
    :param q_loads: shape (N-1, T)
    :param R: shape (N-1,)
    :param X: shape (N-1,)
    :param inv_voltage_graph: shape (N-1,N-1)
    :param inv_current_graph: shape (N-1,N-1)
    :param u0: slack node voltage mag
    :return:
    """
    p_line = np.matmul(inv_current_graph, p_loads)
    q_line = np.matmul(inv_current_graph, q_loads)

    U0vec = np.vstack([u0*np.ones((1, T)), np.zeros((N - 2, T))])
    B = U0vec-2 * np.expand_dims(R,1) * p_line + -2 * np.expand_dims(X, 1) * q_line


    U = np.vstack([u0 * np.ones((1, T)),np.matmul(inv_voltage_graph, B)])
    return U, p_line, q_line

p = np.random.rand(N-1, T)
q = 0.5*np.random.rand(N-1, T)
# q = 0.5*p

U, p_line, q_line = linearApproximateDistflow(p, q, R, X, inv_voltage_graph, inv_current_graph, T, u0)


# y = np.vstack([U, p, q, p_line[[0],:], q_line[[0],:]]) + noise_std * np.random.randn((N-1)*3+2, T)
yU = U + np.random.normal(0, noise_std, U.shape)
yPl = p + np.random.normal(0, noise_std, p.shape)
yQl = q + np.random.normal(0, noise_std, q.shape)
ypline = p_line[0,:] + np.random.normal(0, noise_std, p_line[0,:].shape)
yqline = q_line[0,:] + np.random.normal(0, noise_std, q_line[0,:].shape)
x_true = np.vstack([p, q, U[[0],:]])

y = np.vstack([ypline, yqline, yPl, yQl, yU])


# build C - matrix that takes p and q loads to the outputs
# build the C3 matrix, matrix that uses R and X as if we know R and X
A = inv_current_graph
B = inv_voltage_graph
e1 = np.zeros((N - 1, 1))
e1[0] = 1
Abar = linalg.block_diag(A, A)
Rbar = np.diag(R)
Xbar = np.diag(X)
C3_true = np.hstack([-2 * B @ np.hstack([Rbar, Xbar]) @ Abar, B @ e1])

# def build_C(inv_current_graph, inv_voltage_graph, Ctheta):
#     A = inv_current_graph
#     B = inv_voltage_graph
#     ones_vec = np.vstack([np.ones((1, 1)), np.zeros((N - 2, 1))])
#     Abar = linalg.block_diag(A, A)
#     C1 = np.hstack([linalg.block_diag(ones_vec.T, ones_vec .T) @ Abar, np.zeros((2,1))])          # pline, qline at slack node
#     C2 = np.eye(2*(N-1)+1)                                            # pload, qload
#     C3 = np.hstack([Ctheta, B @ ones_vec])
#     C = np.vstack([C1, C2, C3])
#     return C

def build_C(inv_current_graph, inv_voltage_graph, C3):
    A = inv_current_graph
    ones_vec = np.vstack([np.ones((1, 1)), np.zeros((N - 2, 1))])
    Abar = linalg.block_diag(A, A)
    C1 = np.hstack([linalg.block_diag(ones_vec.T, ones_vec .T) @ Abar, np.zeros((2,1))])          # pline, qline at slack node
    C2 = np.eye(2*(N-1)+1)                                            # pload, qload
    C = np.vstack([C1, C2, C3])
    return C

C = build_C(inv_current_graph, inv_voltage_graph, C3_true)

prior_var = 10

# Use bayesian gaussian regression to find the states if we know the parameters
xhat = np.linalg.solve((C.T @ C / noise_std**2 + np.eye(2*(N-1)+1)/prior_var), C.T @ y / noise_std**2)
varhat = np.linalg.inv((C.T @ C / noise_std**2 + np.eye(2*(N-1)+1)/prior_var))


# now try EM to learn the C3 matrix


C3hat = np.random.rand(N-1, 2*(N-1)+1) # random initial guess
C = build_C(inv_current_graph, inv_voltage_graph, C3hat)
e1 = np.zeros((N - 1, 1))
e1[0] = 1
sigma = 1.0                     # estimate of scalar noise std
Sigma = np.eye(len(yU)+2*len(yPl)+2)
mse = []
sigma_list = [sigma]
M = 50
for i in range(M):
    # E step to update estimates of states and the state covariances

    prior_var = 10

    xhat = np.linalg.solve((C.T @ C / noise_std ** 2 + np.eye(2 * (N - 1) + 1) / prior_var), C.T @ y / noise_std ** 2)
    varhat = np.linalg.inv((C.T @ C / noise_std ** 2 + np.eye(2 * (N - 1) + 1) / prior_var))

    mse.append(np.mean((y - C@xhat)**2))

    # M step
    Exx = xhat @ xhat.T + varhat                    # the expected value of x*x^T
    EyUx = yU[1:,:] @ xhat.T                   # the expected value of yU * x^T
    C3hat = np.linalg.solve(Exx.T, EyUx.T).T


    # update sigma
    C = build_C(inv_current_graph, inv_voltage_graph, C3hat)
    Sigma = (1/T)*((y - C @ xhat) @ (y - C @ xhat).T + C @ varhat @ C.T)
    sigma = np.mean(np.sqrt(Sigma.diagonal()))  # hack to get scalar sigma
    sigma_list.append(sigma)


# voltage_est = B @ u0 + C3 @ xhat        # not including slack node

plt.semilogy(sigma_list)
plt.ylabel('sigma')
plt.xlabel("EM iteration")
plt.show()

plt.semilogy(mse)
plt.ylabel('MSE')
plt.xlabel("EM iteration")
plt.show()


plt.subplot(2,1,1)
plt.imshow(C3_true)
plt.title('all of C3 True')

plt.subplot(2,1,2)
plt.imshow(C3hat)
plt.title('all of C3 est')

plt.tight_layout()
plt.show()

plt.subplot(2,1,1)
plt.imshow(C3_true[:,:-1])
plt.title('r,x part of C3 True')

plt.subplot(2,1,2)
plt.imshow(C3hat[:,:-1])
plt.title('r, x part of C3 est')

plt.tight_layout()
plt.show()

plt.hist(xhat[-1,:])
plt.xlabel('u0 est')
plt.show()
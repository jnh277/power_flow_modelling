import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import seaborn as sns
from networks import Network
from solvers import backwardforwardsweep
import stan

noise_std = 0.01
observed = [0, 1, 2, 3, 4]    # works great
# observed = [0, 2, 3, 4]       # works great
# observed = [0, 2, 4]            # some bias but pretty close still
# observed =  [0, 1, 4]           # a bit worse bias but still kinda close
# observed = [0, 1, 2, 4]         # some bias but still reasonably close
T = 1000

network = Network('test5', sparse=False)

ts = time.time()
V_all, line_currents, V_mag, V_ang, S_line, max_diff, diff_save = backwardforwardsweep(network)
tf = time.time()

busNo = network.busNo

# measure V_mag
y_mag = V_mag[observed,:] + noise_std * np.random.randn(len(observed), T)

# measure P
y_p = np.real(network.load_powers)[observed,:] + noise_std * np.random.randn(len(observed), T)
# meaure Q
y_q = np.imag(network.load_powers)[observed,:] + noise_std * np.random.randn(len(observed), T)

# stack all meas
y = np.vstack([y_mag, y_p, y_q])


p_true = np.real(network.load_powers).flatten()
q_true = np.imag(network.load_powers).flatten()
length_true = network.length


def mse(theta):
    """ mse using voltage/power formulation """
    N = busNo-1     # assume slack bus is known

    Vreal = theta[:N]
    Vimag = theta[N:2*N]
    l = theta[2*N:3*N]
    sig = theta[-1]

    zpk = network.zpk
    Y = 1 / (zpk * l)

    V = np.hstack([1+0j,Vreal + 1j * Vimag])

    # work out branch currents
    node_a = network.node_a
    node_b = network.node_b
    I = np.zeros((N,),dtype=complex)
    for i in range(N):
        I[i] = Y[i] * (V[node_a[i]]-V[node_b[i]])

    # branch current sums at each node (remainder must be load or generation)
    Isum = np.zeros((N+1,),dtype=complex)
    for i in range(N):
        Isum[node_a[i]] -= I[i]
        Isum[node_b[i]] += I[i]

    shat = V * np.conjugate(Isum)
    # slack node does not have load and gen is not counted
    shat[0] = 0 + 0j

    yhat = np.atleast_2d(np.hstack([abs(V[observed]), np.real(shat[observed]), np.imag(shat[observed])])).T
    err = (y - yhat).reshape([-1,])
    return np.mean(err**2)

def neglogl(theta):
    """ maximum likelihood using voltage/power formulation """
    N = busNo-1     # assume slack bus is known

    Vreal = theta[:N]
    Vimag = theta[N:2*N]
    l = theta[2*N:3*N]
    sig = theta[-1]

    zpk = network.zpk
    Y = 1 / (zpk * l)

    V = np.hstack([1+0j,Vreal + 1j * Vimag])

    # work out branch currents
    node_a = network.node_a
    node_b = network.node_b
    I = np.zeros((N,),dtype=complex)
    for i in range(N):
        I[i] = Y[i] * (V[node_a[i]]-V[node_b[i]])

    # branch current sums at each node (remainder must be load or generation)
    Isum = np.zeros((N+1,),dtype=complex)
    for i in range(N):
        Isum[node_a[i]] -= I[i]
        Isum[node_b[i]] += I[i]

    shat = V * np.conjugate(Isum)
    # slack node does not have load and gen is not counted
    shat[0] = 0 + 0j

    yhat = np.atleast_2d(np.hstack([abs(V[observed]), np.real(shat[observed]), np.imag(shat[observed])])).T
    err = (y - yhat).reshape([-1,])
    logl = np.sum(-0.5 * np.log(sig * np.sqrt(2*np.pi))- 0.5 * err ** 2 / sig ** 2)
    return -logl

theta0 = np.hstack([np.real(V_all[1:]).flatten(), np.imag(V_all[1:]).flatten(), length_true, np.ones((1,))])
theta0 = theta0 * (0.8 + 0.4 *np.random.random(theta0.shape))       # start within +/0 20% of true

# should real voltage always be positive??
res = minimize(neglogl, theta0, bounds = [(0, None)]*4 + [(None,None)]*4 + [(0,None)]*4 +[(1e-6, None)],tol=1e-14)
theta = res.x

N = busNo-1
Vreal_hat = theta[:N]
Vimag_hat = theta[N:2 * N]
l_hat = theta[2 * N:3 * N]
sig_hat = theta[-1]


## now do metropolis hastings to get distributions

## Try out metropolis hastings
M = 5000
samps = np.zeros((M, len(theta)))
samps[0] = theta
log_prob = np.zeros((M,))
log_prob[0] = -neglogl(samps[0])
H_inv = res.hess_inv.todense()
cov = H_inv * mse(samps[0])
L_cov = np.linalg.cholesky(cov)
mh_factor = 1e-4

accept_count = 0
for i in range(1,M):
    # propose =  np.random.multivariate_normal(samps[i-1], cov)
    # propose = samps[i-1] + mh_std * np.random.randn(len(theta),)
    propose = samps[i-1] + L_cov @ np.random.randn(len(theta),) * mh_factor

    propose[8:] = abs(propose[8:])      # lengths and std cant be negative
    logp = - neglogl(propose)     # no priors

    accept_prob = min(1, np.exp(logp - log_prob[i-1]))
    if np.random.rand() < accept_prob:
        samps[i] = propose
        log_prob[i] = logp
        accept_count += 1
    else:
        samps[i] = samps[i-1]
        log_prob[i] = log_prob[i-1]

samps = samps[int(M/2):]

plt.plot(samps[:,-1])
plt.title('sig hat trace')
plt.show()

for i in range(0, 4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(samps[:, i], label='MH', fill=True)
    plt.axvline(np.real(V_all[i+1]), color='k', linestyle='-', linewidth=2., label='True')
    plt.axvline(theta[i],  color='r', linestyle='--', linewidth=2., label='ML')
    if i==4:
        plt.legend()
    plt.title('V_real[{}]'.format(i+1))
    plt.ylabel('')
plt.legend()
plt.tight_layout()
plt.show()

for i in range(0, 4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(samps[:, i+N], label='MH', fill=True)
    plt.axvline(np.imag(V_all[i+1]), color='k', linestyle='-', linewidth=2., label='True')
    plt.axvline(theta[i+N],  color='r', linestyle='--', linewidth=2., label='ML')
    if i==4:
        plt.legend()
    plt.title('V_imag[{}]'.format(i+1))
    plt.ylabel('')
plt.legend()
plt.tight_layout()
plt.show()

for i in range(0, 4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(samps[:, i+N*2], label='MH', fill=True)
    plt.axvline(length_true[i], color='k', linestyle='-', linewidth=2., label='True')
    plt.axvline(theta[i+N*2],  color='r', linestyle='--', linewidth=2., label='ML')
    if i==4:
        plt.legend()
    plt.title('length[{}]'.format(i+1))
    plt.ylabel('')
plt.legend()
plt.tight_layout()
plt.show()

sns.kdeplot(samps[:,-1],fill=True)
plt.axvline(noise_std, color='k', linestyle='-', linewidth=2., label='True')
plt.axvline(theta[-1], color='r', linestyle='--', linewidth=2., label='ML')
plt.title('noise std')
plt.show()
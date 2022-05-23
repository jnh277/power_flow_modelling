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
T = 100

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

# set line_z_pu
# network5.line_z_pu =

p_true = np.real(network.load_powers).flatten()
q_true = np.imag(network.load_powers).flatten()
theta0 = np.hstack([p_true * 0, q_true * 0, np.ones((1,))])


## try ML estimates of loads, std, powers
def neglogl(theta):
    busNo = network.busNo
    P = theta[:busNo]
    Q = theta[busNo:2*busNo]
    length = theta[2*busNo:2*busNo+busNo-1]
    noise_std = theta[-1]

    z3 = (1.2936 + 0.6713j) / network.zbase
    line_z_pu = np.array([z3 * length[0], z3 * length[1],
                          z3 * length[2], z3 * length[3]])
    network.line_z_pu = line_z_pu

    load_powers = np.expand_dims(P + 1j * Q, 1)  # craete one normalised load array
    network.load_powers = load_powers
    _, _, V_mag, _, _, _, _ = backwardforwardsweep(network)

    yhat = np.vstack([V_mag[observed,:], np.atleast_2d(P[observed]).T, np.atleast_2d(Q[observed]).T])

    err = (y - yhat).reshape([-1,])
    logl = np.sum(-0.5 * np.log(noise_std * np.sqrt(2*np.pi))- 0.5 * err ** 2 / noise_std ** 2)
    return -logl

def mse(theta):
    busNo = network.busNo
    P = theta[:busNo]
    Q = theta[busNo:2*busNo]
    length = theta[2*busNo:2*busNo+busNo-1]
    noise_std = theta[-1]

    z3 = (1.2936 + 0.6713j) / network.zbase
    line_z_pu = np.array([z3 * length[0], z3 * length[1],
                          z3 * length[2], z3 * length[3]])
    network.line_z_pu = line_z_pu

    load_powers = np.expand_dims(P + 1j * Q, 1)  # craete one normalised load array
    network.load_powers = load_powers
    _, _, V_mag, _, _, _, _ = backwardforwardsweep(network)

    yhat = np.vstack([V_mag[observed,:], np.atleast_2d(P[observed]).T, np.atleast_2d(Q[observed]).T])

    err = (y - yhat).reshape([-1,])
    mse = np.mean(err **2)
    return mse

length_true = network.length
theta0 = np.hstack([p_true, q_true, length_true, np.ones((1,))])
theta0 = theta0 * (0.8 + 0.4 *np.random.random(theta0.shape))       # start within +/0 20% of true

res = minimize(neglogl, theta0, bounds = [(0, None)]*14 +[(1e-6, None)])
theta = res.x

p_hat = theta[:busNo]
q_hat = theta[busNo:2 * busNo]
l_hat = theta[2 * busNo:2 * busNo + busNo - 1]
sig_hat = theta[-1]


## Try out metropolis hastings
M = 5000
samps = np.zeros((M, len(theta)))
samps[0] = theta
log_prob = np.zeros((M,))
log_prob[0] = -neglogl(samps[0])
H_inv = res.hess_inv.todense()
cov = H_inv * mse(samps[0])
L_cov = np.linalg.cholesky(cov)
mh_std = 1e-4
mh_factor = 0.025

accept_count = 0
for i in range(1,M):
    # propose =  np.random.multivariate_normal(samps[i-1], cov)
    # propose = samps[i-1] + mh_std * np.random.randn(len(theta),)
    propose = samps[i-1] + L_cov @ np.random.randn(len(theta),) * mh_factor


    inds = propose < 0
    propose[inds] = - propose[inds]
    logp = - neglogl(propose)  # no priors
    # if any(propose < 0):
        # logp = -1e8
    # compute acceptance probability
    accept_prob = min(1, np.exp(logp - log_prob[i-1]))
    if np.random.rand() < accept_prob:
        samps[i] = propose
        log_prob[i] = logp
        accept_count += 1
    else:
        samps[i] = samps[i-1]
        log_prob[i] = log_prob[i-1]

samps = samps[int(M/2):]

for i in range(1, 5):
    plt.subplot(2,2,i)
    sns.kdeplot(samps[:, i], label='MH', fill=True)
    plt.axvline(p_true[i], color='k', linestyle='-', linewidth=2., label='True')
    plt.axvline(theta[i],  color='r', linestyle='--', linewidth=2., label='ML')
    if i==4:
        plt.legend()
    plt.title('P[{}]'.format(i))
    plt.ylabel('')


plt.tight_layout()
plt.show()

for i in range(1, 5):
    plt.subplot(2,2,i)
    sns.kdeplot(samps[:, 5+i], label='MH', fill=True)
    plt.axvline(q_true[i], color='k', linestyle='-', linewidth=2., label='True')
    plt.axvline(theta[5+i],  color='r', linestyle='--', linewidth=2., label='ML')
    if i==4:
        plt.legend()
    plt.title('Q[{}]'.format(i))
    plt.ylabel('')


plt.tight_layout()
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(samps[:, 10+i], label='MH', fill=True)
    plt.axvline(length_true[i], color='k', linestyle='-', linewidth=2., label='True')
    plt.axvline(theta[10+i],  color='r', linestyle='--', linewidth=2., label='ML')
    if i==4:
        plt.legend()
    plt.title('length[{}]'.format(i))
    plt.ylabel('')


plt.tight_layout()
plt.show()

# sns.kdeplot(samps[:, -1], fill=True)
# plt.axvline(noise_std, color='k', linestyle='-', linewidth=2., label='True')
# plt.axvline(theta[-1], color='r', linestyle='--', linewidth=2., label='ML')
# plt.title('sigma')
# plt.show()

plt.plot(samps[:,-1])
plt.title('sigma trace')
plt.show()


## test hmc (current voltage model)

stan_data = {'N': busNo-1,
             'T': T,
             'y_mag': y_mag[1:],    # assuming slack stuff is known and not measured
             'y_p': y_p[1:],
             'y_q': y_q[1:],
             'real_zpk': np.real(network.zpk),
             'imag_zpk': np.imag(network.zpk),
             }

# def init_function():
#     output = dict(Vreal=np.real(V_all[1:]),
#                   Vimag=np.imag(V_all[1:]),
#                   l=length_true,
#                   )
#     return output

def init_function():
    output = dict(Vreal=np.ones(V_all[1:].shape),
                  Vimag=np.zeros(V_all[1:].shape),
                  l=0.2*np.ones((4,)),
                  )
    return output

f = open('test5.stan', 'r')
model_code = f.read()
posterior = stan.build(model_code, data=stan_data)
init = [init_function(), init_function(), init_function(), init_function()]
traces = posterior.sample(init=init, num_samples=1000, num_warmup=4000, num_chains=4, max_depth=13, delta=0.85)
# traces = posterior.sample(num_samples=4000, num_warmup=2000, num_chains=4, max_depth=10, delta=0.8)


for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(traces['Vreal'][i],fill=True)
    plt.axvline(np.real(V_all[i+1]),linewidth=2,linestyle='--', color='k')
    plt.title('Vreal[{}]'.format(i+1))
plt.tight_layout()
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(traces['Vimag'][i],fill=True)
    plt.axvline(np.imag(V_all[i+1]),linewidth=2,linestyle='--', color='k')
    plt.title('Vimag[{}]'.format(i + 1))
plt.tight_layout()
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(traces['l'][i],fill=True)
    plt.axvline(length_true[i],linewidth=2,linestyle='--', color='k')
    plt.title('length[{}]'.format(i + 1))
plt.tight_layout()
plt.show()


sns.kdeplot(traces['sig_e'][0],fill=True)
plt.axvline(noise_std,linewidth=2,linestyle='--', color='k')
plt.title('sigma noise')
plt.show()

for i in range(4):
    plt.plot(traces['sig_e'][0, i:400:4])
plt.show()

# for i in range(4):
#     plt.subplot(2,2,1)
#     sns.kdeplot(np.real(traces['shat'][0]),fill=True)
#     plt.axvline(p_true[i+1],linewidth=2,linestyle='--', color='k')
#     plt.title('P[{}]'.format(i))
# plt.tight_layout()
# plt.show()

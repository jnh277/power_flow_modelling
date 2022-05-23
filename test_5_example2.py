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
T = 500

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



## test hmc (power voltage model)

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
    output = dict(Vreal=np.real(V_all[1:].flatten())*(0.8+0.4*np.random.rand(4,)),
                  Vimag=np.imag(V_all[1:].flatten())*(0.8+0.4*np.random.rand(4,)),
                  l=length_true*(0.8+0.4*np.random.rand(4,)),
                  )
    return output

# def init_function():
#     output = dict(Vreal=Vreal_ML,
#                   Vimag=Vimag_ML,
#                   l=l_ML,
#                   sig_e=sig_ML,
#                   )
#     return output

f = open('test5.stan', 'r')
model_code = f.read()
posterior = stan.build(model_code, data=stan_data)
init = [init_function(), init_function(), init_function(), init_function()]
traces = posterior.sample(init=init, num_samples=1000, num_warmup=4000, num_chains=4, max_depth=10, delta=0.9)
# traces = posterior.sample(num_samples=4000, num_warmup=2000, num_chains=4, max_depth=10, delta=0.8)


for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(traces['Vreal'][i],fill=True)
    plt.axvline(np.real(V_all[i+1]),linewidth=2,linestyle='--', color='k')
    # plt.axvline(Vreal_ML[i], linewidth=2, linestyle='--', color='r')
    plt.title('Vreal[{}]'.format(i+1))
plt.tight_layout()
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(traces['Vimag'][i],fill=True)
    plt.axvline(np.imag(V_all[i+1]),linewidth=2,linestyle='--', color='k')
    # plt.axvline(Vimag_ML[i], linewidth=2, linestyle='--', color='r')
    plt.title('Vimag[{}]'.format(i + 1))
plt.tight_layout()
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(traces['l'][i],fill=True)
    plt.axvline(length_true[i],linewidth=2,linestyle='--', color='k')
    # plt.axvline(l_ML[i], linewidth=2, linestyle='--', color='r')
    plt.title('length[{}]'.format(i + 1))
plt.tight_layout()
plt.show()


sns.kdeplot(traces['sig_e'][0],fill=True)
plt.axvline(noise_std,linewidth=2,linestyle='--', color='k')
plt.title('sigma noise')
plt.show()

for i in range(4):
    plt.plot(traces['sig_e'][0, i:1000:4])
plt.show()

for i in range(4):
    plt.plot(traces['l'][1, i:1000:4])
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(traces['p_hat'][i],fill=True)
    plt.axvline(p_true[i+1],linewidth=2,linestyle='--', color='k')
    plt.title('P[{}]'.format(i))
plt.tight_layout()
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    sns.kdeplot(traces['p_hat'][i],fill=True)
    plt.axvline(p_true[i+1],linewidth=2,linestyle='--', color='k')
    plt.title('P[{}]'.format(i))
plt.tight_layout()
plt.show()


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.special import binom, comb
from tqdm import tqdm
from numba import njit
import time
# Grab parameters from params file.
import os

cite()
params = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True,
    'figure.figsize': (16, 12)
}

plt.rcParams.update(params)
# Set some resource limits. Unfortunately the monte-carlo simulations don't seem to give a shit about the thread
# assignment here, and uses every core it can get its hands on. To fix that I'd have to go digging into the
# multiprocessing library files, which doesn't seem best pracrice. Just worth bearing in mind that MC is getting
# access to a lot more resources!
threads = 8
os.environ['NUMEXPR_MAX_THREADS'] = '{}'.format(threads)
os.environ['NUMEXPR_NUM_THREADS'] = '{}'.format(threads)
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)

"""Function for setting up Heisenberg Spin Chain"""
g=1
J=1
N=2

def integrate_setup(N,g,k=0):
    si = qeye(2)
    sz = sigmaz()
    sx = sigmax()
    sm = sigmam()

    op_list=[]
    wf_list = []
    wf=basis(2,0)
    for m in range(N):
        wf_list.append(wf)
        op_list.append(si)
    si_tensor=tensor(op_list)
    wf_tensor=tensor(wf_list)
    # print(wf)
    # init=wf_tensor*wf_tensor.dag()
    # print(init)
    sx_list = []
    sz_list = []
    # init=si_tensor.unit()
    # init=Qobj(np.ones())
    init=wf_tensor.unit()
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))


    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N-1):
        H+=-J*sz_list[n]*sz_list[n+1]
        for m in range(N):
            H += -J*g*sx_list[n]
    E_0=H.eigenenergies(eigvals=1)[0]
    print(H.eigenenergies(eigvals=4)[:])
    H+=k*np.abs(E_0)
        # H=(2**N)*H/H.tr()


    print('Hamiltonian and dissipators constructed. shape of H is {}, and Hermitian check returned {}'.format(H.shape,H.isherm))

    return H, sz_list, sx_list,init,E_0

lam=1
H,sz_list,sx_list,wf_init,E_0=integrate_setup(N,g,k=0)
H_shift,sz_shift,sx_shift,wf_shift,E_shift=integrate_setup(N,g,k=lam)
rho_init=wf_init*wf_init.dag()
H_energies,H_kets=H.eigenstates()
H_shift_energies,H_shift_kets=H_shift.eigenstates()
# print(H_energies)
# print(H_shift_energies)
# print("spectrum w/shift")
# print(H_energies-H_shift_energies-lam*np.abs(E_0))
# for k in range(len(H_kets)):
#     print('ket {}'.format(k))
#     for j in range(len(H_kets)):
#         print(H_kets[k].overlap(H_shift_kets[j]))
# H_root=np.sqrt(H_energies[-1])*H_kets[-1]*H_kets[-1].dag()
# print(H.dims)
# for j in tqdm(range(len(H_energies)-1)):
#     H_root+=np.sqrt(H_energies[j])*H_kets[j]*H_kets[j].dag()

def thermal(H,beta):
    state = ((-beta*H).expm())
    return state.unit()
def Bloch_one_step(H,rho,dt):
    rho_new=rho-0.5*dt*(H*rho + rho*H)
    return rho_new.unit()

def S_one_step(H,rho,dt):
    rho_new=rho-1j*dt*(H*rho - rho*H)
    return rho_new.unit()

def overlap_one_step(H,rho,dt):
    U=(1j*H*np.sqrt(dt/2)).expm()
    rho_new=U*rho*U
    return (rho_new+rho_new.dag()).unit()
def forward_step(H,wf,dt):
    U=(1j * H * np.sqrt(dt/2)).expm()
    wf_new=U*wf
    return wf_new.unit()
beta = 4
steps= 600##needs to be even
wf_steps=steps
betas,dt=np.linspace(0,beta,num=steps,retstep=True)
print(dt)
times,dt2=np.linspace(0,60,num=steps,retstep=True)

Bloch_list=[rho_init]
S_list=[rho_init]

for j in tqdm(range(len(betas)-1)):
    # print(j
    Bloch_list.append(Bloch_one_step(H,Bloch_list[-1],dt))
    S_list.append(S_one_step(H, S_list[-1], dt))
#
# print(overlap_list)
E_Bloch=[]
E_S=[]
# sz_Bloch=[]
# sx_Bloch=[]
for j in range(len(betas)):
    E_Bloch.append(expect(Bloch_list[j],sz_list[0]))
    E_S.append(expect(S_list[j], sz_list[0]))

plt.subplot(211)
plt.plot(betas,E_S)
plt.ylabel('$\\langle\\hat{\sigma}_x\\rangle $')
plt.xlabel('$t$')
plt.subplot(212)



plt.plot(betas,E_Bloch)
plt.ylabel('$\\langle\\hat{\sigma}_x\\rangle $')
plt.xlabel('$\\tau$')
plt.show()

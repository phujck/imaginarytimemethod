
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
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
N=4

def integrate_setup(N,g):
    si = qeye(2)
    sz = sigmaz()
    sx = sigmax()
    sm = sigmam()

    op_list=[]
    wf_list = []
    wf=basis(2,1)
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
    print(H.eigenenergies)
    ""
    # if E_0<0:
    #     H-=1*E_0
    #     H=(2**N)*H/H.tr()


    print('Hamiltonian and dissipators constructed. shape of H is {}, and Hermitian check returned {}'.format(H.shape,H.isherm))

    return H, sz_list, sx_list,init

H,sz_list,sx_list,wf_init=integrate_setup(N,g)
rho_init=wf_init*wf_init.dag()
H_energies,H_kets=H.eigenstates()
H_root=np.sqrt(H_energies[-1])*H_kets[-1]*H_kets[-1].dag()
print(H.dims)
for j in tqdm(range(len(H_energies)-1)):
    H_root+=np.sqrt(H_energies[j])*H_kets[j]*H_kets[j].dag()

def thermal(H,beta):
    state = ((-beta*H).expm())
    return state.unit()
def Bloch_one_step(H,rho,dt):
    rho_new=rho-0.5*dt*(H*rho + rho*H)
    return rho_new.unit()

def overlap_one_step(H,rho,dt):
    U=(1j*H*np.sqrt(dt)).expm()
    rho_new=U*rho*U
    return (rho_new+rho_new.dag()).unit()
def forward_step(H,wf,dt):
    U=(1j * H * np.sqrt(dt)).expm()
    wf_new=U*wf
    return wf_new.unit()
beta = 1
steps= 1000##needs to be even
wf_steps=steps
betas,dt=np.linspace(0,beta,num=steps,retstep=True)
print(dt)

wf_root_list=[wf_init]
wf_list=[wf_init]
Bloch_list=[rho_init]
overlap_list=[wf_init.overlap(wf_init)]
E_overlap_list=[(H*wf_init).overlap(wf_init)]
sx_overlap_list=[(sx_list[0]*wf_init).overlap(wf_init)]
print(sx_overlap_list)
for j in tqdm(range(len(betas)-1)):
    wf_root_list.append(forward_step(H/np.sqrt(2),wf_root_list[-1],dt))
    wf_root_list.insert(0,forward_step(-H/ np.sqrt(2), wf_root_list[0], dt))
    overlap_list.append(2*(wf_root_list[0].overlap(wf_root_list[-1])).real)
    E_overlap_list.append(2*(H*wf_root_list[0]).overlap(wf_root_list[-1]).real)
    sx_overlap_list.append(2*(sx_list[0] * wf_root_list[0]).overlap(wf_root_list[-1]).real)
    Bloch_list.append(Bloch_one_step(H,Bloch_list[-1],dt))

# print(overlap_list)
# print(E_overlap_list)
partition_list=[1]
E_expect=[E_overlap_list[0]]
sx_expect=[sx_overlap_list[0]]
print(len(overlap_list))
print(steps)
for j in range(3,steps,2):
    print('NEW LIST')
    partition=0
    E=0
    s=0
    if j < 10:
        l=j
    else:
        l=j
    for k in range(0,l,2):
        # print(binom(j,int(k/2)))
        # print(wf_root_list[central_index-(j-k)].dag().overlap(wf_root_list[central_index+(j-k)]))
        partition+= binom(j,int(k/2))*overlap_list[j-k]
        E+=binom(j,int(k/2))*E_overlap_list[j-k]
        s+=binom(j,int(k/2))*sx_overlap_list[j-k]
        # print(partition)
    partition=np.abs(partition)
        # norm+=overlap_list[]
    partition_list.append(partition)
    E_expect.append(E/partition)
    sx_expect.append(s/partition)
# print(partition_list)
# print(E_expect)
# print(len(betas))
# print(len(wf_root_list))
# print(thermal_list[1])
# print(overlap_list)
E_thermal=[]
E_square=[]
sz_thermal=[]
sx_thermal=[]
sx_square=[]
E_Bloch=[]
sz_Bloch=[]
sx_Bloch=[]
for j in range(len(betas)):
    E_Bloch.append(expect(Bloch_list[j],H))
    sz_Bloch.append(expect(sz_list, Bloch_list[j]))
    sx_Bloch.append(expect(sx_list, Bloch_list[j]))
#

# plt.plot(betas[::2],E_expect,label='$\\mathcal{L}^n[\\rho],\hat{A}=\sqrt{\hat{H}}$')
plt.plot(betas[::2],E_expect,label='$\\mathcal{L}^n[\\rho]$')
# plt.plot(betas,E_thermal,label='$\\mathcal{L}^n[\\rho],\hat{A}=\sqrt{\hat{H}}$')
plt.plot(betas,E_Bloch,label='Bloch',linestyle='--')
plt.ylabel('$\\langle\\hat{H}\\rangle$')
plt.plot(betas,np.ones(len(betas))*H_energies[0],linestyle='-.',label='$E_0$')
plt.legend()
plt.xlabel('$\\beta$')
plt.show()

plt.subplot(211)
# plt.plot(betas[::2],E_expect,label='$\\mathcal{L}^n[\\rho],\hat{A}=\sqrt{\hat{H}}$')
plt.plot(betas[::2],E_expect+H_energies[0],label='$\\mathcal{L}^n[\\rho]$')
# plt.plot(betas,E_thermal,label='$\\mathcal{L}^n[\\rho],\hat{A}=\sqrt{\hat{H}}$')
plt.plot(betas,E_Bloch+H_energies[0],label='Bloch',linestyle='--')
plt.ylabel('$\\langle\\hat{H}\\rangle$')
plt.plot(betas,np.ones(len(betas))*H_energies[0],linestyle='-.',label='$E_0$')
plt.legend()
plt.subplot(212)
plt.plot(betas[::2],sx_expect)
plt.plot(betas,np.array(sx_Bloch)[:,0],linestyle='--')
plt.ylabel('$\\langle\\hat{\sigma}^{(0)}_x\\rangle$')
plt.xlabel('$\\beta$')
plt.show()
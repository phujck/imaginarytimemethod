##########################################################
# Simulation to compare low-rank approximation to exact  #
# and Monte-Carlo solutions to Lindblad equation for a   #
# Heisenberg spin-chain. Code is based off the QuTiP     #
# example here: https://bit.ly/342ZwC5                   #
# USE THIS FILE ONLY TO ACTUALLY RUN SIMULATION.         #
# SET SYSTEM PARAMETERS IN param.py                      #
# VIEW RESULTS IN analysis.py
##########################################################

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
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
J=6
N=6

def integrate_setup(N,g):
    si = qeye(2)
    sz = sigmaz()
    sx = sigmax()
    sm = sigmam()

    op_list=[]
    wf_list = []
    wf=basis(2,1)

    print(wf)
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
    init=si_tensor.unit()
    # init=Qobj(np.ones())
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
    if E_0<0:
        H-=1.5*E_0
        H=(2**N)*H/H.tr()


    print('Hamiltonian and dissipators constructed. shape of H is {}, and Hermitian check returned {}'.format(H.shape,H.isherm))

    return H, sz_list, sx_list,init

H,sz_list,sx_list,rho_init=integrate_setup(N,g)

H_energies,H_kets=H.eigenstates()
H_root=np.sqrt(H_energies[-1])*H_kets[-1]*H_kets[-1].dag()
print(H.dims)
for j in range(len(H_energies)-1):
    H_root+=np.sqrt(H_energies[j])*H_kets[j]*H_kets[j].dag()
# rho_init=(H_kets[0]*H_kets[0].dag()+2*H_kets[2]*H_kets[2].dag()).unit()
print(H_root*H_root-H)
# print(H_root)
# print(H_energies)
# print(Qobj(np.sqrt(H)))
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

beta = 20
steps= 50
betas,dt=np.linspace(0,beta,num=steps,retstep=True)
print(dt)

thermal_list=[thermal(H,j) for j in betas]
# E_thermal=[thermal(H,j) for j in betas]
print(thermal(H,0))
# print(thermal_list)

Bloch_list=[rho_init]
overlap_list=[rho_init]
overlap_square=[rho_init]
for j in range(len(betas)-1):
    Bloch_list.append(Bloch_one_step(H,Bloch_list[-1],dt))
    overlap_list.append(overlap_one_step(H_root/np.sqrt(2),overlap_list[-1],dt))
    overlap_square.append(overlap_one_step(H/np.sqrt(2), overlap_list[-1], dt))
# print(len(betas))
# print(thermal_list[1])
print(overlap_list[1])
E_thermal=[]
E_square=[]
sz_thermal=[]
sx_thermal=[]
sx_square=[]
E_Bloch=[]
sz_Bloch=[]
sx_Bloch=[]
for j in range(len(betas)):
    E_thermal.append(expect(overlap_list[j],H))
    E_square.append(expect(overlap_square[j],H))
    sz_thermal.append(expect(overlap_list[j],sz_list))
    sx_thermal.append(expect(sx_list, overlap_list[j]))
    sx_square.append(expect(sx_list, overlap_square[j]))
    E_Bloch.append(expect(thermal_list[j],H))
    sz_Bloch.append(expect(sz_list, Bloch_list[j]))
    sx_Bloch.append(expect(sx_list, thermal_list[j]))


plt.subplot(211)
plt.plot(betas,E_square,label='$\\mathcal{L}^n[\\rho],\hat{A}=\hat{H}$')
plt.plot(betas,E_thermal,label='$\\mathcal{L}^n[\\rho],\hat{A}=\sqrt{\hat{H}}$')
plt.plot(betas,E_Bloch,label='Bloch',linestyle='--')
plt.ylabel('$\\langle\\hat{H}\\rangle$')
plt.plot(betas,np.ones(len(betas))*H_energies[0],linestyle='-.',label='$E_0$')
plt.legend()
plt.subplot(212)
plt.plot(betas,np.array(sx_square)[:,0])
plt.plot(betas,np.array(sx_thermal)[:,0])
plt.plot(betas,np.array(sx_Bloch)[:,0],linestyle='--')
plt.ylabel('$\\langle\\hat{\sigma}^{(0)}_x\\rangle$')
plt.xlabel('$\\beta$')
plt.show()


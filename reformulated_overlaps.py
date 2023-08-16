
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

def overlap_one_step(H,rho,dt):
    U=(1j*H*np.sqrt(dt/2)).expm()
    rho_new=U*rho*U
    return (rho_new+rho_new.dag()).unit()
def forward_step(H,wf,dt):
    U=(1j * H * np.sqrt(dt/2)).expm()
    wf_new=U*wf
    return wf_new.unit()
beta = 0.5
steps= 400##needs to be even
wf_steps=steps
betas,dt=np.linspace(0,beta,num=steps,retstep=True)
print(dt)

wf_root_list=[wf_init]
wf_shift_list=[wf_init]
wf_list=[wf_init]
Bloch_list=[rho_init]
overlap_list=[wf_init.overlap(wf_init)]
E_overlap_list=[(H*wf_init).overlap(wf_init)]
sx_overlap_list=[(sx_list[0]*wf_init).overlap(wf_init)]
overlap_shift_list=[wf_init.overlap(wf_init)]
E_overlap_shift_list=[(H_shift*wf_init).overlap(wf_init)]
for j in tqdm(range(len(betas)-1)):
    # print(j)
    phase = np.exp(2j *lam*np.abs(E_0)*np.sqrt(dt/2)*(j+1))
    wf_root_list.append(forward_step(H,wf_root_list[-1],dt))
    wf_root_list.insert(0,forward_step(-H, wf_root_list[0], dt))
    wf_shift_list.append(forward_step(H_shift, wf_shift_list[-1], dt))
    wf_shift_list.insert(0, forward_step(-H_shift, wf_shift_list[0], dt))
    overlap_list.append(wf_root_list[0].overlap(wf_root_list[-1]))
    overlap_shift_list.append(wf_shift_list[0].overlap(wf_shift_list[-1]))
    E_overlap_list.append((H*wf_root_list[0]).overlap(wf_root_list[-1]))
    E_overlap_shift_list.append((H_shift*wf_shift_list[0]).overlap(wf_shift_list[-1]))
    sx_overlap_list.append((sx_list[0] * wf_root_list[0]).overlap(wf_root_list[-1]))
    Bloch_list.append(Bloch_one_step(H,Bloch_list[-1],dt))
#
# phases=[]
# angle=[]
# for k in range(len(overlap_shift_list)):
#     phases.append((E_overlap_list[k]+lam*np.abs(E_0)*overlap_list[k])/(E_overlap_shift_list[k]))
#
#     if k > 0:
#         angle.append((np.angle(phases[-1])-np.angle(phases[-2]))/np.pi)
# print(phases)
# print(angle)
# plt.plot(angle)
# plt.show()
# def argand(a):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     for x in range(len(a)):
#         plt.plot([0,a[x].real],[0,a[x].imag],'ro-',label='python')
#     limit=1 # set limits for axis
#     plt.xlim((-limit,limit))
#     plt.ylim((-limit,limit))
#     plt.ylabel('Imaginary')
#     plt.xlabel('Real')
#     plt.show()
# argand(phases)


# print(E_overlap_list)
#
# print(partition_list)
# print(E_expect)
# print(len(betas))
# print(len(wf_root_list))
# print(thermal_list[1])
# print(overlap_list)
# E_Bloch=[]
# sz_Bloch=[]
# sx_Bloch=[]
# for j in range(len(betas)):
#     E_Bloch.append(expect(Bloch_list[j],H))
#     sz_Bloch.append(expect(sz_list, Bloch_list[j]))
#     sx_Bloch.append(expect(sx_list, Bloch_list[j]))
#
def results(lam,dt):
    partition_list = [1]
    E_expect = [E_overlap_list[0]]
    sx_expect = [sx_overlap_list[0]]
    # print(len(overlap_list))
    # print(steps)
    for j in range(3, steps, 2):
        # print(steps-j)
        partition = 0
        E = 0
        s = 0
        for k in range(0, j, 2):
            phase=np.exp(2j *lam*np.sqrt(dt/2)*(j-k))
            # print(np.abs(phase))
            # print(binom(j,int(k/2)))
            # print(wf_root_list[central_index-(j-k)].dag().overlap(wf_root_list[central_index+(j-k)]))
            partition += binom(j, int(k / 2))*2 * (phase * overlap_list[j - k]).real
            E += binom(j, int(k / 2)) *2* (phase*(E_overlap_list[j - k]+lam*overlap_list[j - k])).real
            # print(partition)
        partition = np.abs(partition)
        # norm+=overlap_list[]
        partition_list.append(partition)
        E_expect.append(E / partition)
    return E_expect

def results_final(lam,dt):
    partition_list = [1]
    E_expect = [E_overlap_list[0]]
    # print(len(overlap_list))
    # print(steps)
    j_f=steps-1
    partition = 0
    E = 0
    s = 0
    for k in range(0,j_f , 2):
        phase=np.exp(2j *lam*np.sqrt(dt/2)*(j_f-k))
        # print(np.abs(phase))
        # print(binom(j,int(k/2)))
        # print(wf_root_list[central_index-(j-k)].dag().overlap(wf_root_list[central_index+(j-k)]))
        partition += binom(j_f, int(k / 2))*2 * (phase * overlap_list[j_f- k]).real
        E += binom(j_f, int(k / 2)) *2* (phase*(E_overlap_list[j_f- k]+lam*overlap_list[j_f- k])).real
        # print(partition)
    partition = np.abs(partition)
    # norm+=overlap_list[]
    partition_list.append(partition)
    E_expect.append(E / partition)
    return E_expect

plt.subplot(211)
l=0
colors = pl.cm.jet(np.linspace(0,1,len(H_energies[::2])+2))
for f in H_energies[::2]:
    print(f)
    f=-f
    # print('ratio')
    # print(f * np.sqrt(dt / 2)/np.pi)
    # plt.plot(betas[::2],results(f,dt),label='$\\mathcal{L}^n[\\rho],\hat{A}=\hat{H}$')
    # lines=plt.plot(betas[2::2],results(f,dt)[1:]-f,label='$\\lambda=E_{{{:2d}}}$'.format(l))
    if l==0 or l==len(H_energies[::2])-1:
        plt.plot(betas[2::2], results(f, dt)[1:] - f, color=colors[l+1], label='$\\lambda=E_{{{:2d}}}$'.format(l),linewidth=3.0)
    plt.plot(betas[2::2],results(f,dt)[1:]-f,color=colors[l],linewidth=3.0)
    plt.plot(betas, -np.ones(len(betas)) *f, linestyle='-.',color=colors[l+1])
    l+=1
# for k in range(len(H_energies)):
#     plt.plot(betas, np.ones(len(betas)) * H_energies[k], linestyle='-.',color='black')
# plt.plot(betas, np.ones(len(betas)) * H_energies[0], linestyle='-.', label='$E_0$')
# plt.plot(betas,E_thermal,label='$\\mathcal{L}^n[\\rho],\hat{A}=\sqrt{\hat{H}}$')
# plt.plot(betas,E_Bloch+lam,label='Bloch',linestyle='--')
# plt.colorbar()
plt.legend()
plt.ylabel('$\\langle\\hat{H}\\rangle$ (arb. u.)')
plt.xlabel('$\\beta$')
# for lam in H_energies[:8:2]:
#     lam=np.abs(lam)+0.01
plt.subplot(212)
lam=0.2
# test_lams=[-2*lam,-lam,0,lam,2*lam]
# test_lams=np.linspace(0,10*lam,10)
# print(H_energies)
# l=0
energies=[]
lambdas=np.linspace(-1.5*np.abs(E_0),1.5*np.abs(E_0),100)
for l in lambdas:
    energies.append(results_final(l,dt)[-1])

h = 0
for f in H_energies[::2]:

    plt.plot(lambdas/np.abs(E_0), f*np.ones(len(lambdas))+lambdas,linestyle='-.', color=colors[h+1])
    h+=1
plt.plot(lambdas/np.abs(E_0),energies, color='black',label='$\\langle\\hat{H}\\rangle+\lambda$',linewidth=3.0)

plt.ylabel('$\\langle\\hat{H}\\rangle + \\lambda$ (arb. u.)')
plt.xlabel('$\lambda/\\left|E_0\\right|$')
# plt.legend()
plt.savefig('method_exact.pdf')
plt.show()
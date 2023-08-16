import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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



N = 100000
# x = np.random.randn(N)
x = stats.norm.rvs(size=N)
num_bins = 8
plt.hist(x, bins=[0,0.5,1,1.5,2], edgecolor='black',facecolor='blue', alpha=0.3,density=True)

y = np.linspace(-4, 4, 1000)
bin_width = (x.max() - x.min()) / num_bins
n, p = 10, 0.1
plt.plot(y, 4*stats.norm.pdf(np.sqrt(4)*y)+1,color='black',linestyle='dashed')
x = np.arange(stats.binom.ppf(0.01, n, p),
              stats.binom.ppf(0.99, n, p))
print(x)
plt.plot(0.5*(x+0.5), 8*stats.binom.pmf(x, n, p,-2)+1, 'bo', ms=8, label='binom pmf')
plt.yticks([])
plt.xticks([])
plt.xlim([0,2])
plt.xlabel('$\\beta$')
plt.ylabel('$\\langle\\hat{H}\\rangle$')
plt.tight_layout()
plt.show()
# for lam in H_energies[:8:2]:
#     lam=np.abs(lam)+0.01

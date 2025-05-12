#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
from scipy import linalg

#########################################################
# 1. OPTIONAL: Environment variables for BLAS, etc.
#########################################################
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


#########################################################
# 2. TOL colormap class (copied from your snippet)
#########################################################
class TOLcmaps(object):
    """
    Class TOLcmaps definition. Used for generating colormaps for plotting
    """

    def __init__(self):
        self.cmap = None
        self.cname = None
        self.namelist = ('rainbow_PuRd',)
        self.funcdict = dict(
            zip(self.namelist,
                (self.__rainbow_PuRd,)
                )
        )

    def __rainbow_PuRd(self):
        """
        Define colormap 'rainbow_PuRd'.
        """
        clrs = ['#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
                '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
                '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
                '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
                '#DF4828', '#DA2222']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')

    def get(self, cname='rainbow_PuRd', lut=None):
        """
        Return requested colormap, default is 'rainbow_PuRd'.
        """
        self.cname = cname
        if cname not in self.namelist:
            cname = 'rainbow_PuRd'
            print('*** Warning: requested colormap not defined, known colormaps are {}. Using {}.'.format(
                self.namelist, cname))
        self.funcdict[cname]()
        return self.cmap


def tol_cmap(colormap=None, lut=None):
    obj = TOLcmaps()
    if colormap is None:
        return obj.namelist
    return obj.get(colormap, lut)


#########################################################
# 3. MATPLOTLIB defaults for nice plots
#########################################################
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#########################################################
# 4. Example post-processing function
#########################################################
def gauss_partition_calc(times, weights, z_vals, O_vals, lam,weight_number):
    """
    Placeholder function to mimic ITQDE's 'shifted Hamiltonian' calculation using
    Gauss-Hermite data.
    Modify according to your actual formula (E1) in the preprint.

    :param times: array of shape (n_nodes,)
    :param weights: array of shape (n_nodes,)
    :param z_vals: array of shape (n_nodes,) => 'partition overlaps'
    :param O_vals: array of shape (n_nodes,) => 'obs overlaps'
    :param lam: shift parameter
    :return: a single value for the 'expectation'
             (in your code you might want to return a list for each tau?)
    """
    # For demonstration, let's do a simple ratio:
    #   Expect = (Sum_k w_k e^{2i lam times[k]} O_vals[k]) / (Sum_k w_k e^{2i lam times[k]} z_vals[k])
    #
    # Possibly multiply or divide by 1/sqrt(pi) if following Gauss-Hermite integration rules.
    num = 0 + 0j
    den = 0 + 0j
    w_n=0
    for w, t, z, O in zip(weights, times, z_vals, O_vals):
        print(w_n)
        w_n+=1
        phase = np.exp(2j * lam * t)
        O_lam = O
        num += w * 2*np.real((phase * O_lam))
        den += w * 2*np.real(phase * z)
        if w_n >= weight_number:
            break
    # den =1

    if abs(den) < 1e-14:
        return np.nan
    return (num / den)

#########################################################

def main():
    parser = argparse.ArgumentParser(description="Analyze ITQDE Gauss-Hermite data.")
    parser.add_argument("--Lx", type=int, default=2, help="Lattice dimension x.")
    parser.add_argument("--Ly", type=int, default=2, help="Lattice dimension y.")
    parser.add_argument("--J", type=float, default=1.0, help="Hopping strength.")
    parser.add_argument("--U", type=float, default=2.0, help="On-site interaction.")
    parser.add_argument("--mu", type=float, default=0.5, help="Chemical potential.")
    parser.add_argument("--tau", type=float, default=1.0, help="Imag time scale.")
    parser.add_argument("--n_nodes", type=int, default=10, help="Number of Gauss-Hermite quadrature nodes.")
    parser.add_argument("--main_path", type=str, default="2D_FH_testing_gauss",
                        help="Directory where NPZ file is stored.")
    parser.add_argument("--lam_points", type=int, default=101,
                        help="Number of lambda points in the sweep.")
    args = parser.parse_args()
    args.tau=5
    args.n_nodes=1000
    args.U=2.0
    args.Ly=2

    # Reproduce the same naming convention from your generation script
    # e.g. "2DFH_gauss_Lx{Lx}_Ly{Ly}_NfUp{N_up}_NfDown{N_down}_J{J}_U{U}_mu{mu}_tau{tau}_n{n_nodes}.npz"
    Lx = args.Lx
    Ly = args.Ly
    N_2d = Lx * Ly
    N_up = N_2d // 2
    N_down = N_2d // 2

    filename = (f"2DFH_gauss_Lx{Lx}_Ly{Ly}_NfUp{N_up}_NfDown{N_down}"
                f"_J{args.J}_U{args.U}_mu{args.mu}"
                f"_tau{args.tau}_n{args.n_nodes}.npz")
    data_path = os.path.join(args.main_path, filename)

    if not os.path.isfile(data_path):
        print(
            f"Error: NPZ file {data_path} does not exist. Check your parameters or run the generation script first.")
        sys.exit(1)

    print(f"Loading data from {data_path}")
    data = np.load(data_path, allow_pickle=True)
    times = data["times"]
    weights = data["weights"]
    print(times)
    print(weights)
    z_vals = data["z"]
    O_vals = data["O"]
    E = data["spectrum"]
    print(f"  -> times.shape={times.shape}, weights.shape={weights.shape}")
    print(f"  -> z_vals.shape={z_vals.shape}, O_vals.shape={O_vals.shape}")
    print(f"  -> E_spectrum.shape={E.shape}")

    # Inspect min/max of the spectrum for reference
    E_min, E_max = E[0], E[-1]
    print(f"H spectrum: E_min={E_min:.4f}, E_max={E_max:.4f}")
    lambda_scale=1.1
    def exp_h(lam):
        e=0
        z=0
        for energy in E:
            z+=np.exp(-args.tau*(energy-lam)**2)
            e  +=energy*np.exp(-args.tau*(energy-lam)**2)
        return e/z

    # ==================== 5b. Compute a sweep over lambda ====================
    lam_values = np.linspace(-lambda_scale * abs(E_min), lambda_scale * abs(E_min), 300)
    expt_vals = []
    val2=[]
    exps=[]
    for lam in lam_values:
        val = gauss_partition_calc(times, weights, z_vals, O_vals, lam,weight_number=12)
        val_2 = gauss_partition_calc(times, weights, z_vals, O_vals, lam, weight_number=35)
        expt_vals.append(val)
        exps.append(exp_h(lam))
        val2.append(val_2)

    # ==================== 5c. Plot and compare with exact energies ====================
    plt.figure(figsize=(6,4))
    # TOL colormap for the exact energies
    cmap = tol_cmap('rainbow_PuRd').resampled(len(E))
    # build discrete colors
    e_colors = [cmap(i/(len(E)-1)) for i in range(len(E))]

    # Plot the final expectation vs. lambda


    # # Overplot each exact E as a horizontal line shifted by +λ (like your old code does)
    for i, e_val in enumerate(E):
        c = e_colors[i]
    # plt.plot(lam_values / abs(E_min), (e_val * np.ones(lam_values.shape)),
             # '--', color=c, label=f"E_{i}" if i in (0, len(E) - 1) else None)
        plt.plot(lam_values / abs(E_min), (e_val * np.ones(lam_values.shape)),
             '--', color=c)
    plt.plot(lam_values / abs(E_min), expt_vals, label="ITQDE-30 overlaps")
    plt.plot(lam_values / abs(E_min), val2, 'k', label="ITQDE-60 overlaps")
    plt.ylim(E_min-1.5, E_max+1.5)
    # plt.plot(lam_values / abs(E_min),exps,'--',label='exact')
    # plt.plot(lam_values / abs(E_min), exps)

    plt.xlabel(r"$\lambda / |E_0|$")
    plt.ylabel(r"$\langle H^{(\lambda)} \rangle(\tau \to \infty)$")
    plt.title("Gauss-Quadrature ITQDE Shifted Observables")
    plt.legend(loc="best")
    plt.tight_layout()


    # Save figure if desired
    plt.savefig("gauss_quad_sweep.pdf", dpi=300)
    print("Plot saved as gauss_quad_sweep.png")
    plt.show()
if __name__ == "__main__":
    main()
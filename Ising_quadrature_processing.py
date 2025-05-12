#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

################################################################
# 1) TOL colormap (same as before)
################################################################
class TOLcmaps(object):
    def __init__(self):
        self.cmap = None
        self.cname = None
        self.namelist = ('rainbow_PuRd',)
        self.funcdict = dict(
            zip(self.namelist, (self.__rainbow_PuRd,))
        )
    def __rainbow_PuRd(self):
        clrs = [
            '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
            '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
            '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
            '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
            '#DF4828', '#DA2222'
        ]
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')

    def get(self, cname='rainbow_PuRd', lut=None):
        self.cname = cname
        if cname not in self.namelist:
            cname = 'rainbow_PuRd'
            print(f'*** Warning: requested colormap not defined, '
                  f'known colormaps are {self.namelist}. Using {cname}.')
        self.funcdict[cname]()
        return self.cmap

def tol_cmap(colormap=None, lut=None):
    obj = TOLcmaps()
    if colormap is None:
        return obj.namelist
    return obj.get(colormap, lut)

################################################################
# 2) Matplotlib styling
################################################################
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

################################################################
# 3) Example post-processing function
################################################################
def gauss_partition_calc(times, weights, z_vals, O_vals, lam):
    """
    Dummy function for ratio-based expectation from Gauss-Hermite data.
    Modify for your actual ITQDE formula.
    """
    num = 0+0j
    den = 0+0j
    for w, t, z, O in zip(weights, times, z_vals, O_vals):
        phase = np.exp(1j * lam * t)  # e^{2 i λ t}
        num += w * phase * O
        den += w * phase * z
    if abs(den) < 1e-14:
        return np.nan
    return (num/den).real

################################################################
# 4) Main
################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Analyze 1D Ising Gauss-Hermite data via a lambda sweep."
    )
    parser.add_argument("--L", type=int, default=6,
                        help="Chain length (number of spins).")
    parser.add_argument("--J", type=float, default=3.0,
                        help="Ising coupling.")
    parser.add_argument("--g", type=float, default=1.0,
                        help="Transverse field strength.")
    parser.add_argument("--bc", type=str, default="periodic",
                        help="Boundary condition, e.g. 'open' or 'periodic'.")
    parser.add_argument("--tau", type=float, default=10.0,
                        help="Imag time scale.")
    parser.add_argument("--n_nodes", type=int, default=10,
                        help="Number of Gauss-Hermite quadrature points.")
    parser.add_argument("--main_path", type=str, default="1D_Ising_gauss",
                        help="Directory containing the NPZ file.")
    parser.add_argument("--lam_points", type=int, default=101,
                        help="Number of points in the lambda sweep.")
    args = parser.parse_args()
    args.tau = 5.0
    args.n_nodes = 10
    # Reconstruct the same filename pattern you used in generation:
    # "1DIsing_gauss_L{L}_J{J}_g{g}_bc{bc}_tau{tau}_n{n_nodes}.npz"
    filename = (
        f"1DIsing_gauss_L{args.L}_J{args.J}_g{args.g}_bc{args.bc}"
        f"_tau{args.tau}_n{args.n_nodes}.npz"
    )
    data_path = os.path.join(args.main_path, filename)

    # Check file existence
    if not os.path.isfile(data_path):
        print(f"Error: NPZ file {data_path} does not exist.")
        sys.exit(1)

    # Load data
    print(f"Loading data from {data_path}")
    data = np.load(data_path, allow_pickle=True)
    times   = data["times"]      # (n_nodes,)
    weights = data["weights"]    # (n_nodes,)
    z_vals  = data["z"]          # (n_nodes,)
    O_vals  = data["O"]          # (n_nodes,)
    E       = data["spectrum"]   # (dim,)

    print(f"  -> times.shape={times.shape}, weights.shape={weights.shape}")
    print(f"  -> z_vals.shape={z_vals.shape}, O_vals.shape={O_vals.shape}")
    print(f"  -> spectrum.shape={E.shape}")

    E_min, E_max = E[0], E[-1]
    print(f"Min/Max of spectrum: E_min={E_min:.3f}, E_max={E_max:.3f}")

    # =========== 4a. lambda sweep =============
    # We'll do from -1.1 * |E_min| to +1.1 * |E_min|
    lam_values = np.linspace(-1.1*abs(E_min), +1.1*abs(E_min), args.lam_points)
    expt_vals  = []
    for lam in lam_values:
        val = gauss_partition_calc(times, weights, z_vals, O_vals, lam)
        expt_vals.append(val)

    # =========== 4b. Plot results =============
    plt.figure(figsize=(6,4))
    # TOL colormap for the exact energies
    cmap = tol_cmap('rainbow_PuRd').resampled(len(E))
    color_list = [cmap(i/(len(E)-1)) for i in range(len(E))] if len(E) > 1 else ['r']

    # Our main black line for the ITQDE-based expectation
    plt.plot(lam_values / abs(E_min), expt_vals, 'k', label="ITQDE Gauss Quad")

    # Overplot each eigenvalue E_i as line: y = E_i + λ
    for i, e_val in enumerate(E):
        c = color_list[i]
        label_txt = f"E_{i}" if i in (0, len(E)-1) else None
        plt.plot(lam_values / abs(E_min), e_val + lam_values, '--', color=c, label=label_txt)

    plt.xlabel(r"$\lambda / |E_{\rm min}|$")
    plt.ylabel(r"$\langle H^{(\lambda)} \rangle(\tau \to \infty)$")
    plt.title("ITQDE Gauss Quadrature (1D Ising) - Lambda Sweep")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("1d_ising_gauss_sweep.png", dpi=300)
    plt.show()
    print("Plot saved as 1d_ising_gauss_sweep.png\nDone.")

if __name__ == "__main__":
    main()

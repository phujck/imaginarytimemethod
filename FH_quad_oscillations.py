import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
from scipy import linalg

# Plotting defaults
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=18)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# TOL colormap
class TOLcmaps:
    def __init__(self):
        self.cmap = None
        self.cname = None
        self.funcdict = {'rainbow_PuRd': self.__rainbow_PuRd}

    def __rainbow_PuRd(self):
        clrs = ['#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
                '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
                '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
                '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
                '#DF4828', '#DA2222']
        self.cmap = LinearSegmentedColormap.from_list('rainbow_PuRd', clrs)
        self.cmap.set_bad('#FFFFFF')

    def get(self, cname='rainbow_PuRd', lut=None):
        if cname not in self.funcdict:
            cname = 'rainbow_PuRd'
            print(f"*** Warning: requested colormap not defined. Using {cname}.")
        self.funcdict[cname]()
        return self.cmap

def tol_cmap(name='rainbow_PuRd'):
    return TOLcmaps().get(name)

def gauss_partition_calc(times, weights, z_vals, O_vals, lam, weight_number):
    num, den = 0.0 + 0.0j, 0.0 + 0.0j
    for i in range(min(weight_number, len(times))):
        phase = np.exp(2j * lam * times[i])
        num += weights[i] * 2 * np.real(phase * O_vals[i])
        den += weights[i] * 2 * np.real(phase * z_vals[i])
    return np.real(num / den) if abs(den) > 1e-14 else np.nan

def generate_plot(Lx=2, Ly=2, J=1.0, U=2.0, mu=0.5, tau=5, n_nodes=1000,
                  lam_points=300, weight_numbers=[25, 35, 40],
                  output_filename="gauss_quad_multiweight.pdf"):
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
    args.tau = tau
    args.n_nodes = n_nodes
    args.U = U
    args.Ly = Ly
    args.Lx = Lx

    # Reproduce the same naming convention from your generation script
    # e.g. "2DFH_gauss_Lx{Lx}_Ly{Ly}_NfUp{N_up}_NfDown{N_down}_J{J}_U{U}_mu{mu}_tau{tau}_n{n_nodes}.npz"
    Lx = args.Lx
    Ly = args.Ly
    N_2d = Lx * Ly
    N_up = N_2d // 2
    N_down = N_2d // 2
    N_2d = Lx * Ly

    filename = (f"2DFH_gauss_Lx{Lx}_Ly{Ly}_NfUp{N_up}_NfDown{N_down}"
                f"_J{args.J}_U{args.U}_mu{args.mu}"
                f"_tau{args.tau}_n{args.n_nodes}.npz")
    filepath = os.path.join(args.main_path, filename)


    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"NPZ file not found: {filepath}")

    print(f"Loading data from {filepath}")
    data = np.load(filepath, allow_pickle=True)
    times, weights = data['times'], data['weights']
    z_vals, O_vals, E = data['z'], data['O'], data['spectrum']
    E_min, E_max = E[0], E[-1]

    lam_values = np.linspace(-1.1 * abs(E_min), 1.1 * abs(E_min), lam_points)
    cmap = tol_cmap('rainbow_PuRd').resampled(len(E))
    e_colors = [cmap(i/(len(E)-1)) for i in range(len(E))]

    expt_vals_all = {}
    colors_m = ['blue', 'red', 'black']
    for w in weight_numbers:
        expt_vals_all[w] = [gauss_partition_calc(times, weights, z_vals, O_vals, lam, weight_number=w)
                            for lam in lam_values]

    plt.figure(figsize=(7, 5))
    for i, e_val in enumerate(E):
        plt.plot(lam_values / abs(E_min), (e_val * np.ones_like(lam_values)), '--', color=e_colors[i])

    for e, w in enumerate(weight_numbers):
        plt.plot(lam_values / abs(E_min), expt_vals_all[w], label="$\\bar{m}$="f"{w}",color=colors_m[e])

    gap_sizes = np.diff(E)
    i_max_gap = np.argmax(gap_sizes)

    import matplotlib.patches as patches
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    # Identify the largest spectral gap
    gap_sizes = np.diff(E)
    i_max_gap = np.argmax(gap_sizes)
    y_bottom = E[i_max_gap]
    y_top = E[i_max_gap + 1]
    x_brace = -0.5  # Adjust x position if needed

    def curly_brace(x, y0, y1, width=0.05, brace_height=0.1, lw=2, color='k'):
        """Draw a vertical curly brace from y0 to y1 at position x."""
        mid = (y0 + y1) / 2
        dy = (y1 - y0)

        verts = [
            (x, y1),
            (x + width, y1 - dy * 0.05),
            (x + width, mid + brace_height),
            (x+2*width, mid),
            (x + width, mid - brace_height),
            (x + width, y0 + dy * 0.05),
            (x, y0),
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE3, Path.CURVE3,
            Path.CURVE3, Path.CURVE3,
            Path.CURVE3, Path.LINETO,
        ]
        path = Path(verts, codes)
        return PathPatch(path, lw=lw, edgecolor=color, facecolor='none')

    brace = curly_brace(x_brace, y_bottom, y_top)
    plt.gca().add_patch(brace)

    # Add label
    plt.text(x_brace + 0.14, (y_bottom + y_top) / 2, r"$\Delta_{\rm max}$",
             fontsize=14, va='center', ha='left')


    import matplotlib.patches as patches


    plt.xlabel(r"$\lambda / |E_0|$")
    plt.ylabel(r"$\langle H^{(\lambda)} \rangle$")
    plt.ylim(E_min - 1.5, E_max + 1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    generate_plot()

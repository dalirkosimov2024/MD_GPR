import numpy as np
from plots import *
from scipy.signal import savgol_filter
from pathlib import Path

def dat_reader(filename):
    data = np.loadtxt(filename, usecols=(0, 1))
    r = data[:, 0]
    g_r = data[:, 1]
    label = "QMD (Hu et al. 2014)"
    return r, g_r, label

def rdf_reader(filename):

    current_x = []
    current_y = []
    current_ts = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # New timestep marker (2 values)
            if len(parts) == 2:
                current_ts = int(parts[0])
                current_x = []
                current_y = []

            # Data line (4 values)
            elif len(parts) == 4:
                current_x.append(float(parts[1]))  # 2nd value -> r
                current_y.append(float(parts[2]))  # 3rd value -> g_r

    # Convert to numpy arrays
    r = np.array(current_x, dtype=float)
    g_r = np.array(current_y, dtype=float)
    label = "MD (Dalir)"

    return r, g_r, label

# Calculates root mean squared difference 
def rdf_rmse(r_md, g_md, r_qmd, g_qmd):
    # overlapping range
    rmin = max(r_md.min(), r_qmd.min())
    rmax = min(r_md.max(), r_qmd.max())
    md_mask = (r_md >= rmin) & (r_md <= rmax)
    qmd_mask = (r_qmd >= rmin) & (r_qmd <= rmax)
    r = r_md[md_mask]
    g_md = g_md[md_mask]

    # interpolate QMD onto MD grid
    g_qmd_interp = np.interp(r, r_qmd[qmd_mask], g_qmd[qmd_mask])
    rmse = np.sqrt(np.mean((g_md - g_qmd_interp)**2))
    return rmse


def rdf_generator(b,Z,kappa):
    
    # location of g(r) data
    path = Path.cwd()

    g_cc = f"{path}/data/gr_C-C_5000K.dat"
    g_ch = f"{path}/data/gr_C-H_5000K.dat"
    g_hh = f"{path}/data/gr_H-H_5000K.dat"
    filename_md = f"{path.parent}/rdf.rdf"

    alpha = 0.4

    label_cc = "QMD C-C"
    label_ch = "QMD C-H"
    label_hh = "QMD H-H"

    r_cc, g_cc, label1 = dat_reader(g_cc)
    r_ch, g_ch, label1 = dat_reader(g_ch)
    r_hh, g_hh, label1 = dat_reader(g_hh)

    r_md, gr_md, label_md = rdf_reader(filename_md)

    # combines individual g(r) according to the weight (this assumes C = H)
    g_total = (1/4)*g_cc + (1/2)*g_ch + (1/4)*g_hh
    label_total = "Total QMD"
    r_md, gr_md = r_md[r_md <= 5], gr_md[r_md <= 5]
    gr_md = savgol_filter(gr_md, 15, 3)
    rmse = rdf_rmse(r_md, gr_md, r_cc, g_total)
    plot_saver(r_cc, g_total, r_md, gr_md,
                label_total, label_md, rmse, b, Z,kappa)

    return rmse
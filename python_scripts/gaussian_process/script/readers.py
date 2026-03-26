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

def rdf_reader(filename, n_last=1):

    rdf_blocks = []

    current_x = []
    current_y = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) == 2:
                if current_x and current_y:
                    rdf_blocks.append((
                        np.array(current_x, dtype=float),
                        np.array(current_y, dtype=float)
                    ))

                current_x = []
                current_y = []

            elif len(parts) == 4:
                current_x.append(float(parts[1]))
                current_y.append(float(parts[2]))

    if current_x and current_y:
        rdf_blocks.append((
            np.array(current_x, dtype=float),
            np.array(current_y, dtype=float)
        ))

    # Take last n blocks
    last_blocks = rdf_blocks[-n_last:]

    # Stack and average
    r = last_blocks[0][0]  # assume same r grid
    g_stack = np.array([block[1] for block in last_blocks])

    g_r = np.mean(g_stack, axis=0)

    label = f"MD (Dalir, avg last {n_last})"

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
    g_total = f"{path}/data/gr_total_15000K.dat"
    filename_md = f"{path.parent}/rdf.rdf"
    r_cc, g_total, label = dat_reader(g_total)

    r_md, gr_md, label_md = rdf_reader(filename_md)

    # combines individual g(r) according to the weight (this assumes C = H)

    label_total = "Total QMD"
    r_md, gr_md = r_md[r_md <= 5], gr_md[r_md <= 5]
    rmse = rdf_rmse(r_md, gr_md, r_cc, g_total)

    plot_saver(r_cc, g_total, r_md, gr_md,
                label_total, label_md, rmse, b, Z,kappa)

    return rmse

def value_reader(filepath="values.txt"):
    data = np.loadtxt(filepath)  # assumes no header

    #if data.shape[1] != 4:
    #    raise ValueError("Expected 4 columns: x y z rmse")

    X = data[:, :3]   # x, y, z
    y = data[:, 3]    # rmse

    return X, y


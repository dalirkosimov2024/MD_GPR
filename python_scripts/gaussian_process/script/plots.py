import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

def counter():
     os.chdir("pics/gpr_comparison")
     iter =  len(os.listdir()) + 1
     os.chdir("..")
     os.chdir("..")
     return iter

# Plots MD g(r) and QMD g(r)
def plotter(r, g_r, label,alpha=1):
    plt.plot(r, g_r, label = label, alpha=alpha)

    plt.legend()

# Saves the plot 
def plot_saver(r_cc, g_total, r_md, gr_md,
             label_total, label_md, fit, b, Z, kappa):
    iter = counter()
    #plotter(r_cc, g_cc, label_cc,alpha=alpha)
    #plotter(r_ch, g_ch, label_ch,alpha=alpha)
    #plotter(r_hh, g_hh, label_hh, alpha=alpha)
    plotter(r_cc, g_total, label_total)
    plotter(r_md, gr_md, label_md)
    plt.title(f"RMSE = {round(float(fit), 3)}, b = {round(float(b),3)}" r" A$^{-3}$" f", Z = {round(float(Z),3)}, k = {round(float(kappa),3)}" )
    plt.xlabel(r"r (Angstrom)")
    plt.ylabel(r"g(r)")
    plt.savefig(f"pics/gpr_comparison/iter_{int(iter)}.png")
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def plot_gp(gpr, x_train, y_train, X, Y, Z, best_y, next_point, label, iteration):
    """
    Single 3D plot with layered contour slices of GP mean.
    Keeps the existing call signature unchanged.
    """

    # Recover 1D axes from meshgrid
    x_vals = np.unique(X.ravel())
    y_vals = np.unique(Y.ravel())
    z_vals = np.unique(Z.ravel())

    # Choose a subset of z slices to avoid clutter
    n_slices =100
    slice_indices = np.linspace(0, len(z_vals) - 1, n_slices, dtype=int)
    z_slices = z_vals[slice_indices]

    # Use ij indexing so shapes are (len(x), len(y))
    XX, YY = np.meshgrid(x_vals, y_vals, indexing="ij")

    # First compute all slices so they share one color scale
    slice_data = []
    global_min = np.inf
    global_max = -np.inf

    for z_fixed in z_slices:
        xyz_local = np.column_stack([
            XX.ravel(),
            YY.ravel(),
            np.full(XX.size, z_fixed)
        ])

        mean_prediction, std_prediction = gpr.predict(xyz_local, return_std=True)
        mean_2d = mean_prediction.reshape(XX.shape)

        slice_data.append((z_fixed, mean_2d))
        global_min = min(global_min, np.min(mean_2d))
        global_max = max(global_max, np.max(mean_2d))

    norm = Normalize(vmin=global_min, vmax=global_max)
    cmap = cm.magma

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw filled contours as horizontal layers in 3D
    for z_fixed, mean_2d in slice_data:
        ax.contourf(
            XX, YY, mean_2d,
            zdir='z',
            offset=z_fixed,
            levels=12,
            cmap=cmap,
            norm=norm,
            alpha=0.05
        )

        ax.contour(
            XX, YY, mean_2d,
            zdir='z',
            offset=z_fixed,
            levels=12,
            colors='k',
            linewidths=0,
            alpha=0.05
        )

    # Training points
    sc = ax.scatter(
        x_train[:, 0],
        x_train[:, 1],
        x_train[:, 2],
        c=y_train,
        cmap="plasma",
        s=70,
        edgecolors="black",
        linewidths=0.7,
        depthshade=False
    )


    # Next point
    ax.scatter(
        next_point[0],
        next_point[1],
        next_point[2],
        color="red",
        marker="*",
        s=120,
        edgecolors="black",
        linewidths=0.8,
        depthshade=False,
    )

    # Axis limits
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(0, 6)
    ax.set_zlim(0,12)

    # Labels
    ax.set_xlabel("Short range repulsion wavevector (b)")
    ax.set_ylabel("Ionisation (Z)")
    ax.set_zlabel(r"Screening wavevector ($\kappa $)")

    print("x range:", np.ptp(x_vals), "unique:", len(x_vals))
    print("y range:", np.ptp(y_vals), "unique:", len(y_vals))
    print("z range:", np.ptp(z_vals), "unique:", len(z_vals))


    # Domain proportions
    ax.set_box_aspect((12,6,12))

    # View angle
    ax.view_init(elev=28, azim=-55)

    # Colorbar for contour values
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.4)
    cbar.set_label("g(r) fit RMSE")

    training_line = mlines.Line2D([], [], color="orange", marker="o", linestyle="none", label="Training points")
    next_point_line = mlines.Line2D([], [], color="red", linestyle="none",marker="*", label="Next point")


    ax.legend(
        handles=[training_line, next_point_line],
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )

    iter = counter()

    print(os.getcwd())
    

    plt.tight_layout()
    plt.savefig(f"pics/gpr_clouds/iter_{iter}.png")
    plt.close()
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

def counter():
     os.chdir("pics")
     iter =  len(os.listdir()) + 1
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
    plt.savefig(f"pics/iter_{int(iter)}.png")
    plt.close()

# Plots gp contour
def plot_gp(gpr, x_train, y_train, X, Y, best_y, next_point, label, iteration):
        xy_local = np.vstack([X.ravel(), Y.ravel()]).T
        mean_prediction, std_prediction = gpr.predict(xy_local, return_std=True)
        mean_prediction = mean_prediction.reshape(X.shape)
        std_prediction = std_prediction.reshape(X.shape)

        fig, ax = plt.subplots(figsize=(7, 5))

        # GP mean contours
        cs = ax.contourf(X, Y, mean_prediction, levels=10, cmap="viridis", alpha=0.6)
        ax.contour(X, Y, mean_prediction, levels=10, colors="black", linewidths=0.5)
        fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.1,label="RMSE", shrink=0.3)

        # Training points
        ax.scatter(x_train[:, 0], x_train[:, 1], color="red", marker="x", s=30)

        # Best point so far
        best_idx = np.argmin(y_train)
        ax.scatter(
            x_train[best_idx, 0], x_train[best_idx, 1],
            color="magenta", marker="*", s=120
        )

        # Next point
        ax.scatter(next_point[0], next_point[1], color="green", marker="o", s=80)

        sample_line = mlines.Line2D([], [], color="red", marker="x", linestyle="none", label="Training points")
        mean_line = mlines.Line2D([], [], color="red", linestyle="--", label="Posterior mean")
        next_line = mlines.Line2D([], [], color="green", marker="o", linestyle="none", label="Next point")
        best_line = mlines.Line2D([], [], color="magenta", marker="*", linestyle="none", label="Best RMSE so far")

        ax.legend(
            handles=[sample_line, next_line],
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )

    
        ax.set_xlabel(r"Short-range-repulsion onset, b (A$^{-1}$)")
        ax.set_ylabel("Ionisation, Z")
        ax.set_xlim(2, 12)
        ax.set_ylim(0, 3)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig("pics/parameter_space.png")
        plt.close()

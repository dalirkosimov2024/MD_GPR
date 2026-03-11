import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import matplotlib.lines as mlines
import subprocess
from scipy.interpolate import interp1d
from scipy.stats import qmc
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

def writeup(filename, b, Z, rmse):
        with open(filename, "a") as f:
            f.write(f"{b} {Z} {rmse}\n")

def gp():

    def target_function(X):
        b, Z = X
        potential_writer(b, Z)
        md_runner()
        rmse = rdf_reader(b, Z)
        return rmse   # minimize this

    # =========================================================
    # Candidate grid
    # =========================================================
    x = np.linspace(2, 12, 100)
    y = np.linspace(0, 3, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    # =========================================================
    # Load initial training data
    # =========================================================
    data = np.loadtxt("b_Z_list.txt")
    x_train = data[:, :2]
    y_train = data[:, 2]

    print(f"x_train:\n{x_train}\ny_train:\n{y_train}")

    # =========================================================
    # GP model
    # =========================================================
    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=[0.5, 0.5],
        length_scale_bounds=(1e-2, 1e1)
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=1
    )

    # Fit before BO starts
    gpr.fit(x_train, y_train)

    # =========================================================
    # Plotting function
    # =========================================================
    def plot_gp(gpr, x_train, y_train, X, Y, best_y, next_point, label, iteration):
            xy_local = np.vstack([X.ravel(), Y.ravel()]).T
            mean_prediction, std_prediction = gpr.predict(xy_local, return_std=True)
            mean_prediction = mean_prediction.reshape(X.shape)
            std_prediction = std_prediction.reshape(X.shape)

            fig, ax = plt.subplots(figsize=(7, 5))

            # GP mean contours
            cs = ax.contourf(X, Y, mean_prediction, levels=10, cmap="viridis")
            ax.contour(X, Y, mean_prediction, levels=10, colors="black", linewidths=0.5)
            fig.colorbar(cs, ax=ax, orientation="vertical", pad=0.1, anchor=(0, 0.2),label="RMSE", shrink=0.3)

            # Training points
            ax.scatter(x_train[:, 0], x_train[:, 1], color="red", marker="x", s=50)

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
                handles=[sample_line, next_line, best_line],
                bbox_to_anchor=(1, 1),
                loc="upper left",
            )

      
            ax.set_xlabel(r"Short-range-repulsion onset, b (A$^{-1}$)")
            ax.set_ylabel("Ionisation, Z")
            ax.set_xlim(2, 12)
            ax.set_ylim(0, 3)
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.show()


    # =========================================================
    # Remove already-sampled points
    # =========================================================
    def mask_sampled_points(candidates, sampled_points, tol=1e-8):
        keep = np.ones(len(candidates), dtype=bool)
        for i, c in enumerate(candidates):
            if np.any(np.linalg.norm(sampled_points - c, axis=1) < tol):
                keep[i] = False
        return keep

    # =========================================================
    # Minimization acquisitions
    # =========================================================
    def expected_improvement_min(x_candidates, gpr, best_y, xi=0.01):
        mu, sigma = gpr.predict(x_candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-12)

        improvement = best_y - mu - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def lower_confidence_bound(x_candidates, gpr, kappa=1.5):
        mu, sigma = gpr.predict(x_candidates, return_std=True)
        return mu - kappa * sigma   # minimize this


    # =========================================================
    # Bayesian optimization
    # =========================================================
    def bayesian_optimization(gpr, x_train, y_train, num_iter=10, acquisition='lcb'):
        for i in range(num_iter):
            keep_mask = mask_sampled_points(xy, x_train)
            candidate_points = xy[keep_mask]

            best_y = np.min(y_train)

            if acquisition == 'ei':
                scores = expected_improvement_min(candidate_points, gpr, best_y)
                next_point = candidate_points[np.argmax(scores)]
                label = "expected improvement"
            elif acquisition == 'lcb':
                scores = lower_confidence_bound(candidate_points, gpr, kappa=1.5)
                next_point = candidate_points[np.argmin(scores)]
                label = "lower confidence bound"
            elif acquisition == 'mean':
                mu, _ = gpr.predict(candidate_points, return_std=True)
                next_point = candidate_points[np.argmin(mu)]
                label = "posterior mean"
            else:
                raise ValueError("acquisition must be 'ei', 'lcb', or 'mean'")

            print(f"next point: b={next_point[0]}, Z={next_point[1]}")

            plot_gp(
                gpr=gpr,
                x_train=x_train,
                y_train=y_train,
                X=X,
                Y=Y,
                best_y=best_y,
                next_point=next_point,
                label=label,
                iteration=i + 1
            )

            # Evaluate new point
            new_y = target_function(next_point)

            # Add to training set
            x_train = np.vstack([x_train, next_point])
            y_train = np.append(y_train, new_y)

            # Refit GP
            gpr.fit(x_train, y_train)

            # Plot results


            print(f"Iteration {i+1}: next_point = {next_point}, RMSE = {new_y:.6f}")
            print(f"Best RMSE so far = {np.min(y_train):.6f}")

            writeup("b_Z_list.txt", next_point[0], next_point[1], float(new_y))

        return x_train, y_train

    x_train, y_train = bayesian_optimization(
        gpr, x_train, y_train, num_iter=10, acquisition='lcb'
    )

    return x_train, y_train
       
    
def potential_writer(b, Z):
    import numpy as np
    from numpy import exp, pi, sqrt
    import matplotlib.pyplot as plt
    import datetime

    def calculator(b,Z):

        #b = 1.323 #A-1
        Z_C = 6
        A = 14.400778
        T_e = 5000 #K
        e = 1 
        n_i = 0.12755 # A ^-3
        n_e = Z*n_i
        r_bohr = 0.529177210903 # A
        r_s = (3/(4 * pi * n_e))**(1/3) / r_bohr
        eps0 =  e**2 / (4*pi*A) # from simple rearraning
        k_B = 8.617333262e-5 

        E_F = 3.80998* (3*pi**2* n_e ) ** (2/3) # constant comes from conversion of hbar^2/2me into metal units
        print(E_F)
        T_F = E_F / k_B

        T_q = T_F / (1.3251 - 0.1779 * sqrt(r_s))
        #T_q = 2.3 * T_F

        T_eff = (T_e **2 + T_q **2)**0.5

        k_PW = sqrt( (e**2 * n_e) / (eps0 * k_B * T_eff))
        k_TF =  sqrt( (3*e**2 * n_e) / (2* eps0 *E_F))
            
        print(k_TF)
        print(k_PW)

        def PW_func(r):
            return (A*Z**2)/ r * exp(-k_PW * r)
        
        def PW_SRR_func(r):
            return (A*Z**2)/ r * exp(-k_PW * r) + (Z_C**2 - Z**2)*A/r * exp(-b*r)
            
        def TF_SRR_func(r):
            return (A*Z**2)/ r * exp(-k_TF * r) + (Z_C**2 - Z**2)*A/r * exp(-b*r)
        
        def TF_func(r):
            return (A*Z**2)/ r * exp(-k_TF * r)
        
        r_array = np.arange(0.001, 10, 0.001)

        TF_array = np.array([])
        TF_SRR_array = np.array([])
        PW_array = np.array([])
        PW_SRR_array = np.array([])

        for r in r_array:
            TF = TF_func(r)
            TF_SRR = TF_SRR_func(r)
            PW  = PW_func(r)
            PW_SRR = PW_SRR_func(r)
            
            TF_array = np.append(TF_array, TF)
            TF_SRR_array = np.append(TF_SRR_array, TF_SRR)
            PW_array = np.append(PW_array, PW)
            PW_SRR_array = np.append(PW_SRR_array, PW_SRR)
        
        #plt.plot(r_array, TF_array, label="Thomas-Fermi", color="orange") 
        #plt.plot(r_array, TF_SRR_array, linestyle = "--", color="orange", label="Thomas-Fermi + SRR") 
        #plt.plot(r_array, PW_array, color = "green", label = "Perrot-Dharma-Wardana")

        plt.plot(r_array, PW_SRR_array,label=f"Z= {Z},b={b}")
        plt.suptitle(r"Pair potential, ionisation (Z) and short-range-repulsion wavevector (b) parameter scan (C-C, 5000 K, 0.914 Mbar, 2.429 $\rho$/$\rho_0$)")
        
        plt.ylim(0, 10)
        plt.xlabel("r (angstrom)")
        plt.ylabel("Energy (eV)")
        plt.legend()
        
        dr = r_array[1] - r_array[0]
        F = -np.gradient(PW_SRR_array, dr)

        return r_array, PW_SRR_array,  F

    def writeup(filename, r, V, F):
        
        with open(filename, "w") as f:

            f.write(f"# C-C, Z = {Z}, b={b}, {datetime.datetime.now()} \n")
            f.write("MY_POTENTIAL\n")        # <-- table name used by pair_coeff
            f.write(f"N {len(r)} R 0.001 7.999 \n\n")

            for i in range(len(r)):
                f.write("%d %f %f %f\n" % (i+1, r[i], V[i], F[i]))

        print("LAMMPS table written to:", filename)

    r, V, F = calculator(
                    b=b,
                    Z=Z,
                    )
    plt.show()
    writeup("veff.table", r, V, F)

def md_runner():
    subprocess.run([
        "lmp",
    "-in", "test.in",
    "-var", "rdf_file", "rdf.rdf"
    ], check=True)

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

def plotter(r, g_r, label,alpha=1):

    plt.plot(r, g_r, label = label, alpha=alpha)


    plt.xlabel(r"r (Angstrom)")
    plt.ylabel(r"g(r)$")
    plt.legend()

def plot_saver(r_cc, g_total, r_md, gr_md, label_total, label_md, rmsd, b, Z):
    title = f"RMSE = {round(float(rmsd), 3)}, b = {round(float(b),3)}" r" A$^{-3}$" f", Z = {round(float(Z),3)}"
    #plotter(r_cc, g_cc, label_cc,alpha=alpha)
    #plotter(r_ch, g_ch, label_ch,alpha=alpha)
    #plotter(r_hh, g_hh, label_hh, alpha=alpha)
    plotter(r_cc, g_total, label_total)
    plotter(r_md, gr_md, label_md)
    plt.suptitle(title)

    plt.savefig(f"pics/b_{round(float(b), 3)}_Z_{round(float(Z),3)}.png")
  

    plt.show()
    plt.close()

def rdf_reader(b,Z):

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





 

    g_cc = "/home/lcv510/Documents/cdt/york/lammps/LAMMPS/quantum_plasmas/pair_distribution_function/gr_C-C_5000K.dat"
    g_ch = "/home/lcv510/Documents/cdt/york/lammps/LAMMPS/quantum_plasmas/pair_distribution_function/gr_C-H_5000K.dat"
    g_hh = "/home/lcv510/Documents/cdt/york/lammps/LAMMPS/quantum_plasmas/pair_distribution_function/gr_H-H_5000K.dat"
    filename_md = "/home/lcv510/Documents/cdt/york/lammps/LAMMPS/python_scripts/gaussian_process/rdf.rdf"

    alpha = 0.3

    label_cc = "QMD C-C"
    label_ch = "QMD C-H"
    label_hh = "QMD H"

    r_cc, g_cc, label1 = dat_reader(g_cc)
    r_ch, g_ch, label1 = dat_reader(g_ch)
    r_hh, g_hh, label1 = dat_reader(g_hh)

    r_md, gr_md, label_md = rdf_reader(filename_md)

    g_total = (1/3)*g_cc + (1/3)*g_ch + (1/3)*g_hh
    label_total = "Total QMD"

    r_md, gr_md = r_md[r_md <= 5], gr_md[r_md <= 5]
    gr_md = savgol_filter(gr_md, 15, 3)

    rmsd = rdf_rmse(r_md, gr_md, r_cc, g_total)

    plot_saver(r_cc, g_total, r_md, gr_md, label_total, label_md, rmsd, b, Z)
    writeup("b_Z_list.txt", b, Z, rmsd)

    return rmsd


        
if __name__ == "__main__":
    gp()



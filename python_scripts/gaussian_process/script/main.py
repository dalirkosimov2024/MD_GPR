# imports 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import subprocess
from scipy.stats import qmc
import os

from potential_writer import *
from plots import *
from readers import *

# target function, outputs z axis fit 
def target_function(X):
    b, Z, kappa = X
    potential_writer(b, Z, kappa)
    md_runner()
    rmse = rdf_generator(b, Z, kappa)
    return rmse   

# write b, Z and rmse text file
def writeup(filename, b, Z, kappa, rmse):
        with open(filename, "a") as f:
            f.write(f"{b} {Z} {kappa} {rmse}\n")


def mask_sampled_points(candidates, sampled_points, tol=1e-8):
    keep = np.ones(len(candidates), dtype=bool)
    for i, c in enumerate(candidates):
        if np.any(np.linalg.norm(sampled_points - c, axis=1) < tol):
            keep[i] = False
    return keep


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

def bayesian_optimization(gpr,xyz,x_train, y_train, X, Y, Z,num_iter=10, acquisition='lcb'):
    for i in range(num_iter):
        keep_mask = mask_sampled_points(xyz, x_train)
        candidate_points = xyz[keep_mask]

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
    
        print("\n....................\n")
        print(f"Next point: b={next_point[0]}, Z={next_point[1]}, kappa = {next_point[2]}")
        print("\n....................\n")

        plot_gp(
            gpr=gpr,
            x_train=x_train,
            y_train=y_train,
            X=X,
            Y=Y,
            Z=Z,
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


        

        writeup("values.txt", next_point[0], next_point[1], next_point[2], float(new_y))

    return x_train, y_train



# runs LAMMPS remotely
def md_runner():
    og_dir  = os.getcwd()
    os.chdir("..")
    subprocess.run([
        "lmp",
    "-in", "test.in"
    ], check=True)
    os.chdir(og_dir)

from scipy.signal import find_peaks
import numpy as np

def gp():
    # Search space
    x_min, x_max = 0.0, 12.0
    y_min, y_max = 0.0, 6.0
    z_min, z_max = 0.0, 12.0

    x = np.linspace(x_min, x_max, 10)
    y = np.linspace(y_min, y_max, 10)
    z = np.linspace(z_min, z_max, 100)
    X, Y,Z = np.meshgrid(x, y, z)
    xyz = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    lhc = True
    # If fewer than 4 points exist, generate the remaining ones with LHS
    if lhc:
        n_needed = 3

        sampler = qmc.LatinHypercube(d=3, seed=1)
        lhs_unit = sampler.random(n=n_needed)

        lhs_points = qmc.scale(
            lhs_unit,
            l_bounds=[x_min, y_min, z_min],
            u_bounds=[x_max, y_max, z_max]
        )


        print("\n\nInitial LHS points to evaluate:")
        print(lhs_points)
        print("\n\n")

        for value in lhs_points:
            b = value[0]
            Z = value[1]
            kappa = value[2]
            
            rmse = target_function(value)
            writeup("values.txt", b, Z, kappa, rmse)

   # read values.txt and output them
   # x_train = b, Z , kappa
   # y_train = rmse
    x_train, y_train = value_reader()

    print(f"\n\n{x_train},{y_train}\n\n")
        

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=[0.5, 0.5, 0.5],
        length_scale_bounds=(1e-2, 1e1)
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=1
    )

    # Fit GP after initial 4 points are available
    gpr.fit(x_train, y_train)

    # Begin Bayesian optimization only after initial LHS phase is complete
    x_train, y_train = bayesian_optimization(
        gpr, xyz, x_train, y_train, X, Y,Z, num_iter=10, acquisition='lcb'
    )

    return x_train, y_train

def main():
    gp()
     
if __name__ == "__main__":
    main()




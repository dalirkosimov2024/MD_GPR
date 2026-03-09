import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.lines as mlines



# Global variables
n_sample_list = np.arange(2,5)
n_sample_list_squared = n_sample_list**2
test_size = 0.05

# 3D Gaussian function, evaluated only on the plane z = 0
def custom_function(x, y, z=0.0, A=1.0,
                    x0=2.0, y0=3.5, z0=0.0,
                    sigma_x=0.35, sigma_y=0.9, sigma_z=1.0):

    return A * np.exp(
        -(
            ((x - x0) ** 2) / (2 * sigma_x ** 2)
            + ((y - y0) ** 2) / (2 * sigma_y ** 2)
            + ((z - z0) ** 2) / (2 * sigma_z ** 2)
        )
    )

# Generate sample points using Latin Hypercube Sampling in x-y only
# All points lie on z = 0 plane
def generate_latin_hypercube_samples(n_samples, dimensions=2):
    sampler = qmc.LatinHypercube(d=dimensions)
    sample = sampler.random(n=n_samples)
    sample = qmc.scale(sample, l_bounds=[1, 1], u_bounds=[3, 6])
    return sample

# Function that iterates a GPR process over a LHC sampling distribution
def lhc_gpr():
    rmse_list = np.array([])

    for n_samples in n_sample_list_squared:
        # Generate LHS samples in 2D
        samples = generate_latin_hypercube_samples(n_samples, dimensions=2)

        # Evaluate the Gaussian on z = 0 plane
        z_values = np.array([custom_function(x, y, z=0.0) for x, y in samples])

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            samples, z_values, test_size=test_size
        )

        # Define kernel and GPR model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Train the model
        gp.fit(x_train, y_train)

        # Generate a 2D grid over x-y space
        x = np.linspace(1, 3, 100)
        y = np.linspace(1, 6, 100)
        x_grid, y_grid = np.meshgrid(x, y)
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        # True Gaussian values on z = 0 plane
        z_true = np.array(
            [custom_function(xi, yi, z=0.0) for xi, yi in grid_points]
        ).reshape(x_grid.shape)

        # GP prediction
        zfit, z_pred_std = gp.predict(grid_points, return_std=True)
        zfit = zfit.reshape(x_grid.shape)

        # RMSE
        mse = mean_squared_error(z_true, zfit)
        rmse = np.sqrt(mse)
        rmse_list = np.append(rmse_list, rmse)

        # Plotting in 2D with contours
        fig, ax = plt.subplots(figsize=(7, 6))

        # Filled contours of true function
        contour_true = ax.contourf(x_grid, y_grid, z_true, levels=30, cmap="viridis", alpha=0.75)

        # Contours of GP prediction
        ax.contour(x_grid, y_grid, zfit, levels=10, colors="red", linestyles="--", linewidths=1.2)

        # Training and testing points
        ax.scatter(x_train[:, 0], x_train[:, 1], marker="x", color="blue", s=50)
        ax.scatter(x_test[:, 0], x_test[:, 1], color="orange", s=35)

        # Legend
        truth_line = mlines.Line2D([], [], color="black", label="True Gaussian (filled contours)")
        mean_line = mlines.Line2D([], [], color="red", linestyle="--", label="GPR prediction contours")
        sample_line = mlines.Line2D([], [], color="blue", marker="x", linestyle="none", label="Training points")
        testing_points = mlines.Line2D([], [], color="orange", marker="o", linestyle="none", label="Testing points")

        ax.legend(
            handles=[truth_line, mean_line, sample_line, testing_points],
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        cbar = plt.colorbar(contour_true, ax=ax, shrink=0.3)
        cbar.set_label("Function value")

        ax.set_title(f"2D contour map of 3D Gaussian at z = 0 with {n_samples} LHS points")
        ax.set_xlabel("Onset of short-range-repulsion")
        ax.set_ylabel("Ionisation")
        ax.set_xlim(1, 3)
        ax.set_ylim(1, 6)
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

    return rmse_list


lhc_gpr()
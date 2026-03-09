import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import matplotlib.lines as mlines


def gp():


    # =========================================================
    # 2D Gaussian parameters
    # =========================================================
    A = 1.0
    x0 = 2.0
    y0 = 3.5
    sigma_x = 0.25
    sigma_y = 0.75

    # =========================================================
    # Define the 2D Gaussian peak function
    # Domain:
    #   x in [1, 3]
    #   y in [1, 6]
    # =========================================================
    def target_function(X):
        x, y = X
        return A * np.exp(
            -(
                ((x - x0) ** 2) / (2 * sigma_x ** 2)
                + ((y - y0) ** 2) / (2 * sigma_y ** 2)
            )
        )

    # =========================================================
    # Generate a grid of points over the new domain
    # =========================================================
    x = np.linspace(1, 3, 100)
    y = np.linspace(1, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = target_function([X, Y])

    # Flatten the grid for predictions / candidate search
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    z = Z.ravel()

    # =========================================================
    # Select the kernel and fit the Gaussian Process model
    # =========================================================
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[0.3, 0.8], length_scale_bounds=(1e-2, 1e2))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=1
    )

    # =========================================================
    # Initial training points sampled from the new domain
    # =========================================================
    np.random.seed(1)
    initial_indices = np.random.choice(len(xy), 10, replace=False)
    x_train = xy[initial_indices]
    y_train = z[initial_indices]

    gpr.fit(x_train, y_train)

    # =========================================================
    # Plotting function
    # =========================================================
    def plot_gp(gpr, x_train, y_train, X, Y, Z, best_y, next_point, label, iteration):
        xy_local = np.vstack([X.ravel(), Y.ravel()]).T
        mean_prediction, std_prediction = gpr.predict(xy_local, return_std=True)
        mean_prediction = mean_prediction.reshape(X.shape)


        fig, ax = plt.subplots(figsize=(7, 5))



        # True function as filled contours
        contour_true = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.75)

        # GP mean contours
        ax.contour(X, Y, mean_prediction, levels=10, colors="red", linestyles="--")

        # Training points
        ax.scatter(x_train[:, 0], x_train[:, 1], color="blue", marker="x", s=50)

        # Next point
        ax.scatter(next_point[0], next_point[1], color="green", marker="o", s=80)

        # True peak
        ax.scatter(x0, y0, color="cyan", marker="*", s=180)

        truth_line = mlines.Line2D([], [], color="black", label="True Gaussian")
        sample_line = mlines.Line2D([], [], color="blue", marker="x", linestyle="none", label="Training points")
        mean_line = mlines.Line2D([], [], color="red", linestyle="--", label="Posterior mean")
        next_line = mlines.Line2D([], [], color="green", marker="o", linestyle="none", label="Next point")
        peak_line = mlines.Line2D([], [], color="cyan", marker="*", linestyle="none", label="True peak")

        ax.legend(
            handles=[truth_line, sample_line, mean_line, next_line, peak_line],
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )

        cbar = plt.colorbar(contour_true, ax=ax, shrink=0.5)
        cbar.set_label("Function value")

        plt.suptitle(f"2D Gaussian fit with {label}, iteration {iteration}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(1, 3)
        ax.set_ylim(1, 6)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

    # =========================================================
    # Expected improvement for maximisation
    # This searches for the peak by preferring points that could
    # improve on the current best value.
    # =========================================================
    def expected_improvement(x_candidates, gpr, best_y, xi=0.01):
        mean_prediction, std_prediction = gpr.predict(x_candidates.reshape(-1, 2), return_std=True)

        std_prediction = np.maximum(std_prediction, 1e-12)
        improvement = mean_prediction - best_y - xi
        z = improvement / std_prediction
        ei = improvement * norm.cdf(z) + std_prediction * norm.pdf(z)
        return ei

    # =========================================================
    # Greedy peak-seeking acquisition:
    # pick the point with the highest GP posterior mean
    # This is the most direct "search for the peak" rule.
    # =========================================================
    def peak_acquisition(x_candidates, gpr):
        mean_prediction, _ = gpr.predict(x_candidates.reshape(-1, 2), return_std=True)
        return mean_prediction

    # =========================================================
    # Remove already-sampled points from candidate list
    # =========================================================
    def mask_sampled_points(candidates, sampled_points, tol=1e-8):
        keep = np.ones(len(candidates), dtype=bool)
        for i, c in enumerate(candidates):
            if np.any(np.linalg.norm(sampled_points - c, axis=1) < tol):
                keep[i] = False
        return keep

    # =========================================================
    # Perform Bayesian optimization
    # acquisition='peak' directly searches for the peak
    # acquisition='ei' searches for improvement toward the peak
    # =========================================================
    def bayesian_optimization(gpr, x_train, y_train, num_iter=10, acquisition='peak'):
        for i in range(num_iter):
            keep_mask = mask_sampled_points(xy, x_train)
            candidate_points = xy[keep_mask]

            best_y = np.max(y_train)

            if acquisition == 'ei':
                
                scores = expected_improvement(candidate_points, gpr, best_y)
                next_point = candidate_points[np.argmax(scores)]
                label = "expected improvement"
            elif acquisition == 'peak':
                scores = peak_acquisition(candidate_points, gpr)
                next_point = candidate_points[np.argmax(scores)]
                label = "peak search"
            else:
                raise ValueError("acquisition must be 'ei' or 'peak'")

            # Evaluate the new point
            new_y = target_function(next_point)

            # Add new point to training data
            x_train = np.vstack([x_train, next_point])
            y_train = np.append(y_train, new_y)

            # Retrain the GP
            gpr.fit(x_train, y_train)

            # Plot results
            plot_gp(gpr, x_train, y_train, X, Y, Z, best_y, next_point=next_point, label=label, iteration=i + 1)

            print(f"Iteration {i+1}: next_point = {next_point}, value = {new_y:.6f}")

        return x_train, y_train

    # Run Bayesian optimization with direct peak search
    x_train_final, y_train_final = bayesian_optimization(
        gpr, x_train, y_train, num_iter=10, acquisition='peak')
    


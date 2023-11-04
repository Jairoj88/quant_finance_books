import numpy as np


def brownian_paths(T, N_steps, n_paths):
    """
    Generate Brownian paths (uncorrelated) using random normal increments.

    Parameters:
    T (float): Total time horizon.
    N_steps (int): Number of time steps.
    n_paths (int): Number of paths to generate.

    Returns:
    numpy.ndarray: Array of Brownian motion paths with shape (n_paths, N_steps).
    """
    delta_t = T / N_steps

    # Generate the increments
    delta_B = np.random.normal(0, np.sqrt(delta_t), (n_paths, N_steps))

    # Calculate the Brownian Motion
    b_t = np.cumsum(delta_B, axis=1)

    return b_t


def brownian_reflected_paths(T, N_steps, n_half_paths):
    """
    Generate reflected Brownian paths (uncorrelated) using random normal increments.

    Parameters:
    T (float): Total time horizon.
    N_steps (int): Number of time steps.
    n_half_paths (int): Number of half-paths to generate (full paths will be reflected).

    Returns:
    numpy.ndarray: Array of reflected Brownian motion paths with shape (n_half_paths * 2, N_steps).
    """
    delta_t = T / N_steps

    # Generate the increments
    delta_B_half = np.random.normal(0, np.sqrt(delta_t), (n_half_paths, N_steps))
    delta_B = np.concatenate((delta_B_half, -delta_B_half))

    # Calculate the Brownian Motion
    b_t = np.cumsum(delta_B, axis=1)

    return b_t


def correlated_brownian_paths(T, N_steps, n_paths, correlation_matrix):
    """
    Generate correlated Brownian paths using a specified correlation matrix.

    Parameters:
    T (float): Total time horizon.
    N_steps (int): Number of time steps.
    n_paths (int): Number of paths to generate.
    correlation_matrix (numpy.ndarray): The correlation matrix for the paths.

    Returns:
    numpy.ndarray: Array of correlated Brownian motion paths with shape (n_paths, N_steps).
    """
    delta_t = T / N_steps

    # Generate the means and covariances for the multivariate normal distribution
    means = np.zeros(n_paths)
    cov = correlation_matrix * delta_t

    # Generate correlated Brownian increments
    correlated_delta_B = np.random.multivariate_normal(means, cov, (N_steps,))

    # Calculate the correlated Brownian Motion
    b_t = np.cumsum(correlated_delta_B, axis=0)

    return b_t

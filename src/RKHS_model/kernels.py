import numpy as np

def rbf_kernel(x, x_prime, gamma=0.01): #gamma=1/(2*sigma **2)
    """
    Compute the RBF (Radial Basis Function) kernel between two data points.

    Parameters:
    x (np.ndarray): First data point (1D array).
    x_prime (np.ndarray): Second data point (1D array).
    sigma (float): Kernel width parameter (default is 1.0).

    Returns:
    float: The RBF kernel value between x and x_prime.
    """
    # Ensure x and x_prime are numpy arrays
    x = np.asarray(x)
    x_prime = np.asarray(x_prime)

    # Compute the squared Euclidean distance between x and x_prime
    squared_distance = np.sum((x - x_prime) ** 2)

    # Compute the RBF kernel
    kernel_value = np.exp(-squared_distance*gamma)

    return kernel_value

#Euclidean kernel
# Euclidean kernel function
def kernel_euc(a, b):
    """
    Computes the Euclidean kernel (dot product) between two vectors.

    Args:
    a (numpy.ndarray): The first input vector.
    b (numpy.ndarray): The second input vector.

    Returns:
    float: The dot product of vectors a and b.

    Notes:
    - This kernel computes the inner product of the two input vectors.
    - In the context of kernel methods, this is a linear kernel, which is a special case of the more general dot product kernel.
    """
    return np.dot(a, b)
def kernel_circ(a, b):
    """
    Computes a custom kernel function between two vectors.

    The kernel function is defined as:
    K(a, b) = dot(a, b) + ||a||^2 * ||b||^2

    where:
    - dot(a, b) is the dot product between vectors a and b.
    - ||a||^2 is the squared Euclidean norm (or squared L2 norm) of vector a.
    - ||b||^2 is the squared Euclidean norm (or squared L2 norm) of vector b.

    Parameters:
    a (numpy.ndarray): A vector (1D numpy array).
    b (numpy.ndarray): Another vector (1D numpy array).

    Returns:
    float: The computed kernel value between vectors a and b.
    """
    # Compute the dot product between vectors a and b
    dot_product = np.dot(a, b)
    
    # Compute the squared Euclidean norm of vector a
    norm_a_squared = np.linalg.norm(a)**2
    
    # Compute the squared Euclidean norm of vector b
    norm_b_squared = np.linalg.norm(b)**2
    
    # Calculate the kernel value
    kernel_value = dot_product + norm_a_squared * norm_b_squared
    
    return kernel_value



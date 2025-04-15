from firedrake import *
from firedrake.petsc import PETSc


# The time partition must be of size (n+1) where n = p*2^k. p is the number of temporal processes. k is an integer.
def create_time_partition(n, p) -> list:
    """
    Create a time partition of length equal to p. \n
    n + 1 is the number of time steps and should satisfy \n
    the condition n = p*2^k.
    
    :param n: Number of time steps.
    :param p: Number of temporal processors.

    :return: A list of integers representing the time partition.

    :example:
    >>> create_time_partition(9, 2)
    [5, 4]
    >>> create_time_partition(9, 4)
    [3, 2, 2, 2]
    >>> create_time_partition(65, 8)
    [9, 8, 8, 8, 8, 8, 8, 8]
    """

    nbyp = n // p

    time_partition = [nbyp for _ in range(p)]
    time_partition[0] += 1
    
    return time_partition

def check_setup(problem_parameters):
    """
    Check the setup of the problem parameters.
    """
    temporal_processors = problem_parameters["Number of temporal processors"]
    number_of_time_steps = problem_parameters["Number of time steps"]
    processors = COMM_WORLD.size

    if processors%temporal_processors!=0:
        raise ValueError(f"Total number of processors {processors} must be divisible by number of temporal processors {temporal_processors}.")
    
    if (number_of_time_steps-1)%temporal_processors != 0:
        raise ValueError(f"Number of time steps {number_of_time_steps} must be divisible by number of temporal processors {temporal_processors}.")
    
    # Check if the number of time steps is of the form 2^k+1, 2^k, ..., 2^k where k is an integer
    n = number_of_time_steps - 1
    nbyp = n // temporal_processors
    if ((nbyp & (nbyp-1) == 0) and nbyp != 0):
        raise ValueError(f"Number of time steps {number_of_time_steps} must be of the form 2^k+1, 2^k, ..., 2^k where k is an integer.")
    
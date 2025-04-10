from firedrake import *
from asQ import (create_ensemble)
from CyclicReduction.check_setup import create_time_partition

problem_parameters = {
    "Number of time windows": 1, # No functionality for this yet
    "Number of temporal processors": 4,
    "Number of time steps": 9, # Number of time steps must fit into a list following [2^k+1, 2^k, ..., 2^k] where k is an integer and the list length is equal to the number of temporal processors.
    "dt": 0.001,
    "nx": 4,
    "ny": 4,
    "degree_space": 1,
    "theta": 1,
}

processors = COMM_WORLD.size
n_timesteps = problem_parameters["Number of time steps"]
temporal_processors = problem_parameters["Number of temporal processors"]
nx = problem_parameters['nx']
ny = problem_parameters['ny']
dt = problem_parameters['dt']
degree_space = problem_parameters['degree_space']
theta = problem_parameters['theta']
time_partition = create_time_partition(n_timesteps-1, temporal_processors)
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
mesh = UnitSquareMesh(nx = nx, ny = ny, distribution_parameters = distribution_parameters, comm = ensemble.comm)
n = FacetNormal(mesh)
V = FunctionSpace(mesh, "CG", degree_space)
x, y = SpatialCoordinate(V.mesh())



def u(t):

    u_exact = Function(V)
    u_exact.project(cos(pi*x)*cos(2*pi*y))
    u_exact.dat.data[:] *= np.exp(-5*pi*t)
    return u_exact.dat.data[:]

u_exact = u(0.1)
if ensemble.ensemble_comm.rank == 0:
    PETSc.Sys.Print(f"u_exact: {u_exact}", comm = COMM_SELF)



def get_global_rank(ensemble) -> int:
    """
    Given the total number of ranks, the local temporal rank, and the local spatial rank,
    this function computes the global rank.
    
    Args:
        total_ranks (int): Total number of ranks.
        local_temporal_rank (int): Local temporal rank.
        local_spatial_rank (int): Local spatial rank.
    Returns:
        int: Global rank.

    Example:
        >>> get_global_rank(8, 1, 2)
        10
    """

    #total_ranks, local_temporal_rank, local_spatial_rank

    # If total_ranks is 8, temporal rank is 0, and spatial rank is 2
    #return local_temporal_rank * total_ranks + local_spatial_rank



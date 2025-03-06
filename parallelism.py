from firedrake import *
from firedrake.petsc import PETSc


# We create an ensamble with the global communicator COMM_WORLD


# Total number of ranks/processes
number_of_ranks : int = COMM_WORLD.size

# Define how many temporal ranks/processes we want per spatial separation (per ensemble)
number_of_temporal_ranks : int = 2

if number_of_ranks % number_of_temporal_ranks != 0:
        raise ValueError("Number of time slices must be exact factor of number of MPI ranks")

# This gives the number of ensembles
number_of_spatial_ranks : int = number_of_ranks // number_of_temporal_ranks

PETSc.Sys.Print('Setting up mesh across %d processes. There are %d spatial processes, each with %d temporal processes' % (number_of_ranks, number_of_spatial_ranks, number_of_temporal_ranks))

# Next we create the ensembles
my_ensemble = Ensemble(COMM_WORLD, number_of_spatial_ranks)

# Let's create a problem.
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

mesh = UnitSquareMesh(5,5, distribution_parameters = distribution_parameters, comm = my_ensemble.comm)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)

q = Constant(my_ensemble.ensemble_comm.rank + 1)
u.interpolate(sin(q*pi*x)*cos(q*pi*y))

# my_ensemble.reduce(u, usum, root)
# my_ensemble.allreduce(u, usum)
# my_ensemble.bcast(u, root)
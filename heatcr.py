from firedrake import *
from firedrake.petsc import PETSc
from time import time
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
)
import warnings
warnings.simplefilter("ignore", FutureWarning)

class ProblemParameters:
    def __init__(self):
        self.M = 25 # Number of spatial points
        self.dt = 0.001
        self.degree = {'space': 2,
                       'time': 0} # DG degree 0 gives backward Euler
        self.plot = True
        self.theta = 1 # Theta parameter for the time-stepping scheme

parameters = ProblemParameters()

# Create a time partition and an ensemble communicator
time_partition = [8,8,8,8]  # Total of 60 time steps
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Define mesh
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
mesh = UnitSquareMesh(nx = parameters.M, ny = parameters.M,
                      distribution_parameters=distribution_parameters,comm = ensemble.comm)

# Define function spaces
space_element = FiniteElement("CG", triangle, parameters.degree['space'])
U = FunctionSpace(mesh,space_element)

# I.C and B.C
x, y = SpatialCoordinate(U.mesh())
u0 = Function(U)

u0.interpolate(cos(pi*x)*cos(2*pi*y))
bcs = []
# u0.project(sin(0.25*pi*x)*cos(2*pi*y))
# bcs = [DirichletBC(U, 0, sub_domain=1)]

# Define forms
def form_mass(u, v):
    return u*v*dx

def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx

aaofunc = AllAtOnceFunction(ensemble, time_partition, U)
aaofunc.initial_condition.assign(u0)

aaoform = AllAtOnceForm(aaofunc, 
                        parameters.dt, 
                        parameters.theta, 
                        form_mass,
                        form_function, 
                        bcs=bcs)

#Set up solver parameters
patch_parameters = {'patch_type': 'star',
                    'construct_dim': 0, 
                    'mat_ordering_type': 'natural',
                   }

solver_parameters = {'snes_type': 'ksponly',
                     'mat_type': 'aij', 
                     'ksp_type': 'fgmres',
                     'ksp_monitor_true_residual': None,
                     'ksp_max_it': 100,
                     'ksp_rtol': 1e-6,
                     'ksp_atol': 1e-6,
                     'pc_type': 'python',
                     'pc_python_type': 'CyclicReduction.CyclicReductionPC4',
                     'cr_opts': patch_parameters
                    }

solver = AllAtOnceSolver(aaoform, 
                         aaofunc, 
                         solver_parameters,
                         )

# Some useful prints
processes = COMM_WORLD.size
PETSc.Sys.Print(f"Running with {processes} MPI processes. {len(time_partition)} in time, each with {processes//len(time_partition)} in space")
PETSc.Sys.Print(f"Time partition: {time_partition}, total time steps: {sum(time_partition)}")

A,_ = solver.snes.ksp.getOperators()
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")

space_dofs = U.dim()
time_dofs = sum(time_partition)
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's space: {space_dofs}, DOF's time: {time_dofs}, DOF's total: {total_dofs} \n")

start = time()
solver.solve()
PETSc.Sys.Print(f"Finished solve in {time()-start}s")

# --------------------------------------------------
# Output results to ParaView
# --------------------------------------------------
if parameters.plot:
    import shutil
    import os
    from pyop2.mpi import MPI

    file_dir = "ParaView"

    # Clear directory if it exists
    if COMM_WORLD.rank == 0:
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)
        os.makedirs(f"{file_dir}", exist_ok=True)

    COMM_WORLD.Barrier()

    # Only first temporal rank handles file output
    if ensemble.ensemble_comm.rank == 0:
        vtkfile = VTKFile(f"{file_dir}/u_t.pvd", comm=ensemble.comm)

    # Sending data from all time ranks
    for step in range(aaofunc.ntimesteps):
        if aaoform.layout.is_local(step):
            local_step = aaofunc.transform_index(step, from_range='window')
            w_local = Function(U)
            w_local.assign(aaofunc[local_step])
            w_local.rename("u")
            if ensemble.ensemble_comm.rank != 0:
                ensemble.send(w_local, dest=0, tag=step)
            else:
                w_recv = w_local.copy(deepcopy=True)  # if rank 0 owns it, just copy

        if ensemble.ensemble_comm.rank == 0:
            if not aaoform.layout.is_local(step):
                w_recv = Function(U, name="u")
                ensemble.recv(w_recv, source=MPI.ANY_SOURCE, tag=step)

            vtkfile.write(w_recv, time=step * parameters.dt)
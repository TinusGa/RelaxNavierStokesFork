import faulthandler; faulthandler.enable()
import numpy as np
import os

from mpi4py import MPI
MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)

from firedrake import *
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
    LinearSolver,
    SharedArray,
)
from asQ.pencil import Pencil, Subcomm
from CyclicReduction.check_setup import check_setup, create_time_partition
import time
import warnings
warnings.simplefilter("ignore", FutureWarning)

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

problem_parameters = {
    "Number of time windows": 1, # No functionality for this yet
    "Number of temporal processors": 8, # Optimal choice is the root of the number of time steps
    "Number of time steps": 1, # Number of time steps must fit into a list following [2^k+1, 2^k, ..., 2^k] where k is an integer and the list length is equal to the number of temporal processors.
    "dt": 0.00001,
    "nx": 4, # Some choices of nx & ny lead to an uneven mesh distribution across ensemble ranks and causes the program to crash. Very strange
    "ny": 4, 
    "degree_space": 2,
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

# Create a time partition and an ensemble communicator
time_partition = create_time_partition(n_timesteps-1, temporal_processors)
time_partition = [1]
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Create a mesh with nx+1 and ny+1 vertices
mesh = UnitSquareMesh(nx = nx, ny = ny, comm = ensemble.comm)

n = FacetNormal(mesh)

V = FunctionSpace(mesh, "CG", degree_space)
x, y = SpatialCoordinate(V.mesh())

u0 = Function(V)
u0.project(cos(pi*x)*cos(2*pi*y))
bcs = []

# PETSc.Sys.Print(f"Rank {COMM_WORLD.rank} ensemble {ensemble.ensemble_comm.rank}: V.dof_count = {V.dof_count}, u0.local_size = {u0.dat._vec.getLocalSize()}", comm=fd.COMM_SELF)


# bcs = [DirichletBC(V, 0, sub_domain=1)]

# u0.project(sin(0.25*pi*x)*cos(2*pi*y))


def form_mass(u, v):
    return u*v*dx

def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx

aaofunc = AllAtOnceFunction(ensemble, time_partition, V)

aaoform = AllAtOnceForm(aaofunc, 
                        dt, 
                        theta, 
                        form_mass,
                        form_function, 
                        bcs=bcs)


solver_parameters = {
    'snes_type': 'ksponly',
    'mat_type': 'aij',
    'ksp_type': 'richardson',  # fgmres, gmres, richardson ok. cg, chebyshev no.
    'ksp_richardson_scale': 1,
    'ksp_max_it': 50,
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'pc_type': 'python',
    'pc_python_type': 'CyclicReduction.CyclicReductionPC3',
}

aaosolver = AllAtOnceSolver(aaoform, 
                            aaofunc, 
                            solver_parameters)


# Some useful prints
PETSc.Sys.Print(f"Running with {processors} MPI processes")
PETSc.Sys.Print(f"Time partition: {time_partition}, total time steps: {n_timesteps}")

A,_ = aaosolver.snes.ksp.getOperators()
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")

space_dofs = (nx+1)*(ny+1)*degree_space
time_dofs = sum(time_partition)
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's space: {space_dofs}, DOF's time: {time_dofs}, DOF's total: {total_dofs} \n")


start = time.time()
aaofunc.assign(u0)
for i in range(1):
    aaosolver.solve()
PETSc.Sys.Print(f"Global solve time: {time.time()-start}s")
# --------------------------------------------------
# Postprocessing
# --------------------------------------------------

# We find the L2-error at each timestep
# q_exact = Function(V)
# errors = SharedArray(time_partition, comm=ensemble.ensemble_comm)
# times = SharedArray(time_partition, comm=ensemble.ensemble_comm)
# def window_postproc(aaofunc):
#     total_dof = 0
#     for step in range(aaofunc.ntimesteps):
#         if aaoform.layout.is_local(step):
#             local_step = aaofunc.transform_index(step, from_range='window')
#             t = aaoform.time[local_step]
#             q_exact.project(exp(-5*pi*t)*cos(pi*x)*cos(2*pi*y))
#             total_dof += q_exact.dof_dset.size
#             qp = aaofunc[local_step]
#             errors.dlocal[local_step] = errornorm(qp, q_exact)
#             times.dlocal[local_step] = t
#     errors.synchronise()
#     times.synchronise()
#     nsteps = aaofunc.ntimesteps
#     # Format each entry to fixed width
#     col_width = 10 # Adjust as needed
#     time_row = "".join(f"{times.dglobal[i]:>{col_width}.3f}" for i in range(nsteps))
#     qerr_row = "".join(f"{errors.dglobal[i]:>{col_width}.3e}" for i in range(nsteps))

#     # Add labels
#     time_row = f"{'Times:':<7}" + time_row
#     qerr_row = f"{'Errors:':<7}" + qerr_row

#     # Print both rows
#     PETSc.Sys.Print(time_row)
#     PETSc.Sys.Print(qerr_row)
    
# window_postproc(aaofunc)

# exact_sol = AllAtOnceFunction(ensemble, time_partition, V)
# q_exact = Function(V)

# subcomm = Subcomm(exact_sol.ensemble.ensemble_comm, [0, 1])
# nlocal = exact_sol.field_function_space.node_set.size # Spatial DOFs for this rank
# NN = np.array([exact_sol.ntimesteps, nlocal], dtype=int)
# p0 = Pencil(subcomm, NN, axis=1)
# a0 = np.zeros(p0.subshape,dtype=np.float64)

# # exact_array = SharedArray(time_partition, comm=ensemble.ensemble_comm)
# exact_sol.zero()
# exact_sol.initial_condition.assign(u0)
# for step in range(exact_sol.ntimesteps):
#     if aaoform.layout.is_local(step):

#         local_step = aaofunc.transform_index(step, from_range='window')
#         t = aaoform.time[local_step]
#         q_exact.project(exp(-5*pi*t)*cos(pi*x)*cos(2*pi*y))

#         a0[local_step,:] = q_exact.dat._vec.getArray()
        
#         with exact_sol.global_vec_wo() as gvec:
#             gvec.array[:] = a0.reshape(-1)[:]


# --------------------------------------------------
# Plot a GIF of the solution
# --------------------------------------------------
plot = False
if plot:
    import shutil
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
            w_local = Function(V)
            w_local.assign(aaofunc[local_step])
            w_local.rename("u")
            if ensemble.ensemble_comm.rank != 0:
                ensemble.send(w_local, dest=0, tag=step)
            else:
                w_recv = w_local.copy(deepcopy=True)  # if rank 0 owns it, just copy

        if ensemble.ensemble_comm.rank == 0:
            if not aaoform.layout.is_local(step):
                w_recv = Function(V, name="u")
                ensemble.recv(w_recv, source=MPI.ANY_SOURCE, tag=step)

            vtkfile.write(w_recv, time=step * dt)

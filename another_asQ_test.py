import faulthandler; faulthandler.enable()
import numpy as np

from firedrake import *
import firedrake as fd
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
    "Number of temporal processors": 4, # Optimal choice is the root of the number of time steps
    "Number of time steps": 5, # Number of time steps must fit into a list following [2^k+1, 2^k, ..., 2^k] where k is an integer and the list length is equal to the number of temporal processors.
    "dt": 0.001,
    "nx": 32,
    "ny": 32,
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

# Create a time partition and an ensemble communicator
time_partition = create_time_partition(n_timesteps-1, temporal_processors)
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Create a mesh with nx+1 and ny+1 vertices
mesh = UnitSquareMesh(nx = nx, ny = ny, comm = ensemble.comm)
n = FacetNormal(mesh)

V = FunctionSpace(mesh, "CG", degree_space)
x, y = SpatialCoordinate(V.mesh())

u0 = Function(V)
# u0.project(cos(pi*x)*cos(2*pi*y))
# bcs = []

bcs = [DirichletBC(V, 0, sub_domain=1)]

u0.project(sin(0.25*pi*x)*cos(2*pi*y))


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

# asQ solver parameters
# solver_parameters = {
#     'snes_type': 'ksponly',
#     'mat_type': 'mpiaij',
#     'ksp_type': 'richardson',
#     'ksp_max_it': 0, # Since Cyclic Reduction is a direct solver/method
#     #'ksp_rtol': 1e-12,
#     'ksp_monitor': None,
#     'ksp_converged_rate': None,
#     'pc_type': 'python',
#     'pc_python_type': 'CyclicReduction.CyclicReductionPC',
#     'cyclic_reduction_pc_factor_mat_solver_type': 'mumps',
# }

solver_parameters = {
'snes_type': 'ksponly',
'mat_type': 'matfree',
'ksp_type': 'richardson',
'ksp_rtol': 1e-12,
'ksp_monitor': None,
'ksp_converged_rate': None,
'pc_type': 'python',
'pc_python_type': 'asQ.CirculantPC',
'circulant_block': {'pc_type': 'lu'},
'circulant_alpha': 1e-4}

# solver_parameters = {
#     'ksp_monitor': None,
#     'ksp_converged_rate': None,
#     'snes_type': 'ksponly',
#     'mat_type': 'matfree',
#     'ksp_type': 'richardson',
#     'ksp_rtol': 1e-10,
#     'pc_type': 'python',
#     'pc_python_type': 'asQ.CirculantPC',
#     'circulant_alpha': 1e-4,
#     'circulant_block': {
#         'ksp_rtol': 1e-6,
#         'ksp_type': 'gmres',
#         'pc_type': 'ilu',
#     },
# }

# solver_parameters = {
#     'snes_type': 'ksponly',
#     'mat_type': 'matfree',
#     'ksp_type': 'richardson',
#     'ksp_rtol': 1e-12,
#     'ksp_monitor': None,
#     'ksp_converged_rate': None,
#     'pc_type': 'python',
#     'pc_python_type': 'asQ.SliceJacobiPC', # to replace 'pc_python_type': 'asQ.CirculantPC',
#     'slice_jacobi_nsteps': time_partition[0]*4,
#     #'slice_jacobi_slice': {'pc_type': 'lu'}
# }

# solver_parameters = {
#     'snes_type': 'ksponly',
#     'mat_type': 'matfree',
#     'ksp_type': 'richardson',
#     'ksp_rtol': 1e-12,
#     'ksp_monitor': None,
#     'ksp_converged_rate': None,
#     'pc_type': 'python',
#     'pc_python_type': 'asQ.JacobiPC',
# }


aaosolver = AllAtOnceSolver(aaoform, 
                            aaofunc, 
                            solver_parameters)


# Some useful prints
PETSc.Sys.Print(f"Running with {processors} MPI processes")
PETSc.Sys.Print(f"Time partition: {time_partition}, total time steps: {n_timesteps}")

A,_ = aaosolver.snes.ksp.getOperators()
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")
#PETSc.Sys.Print(f"Ownership ranges of A: {A.getOwnershipRanges()}") 
#PETSc.Sys.Print(f"Ownership ranges col of A: {A.getOwnershipRangesColumn()}")

space_dofs = (nx+1)*(ny+1)*degree_space
time_dofs = sum(time_partition)
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's space: {space_dofs}, DOF's time: {time_dofs}, DOF's total: {total_dofs} \n")
# PETSc.Sys.Print(f"")

# Solves over windows. Each window is solved using space-time parallelism. 
# Doing the loop over a single step should solve the entire system all-at-once.

start = time.time()

aaofunc.assign(u0)
for i in range(1):
    aaosolver.solve()
    #aaofunc.bcast_field(-1, aaofunc.initial_condition)
    # aaofunc.assign(aaofunc.initial_condition)

PETSc.Sys.Print(f"Global solve time: {time.time()-start}s")

# final_sol = aaosolver.aaofunc._vec.getArray() # aaosolver.aaofunc._vec.getArray() -> np.array() with final sol?
# PETSc.Sys.Print(f"{type(aaosolver.aaofunc._vec.getArray())}") 

# PETSc.Sys.Print(f"yvec = {aaofunc._vec.view()}")

 # We find the L2-error at each timestep
q_exact = Function(V)
errors = SharedArray(time_partition, comm=ensemble.ensemble_comm)
times = SharedArray(time_partition, comm=ensemble.ensemble_comm)
def window_postproc(aaofunc):
    total_dof = 0
    for step in range(aaofunc.ntimesteps):
        if aaoform.layout.is_local(step):
            local_step = aaofunc.transform_index(step, from_range='window')
            t = aaoform.time[local_step]
            q_exact.project(exp(-5*pi*t)*cos(pi*x)*cos(2*pi*y))
            total_dof += q_exact.dof_dset.size
            qp = aaofunc[local_step]
            errors.dlocal[local_step] = errornorm(qp, q_exact)
            times.dlocal[local_step] = t
    errors.synchronise()
    times.synchronise()
    nsteps = aaofunc.ntimesteps
    # Format each entry to fixed width
    col_width = 10 # Adjust as needed
    time_row = "".join(f"{times.dglobal[i]:>{col_width}.3f}" for i in range(nsteps))
    qerr_row = "".join(f"{errors.dglobal[i]:>{col_width}.3e}" for i in range(nsteps))

    # Add labels
    time_row = f"{'Times:':<7}" + time_row
    qerr_row = f"{'Errors:':<7}" + qerr_row

    # Print both rows
    PETSc.Sys.Print(time_row)
    PETSc.Sys.Print(qerr_row)
    
window_postproc(aaofunc)

exact_sol = AllAtOnceFunction(ensemble, time_partition, V)
q_exact = Function(V)

subcomm = Subcomm(exact_sol.ensemble.ensemble_comm, [0, 1])
nlocal = exact_sol.field_function_space.node_set.size # Spatial DOFs for this rank
NN = np.array([exact_sol.ntimesteps, nlocal], dtype=int)
p0 = Pencil(subcomm, NN, axis=1)
a0 = np.zeros(p0.subshape,dtype=np.float64)

# exact_array = SharedArray(time_partition, comm=ensemble.ensemble_comm)
exact_sol.zero()
exact_sol.initial_condition.assign(u0)
for step in range(exact_sol.ntimesteps):
    if aaoform.layout.is_local(step):

        local_step = aaofunc.transform_index(step, from_range='window')
        t = aaoform.time[local_step]
        q_exact.project(exp(-5*pi*t)*cos(pi*x)*cos(2*pi*y))

        a0[local_step,:] = q_exact.dat._vec.getArray()
        
        with exact_sol.global_vec_wo() as gvec:
            gvec.array[:] = a0.reshape(-1)[:]

        #exact_sol[local_step].copy(q_exact)

window_postproc(exact_sol)

# set up diagnostic recording

linear_iterations = 0
nonlinear_iterations = 0
total_timesteps = 0
total_windows = 0

import argparse

parser = argparse.ArgumentParser(
    description='ParaDiag timestepping for scalar advection of a Gaussian bump in a periodic square with DG in space and implicit-theta in time. Based on the Firedrake DG advection example https://www.firedrakeproject.org/demos/DG_advection.py.html',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=16, help='Number of cells along each square side.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar spaces.')
parser.add_argument('--theta', type=float, default=theta, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.2, help='Width of the Gaussian bump.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--nsample', type=int, default=32, help='Number of sample points for plotting.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--mpeg', action='store_true', help='Create mp4 of timeseries')

args = parser.parse_known_args()
args = args[0]

w = Function(V)

# The last time-slice will be saving snapshots to create an animation.
# The layout member describes the time_partition.
# layout.is_local(i) returns True/False if the timestep index i is on the
# current time-slice. Here we use -1 to mean the last timestep in the window.


# timeseries = [u0.copy()]



# for step in range(aaofunc.ntimesteps):
#     if aaoform.layout.is_local(step):
#         local_step = aaofunc.transform_index(step, from_range='window')
#         w.assign(aaofunc[local_step])
#         timeseries.append(w.copy(deepcopy=True))



# # Make an animation from the snapshots we collected and save it to periodic.mp4.
# if True:
#     PETSc.Sys.Print("Creating mp4 of timeseries")
#     fn_plotter = fd.FunctionPlotter(mesh, num_sample_points=args.nsample)

#     fig, axes = plt.subplots()
#     axes.set_aspect('equal')
#     colors = fd.tripcolor(w, num_sample_points=args.nsample, vmin=1, vmax=2, axes=axes)
#     fig.colorbar(colors)

#     def animate(q):
#         colors.set_array(fn_plotter(q))

#     interval = 1e2
#     animation = FuncAnimation(fig, animate, frames=timeseries, interval=interval)

#     animation.save("periodic.gif", writer="ffmpeg")
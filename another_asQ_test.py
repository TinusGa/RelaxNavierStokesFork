import faulthandler; faulthandler.enable()
import numpy as np

from firedrake import *
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

# opts = PETSc.Options()
# opts.setValue("ksp_monitor_true_residual", "")
# opts.setValue("ksp_converged_reason", "")

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
#check_setup(problem_parameters)

# Create a time partition and an ensemble communicator
time_partition = create_time_partition(n_timesteps-1, temporal_processors)
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Create a mesh with nx+1 and ny+1 vertices
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
mesh = UnitSquareMesh(nx = nx, ny = ny, distribution_parameters = distribution_parameters, comm = ensemble.comm)
# mesh = UnitSquareMesh(nx = nx, ny = ny, comm = ensemble.comm)
n = FacetNormal(mesh)

V = FunctionSpace(mesh, "CG", degree_space)
x, y = SpatialCoordinate(V.mesh())

u0 = Function(V)
#u0.project(cos(pi*x)*cos(2*pi*y))
#bcs = []

u0.project(sin(0.25*pi*x)*cos(2*pi*y))
bcs = [DirichletBC(V, 0, sub_domain=1)]


def form_mass(u, v):
    return u*v*dx

def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx

aaofunc = AllAtOnceFunction(ensemble, time_partition, V)
aaofunc.initial_condition.assign(u0)

aaoform = AllAtOnceForm(aaofunc, 
                        dt, 
                        theta, 
                        form_mass,
                        form_function, 
                        bcs=bcs)

# asQ solver parameters
solver_parameters = {
    'snes_type': 'ksponly',
    'mat_type': 'mpiaij',
    'ksp_type': 'richardson',
    'ksp_max_it': 0, # Since Cyclic Reduction is a direct solver/method
    #'ksp_rtol': 1e-12,
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'pc_type': 'python',
    'pc_python_type': 'CyclicReduction.CyclicReductionPC',
    'cyclic_reduction_pc_factor_mat_solver_type': 'mumps',
}

# solver_parameters = {
# 'snes_type': 'ksponly',
# 'mat_type': 'matfree',
# 'ksp_type': 'richardson',
# 'ksp_rtol': 1e-12,
# 'ksp_monitor': None,
# 'ksp_converged_rate': None,
# 'pc_type': 'python',
# 'pc_python_type': 'asQ.CirculantPC',
# 'circulant_block': {'pc_type': 'lu'},
# 'circulant_alpha': 1e-4}

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


aaofunc.assign(u0)

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

for i in range(1):
    aaosolver.solve()
    # aaofunc.bcast_field(-1, aaofunc.initial_condition)
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
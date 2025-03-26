from firedrake import *
from firedrake.petsc import PETSc
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
    LinearSolver,
)

import warnings
warnings.simplefilter("ignore", FutureWarning)

time_partition = [8, 8, 8, 8] # Add one additional time step to the first partition for an (n+1) - setup. Rest of the partitions should be 2^k for som int k. 

ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
nx = 9
ny = 9
mesh = UnitSquareMesh(nx = nx, ny = ny, distribution_parameters=distribution_parameters, comm = ensemble.comm)

processors = COMM_WORLD.size # total number of processors
temporal_processors = len(time_partition) # number of temporal processors

# The all-at-once matrix must be of block size (n+1)x(n+1) where n = p*2^k. p is the number of temporal processes. k is an integer.
# If the spatial discretization has m DOF's, then all-at-once matrix should have total size (n+1)*m x (n+1)*m.

N = 20 # This is useless per now
dt = 0.001

n = FacetNormal(mesh)

degree_space = 1

# Expected dof's in space:
space_dofs = (nx+1)*(ny+1)*degree_space
PETSc.Sys.Print(f"DOF's space: {space_dofs}")
# Expected dof's in time
time_dofs = sum(time_partition)
PETSc.Sys.Print(f"DOF's time: {time_dofs}")
# Total
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's total: {total_dofs}")
# dof's division per proc:
dof_by_total_proc = total_dofs/processors
dof_distribution = [int(dof_by_total_proc*i) for i in range(processors+1)] # Does not take into account overlapping dof's between procs
PETSc.Sys.Print(f"DOF's distribution: {dof_distribution}")


V = FunctionSpace(mesh, "CG", degree_space)

x, y = SpatialCoordinate(V.mesh())
u0 = Function(V)
u0.project(sin(pi*x)*cos(2*pi*y))
################################################

aaofunc = AllAtOnceFunction(ensemble, time_partition, V)
aaofunc.initial_condition.assign(u0)

dt = 0.05
theta = 1

bcs = [DirichletBC(V, 0, sub_domain=1)]

def form_mass(u, v):
    return u*v*dx

def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx

aaoform = AllAtOnceForm(aaofunc, 
                        dt, 
                        theta, 
                        form_mass,
                        form_function, 
                        bcs=bcs)

# asQ solver parameters
solver_parameters = {
    'snes_type': 'ksponly',
    'mat_type': 'aij',
    'ksp_type': 'richardson',
    'ksp_rtol': 1e-12,
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'pc_type': 'python',
    'pc_python_type': 'CyclicReduction.CyclicReductionPC', # to replace 'pc_python_type': 'asQ.CirculantPC',
    'cyclic_reduction_nsteps': time_partition[0], # n steps per time processor of CR
    #'circulant_block': {'pc_type': 'lu'},
    #'circulant_alpha': 1e-4
}

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
#     'pc_python_type': 'asQ.JacobiPC', # to replace 'pc_python_type': 'asQ.CirculantPC',
#     #'slice_jacobi_nsteps': time_partition[0]*4,
#     #'slice_jacobi_slice': {'pc_type': 'lu'}
# }

# Solver Parameters for heat equation from RelaxNavierStokes code.
# Want to transition from above parameters to the ones below.
 
# solver_parameters = {'snes_type': 'ksponly',
#                     'mat_type': 'aij',
#                     'ksp_type': 'fgmres',
#                     "ksp_monitor_true_residual": None,
#                     "ksp_max_it": 100,
#                     "ksp_gmres_restart": 100,
#                     "ksp_atol": 1e-6,
#                     "ksp_rtol": 1e-6,
#                     'pc_type': 'mg',
#                     "pc_mg_type": "multiplicative",
#                     "pc_mg_cycles": "v",
#                     "mg_levels_ksp_type": "chebyshev",
#                     "mg_levels_ksp_chebyshev_esteig": "0,0.25,0,1.05",
#                     "mg_levels_ksp_max_it": 2,
#                     "mg_levels_ksp_convergence_test": "skip",
#                     "mg_levels_pc_type": "python",
#                     "mg_levels_pc_python_type": "firedrake.ASMStarPC",
#                     "mg_levels_pc_star_construct_dim": 0,
#                     "mg_levels_pc_star_sub_sub_pc_type": "lu",
#                     "mg_levels_pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
#                     "mg_coarse_pc_type": "python",
#                     "mg_coarse_pc_python_type": "firedrake.AssembledPC",
#                     "mg_coarse_assembled_pc_type": "lu",
#                     "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
#                     }

aaosolver = AllAtOnceSolver(aaoform, 
                            aaofunc, 
                            solver_parameters)


aaofunc.assign(u0)

A,_ = aaosolver.snes.ksp.getOperators()

PETSc.Sys.Print(f"Size A: {A.getSize(),A.getSizes()}")
PETSc.Sys.Print(f"Ownership ranges: {A.getOwnershipRanges()}")
PETSc.Sys.Print(f"View: {A.view()}")

# Solves over windows. Each window is solved using space-time parallelism. 
# Doing the loop over a single step should solve the entire system all-at-once.
for i in range(1):
    aaosolver.solve()
    aaofunc.bcast_field(-1, aaofunc.initial_condition)
    aaofunc.assign(aaofunc.initial_condition)

final_sol = aaosolver.aaofunc._vec.getArray() # aaosolver.aaofunc._vec.getArray() -> np.array() with final sol?


PETSc.Sys.Print(f"{type(aaosolver.aaofunc._vec.getArray())}") 



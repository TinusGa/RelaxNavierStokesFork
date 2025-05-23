from firedrake import *
from firedrake.petsc import PETSc
from time import time
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
)
from CyclicReduction.check_setup import create_time_partition
import warnings
warnings.simplefilter("ignore", FutureWarning)

class ProblemParameters:
    def __init__(self):
        self.N = 9 # Number of time steps
        self.dt = 0.001 # Specified instead of end time
        self.M = 9 # Number of spatial points
        self.Mbase = 3 # Number of spatial points in base mesh
        self.Mref = 2 # Number of refinements in the mesh hierarchy
        self.degree = {'space': 1,
                       'time': 0} # DG degree 0 gives backward Euler
        self.plot = False
        self.solver = None
        self.Pt = 1 # Processors in time
        self.theta = 1 # Theta parameter for the time-stepping scheme

parameters = ProblemParameters()

# Create a time partition and an ensemble communicator
time_partition = create_time_partition(parameters.N-1, parameters.Pt)
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Define mesh
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

base_mesh = UnitSquareMesh(nx = parameters.Mbase, ny = parameters.Mbase,
                           distribution_parameters=distribution_parameters,comm = ensemble.comm)

mesh_hierarchy = MeshHierarchy(base_mesh,parameters.Mref)

mesh = mesh_hierarchy[-1] # This is the finest mesh

# Define function space
space_element = FiniteElement("CG", triangle, parameters.degree['space'])
U = FunctionSpace(mesh,space_element)

# Define initial condition
x, y = SpatialCoordinate(U.mesh())

u0 = Function(U)
u0.project(cos(pi*x)*cos(2*pi*y))

bcs = []

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
#Set up solver
solver_parameters = {'snes_type': 'ksponly',
                     'mat_type': 'aij',
                     'ksp_type': 'fgmres',
                     'ksp_monitor_true_residual': None,
                     'ksp_max_it': 100,
                     'ksp_gmres_restart': 100,
                     'ksp_atol': 1e-6,
                     'ksp_rtol': 1e-6,
                     'pc_type': 'python',
                     'pc_python_type': 'CyclicReduction.asQMGPC',
                     'pc_mg_type': 'multiplicative',
                     'mg_levels_ksp_type': 'chebyshev',
                     'mg_levels_ksp_chebyshev_esteig': '0,0.25,0,1.05',
                     'mg_levels_ksp_max_it': 2,
                     'mg_levels_ksp_convergence_test': 'skip',
                     'mg_levels_pc_type': 'python',
                     'mg_levels_pc_python_type': 'CyclicReduction.CyclicReductionPC3', # Contains firedrake.ASMStarPC
                     'mg_levels_pc_opts': {'atch_type': 'star',
                                           'construct_dim': 0,
                                           'mat_ordering_type': 'natural',
                                           },
                     'mg_coarse_pc_type': 'python',
                     'mg_coarse_pc_python_type': 'firedrake.AssembledPC', # Could also use CR here?
                     'mg_coarse_assembled_pc_type': 'lu',
                     'mg_coarse_assembled_pc_factor_mat_solver_type': 'mumps',
                    }

solver = AllAtOnceSolver(aaoform, 
                        aaofunc, 
                        solver_parameters)

# Some useful prints
processes = COMM_WORLD.size
PETSc.Sys.Print(f"Running with {processes} MPI processes. {parameters.Pt} in time, each with {processes//parameters.Pt} in space")
PETSc.Sys.Print(f"Time partition: {time_partition}, total time steps: {parameters.N}")

A,_ = solver.snes.ksp.getOperators()
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")

space_dofs = U.dim()
time_dofs = sum(time_partition)
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's space: {space_dofs}, DOF's time: {time_dofs}, DOF's total: {total_dofs} \n")

start = time()
solver.solve()
PETSc.Sys.Print(f"Finished solve in {time()-start}s")


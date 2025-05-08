"""
Test script solving the heat equation using standard space-time
finite elements
"""
from firedrake import *
from firedrake.petsc import PETSc
import matplotlib.pylab as plt
from time import time
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

import warnings
warnings.simplefilter("ignore", FutureWarning)

#Problem parameters used if running this script
class ProblemParameters:
    def __init__(self):
        self.N = 9 # Number of time steps
        self.dt = 0.001 #Specified instead of end time
        self.M = 10 # Number of spatial points
        self.Mbase = 5 # Number of spatial points in base mesh
        self.Mref = 2 # Number of refinements in the mesh hierarchy
        self.degree = {'space': 1,
                       'time': 0} # DG degree 0 gives backward Euler
        self.plot = False
        self.solver = None
        self.Pt = 4 # Processors in time
        self.theta = 1 # Theta parameter for the time-stepping scheme

parameters = ProblemParameters()

start_setup = time()

# Create a time partition and an ensemble communicator
time_partition = create_time_partition(parameters.N-1, parameters.Pt)
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Define mesh
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base_mesh = UnitSquareMesh(parameters.Mbase,parameters.Mbase,
                           distribution_parameters=distribution_parameters,comm = ensemble.comm)
spatial_mesh = MeshHierarchy(base_mesh,parameters.Mref)

mesh = spatial_mesh[-1]

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


#Set up solver
solver_parameters = {'snes_type': 'ksponly',
                    'mat_type': 'aij',
                    'ksp_type': 'fgmres',
                    "ksp_monitor_true_residual": None,
                    "ksp_max_it": 0,
                    "ksp_gmres_restart": 100,
                    "ksp_atol": 1e-6,
                    "ksp_rtol": 1e-6,
                    "pc_type":"python",
                    "pc_python_type": "CyclicReduction.Setup",
                    "pc_python_mg": {'pc_type':'mg',
                                     'pc_mg_type':'multiplicative',
                                     'pc_mg_cycles':'v',
                                     'mg_levels_ksp_type':'chebyshev',
                                     'mg_levels_ksp_chebyshev_esteig':'0,0.25,0,1.05',
                                     'mg_levels_ksp_max_it':2,
                                     'mg_levels_ksp_convergence_test':'skip',
                                     'mg_levels_pc_type':'python',
                                     'mg_levels_pc_python_type':'CyclicReduction.ASMStarPC',
                                     'mg_levels_pc_star_construct_dim':0,
                                     'mg_levels_pc_star_sub_sub_pc_type':'python',
                                     'mg_levels_pc_star_sub_sub_pc_python_type':'CyclicReduction.CyclicReductionPC2',
                                     'mg_coarse_pc_type':'python',
                                     'mg_coarse_pc_python_type':'lu',
                                     'mg_coarse_pc_factor_mat_solver_type':'mumps',
                                    },
                    }

AllAtOnce = True

if AllAtOnce:
    aaofunc = AllAtOnceFunction(ensemble, time_partition, U)
    aaofunc.initial_condition.assign(u0)

    aaoform = AllAtOnceForm(aaofunc, 
                            parameters.dt, 
                            parameters.theta, 
                            form_mass,
                            form_function, 
                            bcs=bcs)
    
    solver = AllAtOnceSolver(aaoform, 
                            aaofunc, 
                            solver_parameters)
else:
    u = Function(U)
    v = TestFunction(U)

    F = form_function(u, v, 0) - u0*v*dx

    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

end_setup = time()

PETSc.Sys.Print(f"Finished setup in {end_setup - start_setup:.2f} s")

start_solve = time()
solver.solve()
end_solve = time()

PETSc.Sys.Print(f"Finished solve in {end_solve - start_solve:.2f} s")

iterations = solver.snes.getLinearSolveIterations()

# #Get number of nonzero entries
A, P = solver.snes.ksp.getOperators()
nnz = int(A.getInfo()['nz_allocated'])


#Output relevant info
out = {'dof': U.dim(),
        'nnz': nnz,
        'iterations': iterations,
        'time_total': end_solve-start_setup,
        'time_solve': end_solve-start_solve}

PETSc.Sys.Print(out)

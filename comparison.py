from firedrake import *
from firedrake.petsc import PETSc
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
    LinearSolver
)
from time import time
import warnings
warnings.simplefilter("ignore", FutureWarning)
import os 
import csv

#Problem parameters used if running this script
class ProblemParameters:
    def __init__(self):
        self.M = 5 # Number of elements in the base mesh
        self.dt = 0.001
        self.theta = 1.0 # Time-stepping parameter (1.0 for Backward Euler)
        self.degree = {'space': 1,
                       'time': 0}
        self.plot = False  # Whether to output results to ParaView
        self.write_metrics = False  # Whether to write metrics to file

parameters = ProblemParameters()
time_partition = [2]
parameters.N = sum(time_partition) 

# Base mesh for both problems
distribution_parameters={"partition": True,
                            "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base_mesh = UnitSquareMesh(parameters.M,parameters.M,
                       distribution_parameters=distribution_parameters)

def BC_and_initial_condition_0(V, x, y, t=None):
    return [DirichletBC(V, 0, sub_domain=1)], sin(0.25*pi*x)*cos(2*pi*y)

def BC_and_initial_condition_1(V, x, y, t=None):
    return [], cos(pi*x)*cos(2*pi*y)

BC_and_initial_condition = BC_and_initial_condition_1

# ----------------------------------------------------------
#                          HEAT RNS-repo
# ---------------------------------------------------------- 
extruded_mesh = ExtrudedMesh(base_mesh, 
                             layers=parameters.N, 
                             layer_height=parameters.dt,
                             extrusion_type='uniform')

mesh = extruded_mesh[-1]
n = FacetNormal(mesh)

# Define function space
space_element = FiniteElement("CG", triangle, parameters.degree['space'])
time_element = FiniteElement("DG", interval, parameters.degree['time'])
spacetime_element = TensorProductElement(space_element,time_element)
U = FunctionSpace(mesh,spacetime_element)

#Define initial condition
x, y, t = SpatialCoordinate(U.mesh())
u0 = Function(U)
RNSbcs, RNSinitial_condition = BC_and_initial_condition(U, x, y, t)
u0.project(RNSinitial_condition)

#Set up residual
u = Function(U)
u.zero()
phi = TestFunction(U)

gradu = as_vector([u.dx(0),
                    u.dx(1)])
gradphi = as_vector([phi.dx(0),
                        phi.dx(1)])
def plus(v):
    return -0.5*jump(v,n[2]) + avg(v)

F_space = inner(gradu,gradphi) * dx(degree=0)
F_time = u.dx(2) * phi * dx(degree=0) - jump(u,n[2]) * plus(phi) * dS_h(degree=0)
F_ic = 0.5*(u-u0)*phi*ds_b
F = F_space + F_time + F_ic

# ------------------------------------------------------------
#                          HEAT CR-repo
# ------------------------------------------------------------
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)
mesh = base_mesh
space_element = FiniteElement("CG", triangle, parameters.degree['space'])
V = FunctionSpace(mesh, space_element)

x, y = SpatialCoordinate(V.mesh())
v0 = Function(V)
CRbcs, CRinitial_condition = BC_and_initial_condition(V, x, y)
v0.project(CRinitial_condition)

# Define forms
def form_mass(u, v):
    return u*v*dx

def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx

aaofunc = AllAtOnceFunction(ensemble, time_partition, V)
aaofunc.initial_condition.assign(v0)
aaofunc.zero(zero_ics=False)
aaoform = AllAtOnceForm(aaofunc, 
                        parameters.dt, 
                        parameters.theta, 
                        form_mass,
                        form_function, 
                        bcs=CRbcs)

# The two methods seem to have different 'rules' for when they stop depending on the tolerances.
solver_parameters_RNS = {'snes_type': 'ksponly',
                        'mat_type': 'aij',
                        'ksp_type': 'gmres',
                        "ksp_monitor_true_residual": None,
                        "ksp_max_it": 100,
                        "ksp_atol": 1e-16,
                        "ksp_rtol": 1e-16,
                        'pc_type': 'none',
                        }

solver_parameters_CR = {'snes_type': 'ksponly',
                        'mat_type': 'aij',
                        'ksp_type': 'gmres',
                        "ksp_monitor_true_residual": None,
                        "ksp_max_it": 100,
                        "ksp_atol": 1e-16,
                        "ksp_rtol": 1e-16,
                        'pc_type': 'none',
                        }
    
problem = NonlinearVariationalProblem(F, u, bcs=RNSbcs)
RNSsolver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters_RNS)

CRsolver = AllAtOnceSolver(aaoform,
                           aaofunc,
                           solver_parameters=solver_parameters_CR)

# Some useful prints
processes = COMM_WORLD.size
PETSc.Sys.Print(f"Running with {processes} MPI processes")

A,_ = RNSsolver.snes.ksp.getOperators()
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")

space_dofs = FunctionSpace(base_mesh, space_element).dim()
time_dofs = parameters.N
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's space: {space_dofs}, DOF's time: {time_dofs}, DOF's total: {total_dofs} \n")

PETSc.Sys.Print(f"Solving RNS-repo problem...")
RNSsolver.solve()

PETSc.Sys.Print(f"Solving CR-repo problem...")
CRsolver.solve()

# PETSc.Sys.Print(f"u: {u.dat.data[:]}")
# PETSc.Sys.Print(f"aaofunc: {aaofunc._vec.getArray()[:]}")




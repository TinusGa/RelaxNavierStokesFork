"""
Test script solving the heat equation using standard space-time
finite elements
"""
#Global imports
from firedrake import *
import matplotlib.pylab as plt
from time import time
import warnings
warnings.simplefilter("ignore", FutureWarning)

#Problem parameters used if running this script
class ProblemParameters:
    def __init__(self):
        self.N = 8
        self.dt = 0.001 #Specified instead of end time
        self.M = 9
        self.degree = {'space': 2,
                       'time': 0} # DG degree 0 gives backward Euler
        self.plot = True

parameters = ProblemParameters()


def plus(v):
    return -0.5*jump(v,n[2]) + avg(v)

#Define mesh
distribution_parameters={"partition": True,
                            "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base_mesh = UnitSquareMesh(parameters.M,parameters.M,
                       distribution_parameters=distribution_parameters)

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
u0.project(cos(pi*x)*cos(2*pi*y))
bcs = []
# u0.project(sin(0.25*pi*x)*cos(2*pi*y))
# bcs = [DirichletBC(U, 0, sub_domain=1)]

#Set up residual
u = Function(U)
phi = TestFunction(U)

gradu = as_vector([u.dx(0),
                    u.dx(1)])
gradphi = as_vector([phi.dx(0),
                        phi.dx(1)])

F_space = inner(gradu,gradphi) * dx(degree=16)
F_time = u.dx(2) * phi * dx(degree=16) - jump(u,n[2]) * plus(phi) * dS_h(degree=16)
F_ic = 0.5*(u-u0)*phi*ds_b

F = F_space + F_time + F_ic


solver_parameters = {'snes_type': 'ksponly',
                    'mat_type': 'aij',
                    'ksp_type': 'fgmres',
                    "ksp_monitor_true_residual": None,
                    "ksp_max_it": 100,
                    "ksp_atol": 1e-6,
                    "ksp_rtol": 1e-6,
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.ASMStarPC',
                    'pc_star_construct_dim': 0,
                    'pc_star_sub_sub_pc_type': 'lu',
                    'pc_star_sub_sub_pc_factor_mat_solver_type': 'umfpack',}
    
problem = NonlinearVariationalProblem(F, u)
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

# Some useful prints
processes = COMM_WORLD.size
PETSc.Sys.Print(f"Running with {processes} MPI processes")

A,_ = solver.snes.ksp.getOperators()
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")

space_dofs = FunctionSpace(base_mesh, space_element).dim()
time_dofs = parameters.N
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's space: {space_dofs}, DOF's time: {time_dofs}, DOF's total: {total_dofs} \n")

start = time()
solver.solve()
PETSc.Sys.Print(f"Finished solve in {time()-start}s")

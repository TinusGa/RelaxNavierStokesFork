from firedrake import *
from firedrake.petsc import PETSc
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
)
from time import time
import warnings
warnings.simplefilter("ignore", FutureWarning)

class ProblemParameters:
    def __init__(self):
        self.M = 8 # Number of elements in the base mesh
        self.mref = 1 # Number of mesh refinement levels for multigrid
        self.dt = 0.02
        self.R = Constant(1) # Reynolds number
        self.alpha = Constant(1) # Diffusion constant
        self.theta = 1.0 # Time-stepping parameter (1.0 for Backward Euler)
        self.degree = {'space': 2, 'time': 0}  # Space and time degrees
        self.plot = True  # Whether to output results to ParaView

parameters = ProblemParameters()

# 1. ENSEMBLE and MESH setup
# ----------------------------------------------------
time_partition = [16,16,16,16]
parameters.N = sum(time_partition)  # Total number of time steps
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base_mesh = PeriodicUnitSquareMesh(nx=parameters.M, ny=parameters.M, direction='both',
                                   distribution_parameters=distribution_parameters)
spatial_mh = MeshHierarchy(base_mesh,parameters.mref)
extruded_mesh = ExtrudedMeshHierarchy(spatial_mh, parameters.N*parameters.dt,
                        base_layer = parameters.N,
                        refinement_ratio=1,
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
u0.interpolate(sin(1*pi*(x-1))*sin(10*pi*(y-1)))

#Set up residual
u = Function(U)
phi = TestFunction(U)

D = Constant(0.001)  # Diffusion coefficient
r = Constant(0.5)  # Growth rate

def plus(v):
    return -0.5*jump(v,n[2]) + avg(v)

gradu = as_vector([u.dx(0),
                    u.dx(1)])
gradphi = as_vector([phi.dx(0),
                        phi.dx(1)])

F_space = D*inner(gradu,gradphi) * dx(degree=16) - r * u * (1 - u) * phi * dx(degree=16)
F_time = u.dx(2) * phi * dx(degree=16) - jump(u,n[2]) * plus(phi) * dS_h(degree=16)
F_ic = 0.5*(u-u0)*phi*ds_b

F = F_space + F_time + F_ic

# solver_parameters = {'snes_type': 'newtonls',
#                     'snes_ksp_ew': None,
#                     'snes_monitor': None,
#                     'mat_type': 'aij',
#                     'ksp_type': 'fgmres',
#                     "ksp_monitor_true_residual": None,
#                     "ksp_max_it": 100,
#                     "ksp_atol": 1e-6,
#                     "ksp_rtol": 1e-6,
#                     'pc_type': 'python',
#                     'pc_python_type': 'firedrake.ASMStarPC',
#                     'pc_star_construct_dim': 0,
#                     'pc_star_sub_sub_pc_type': 'lu',
#                     'pc_star_sub_sub_pc_factor_mat_solver_type': 'umfpack'}

solver_parameters = {'snes_type': 'newtonls',
                    'snes_ksp_ew': None,
                    'snes_monitor': None,
                    'mat_type': 'aij',
                    'ksp_type': 'fgmres',
                    "ksp_monitor_true_residual": None,
                    "ksp_max_it": 100,
                    "ksp_gmres_restart": 100,
                    "ksp_atol": 1e-6,
                    "ksp_rtol": 1e-6,
                    'pc_type': 'mg',
                    "pc_mg_type": "multiplicative",
                    "pc_mg_cycles": "v",
                    "mg_levels_ksp_type": "chebyshev",
                    "mg_levels_ksp_chebyshev_esteig": "0,0.25,0,1.05",
                    "mg_levels_ksp_max_it": 2,
                    "mg_levels_ksp_convergence_test": "skip",
                    "mg_levels_pc_type": "python",
                    "mg_levels_pc_python_type": "firedrake.ASMStarPC",
                    "mg_levels_pc_star_construct_dim": 0,
                    "mg_levels_pc_star_sub_sub_pc_type": "lu",
                    "mg_levels_pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
                    "mg_coarse_pc_type": "python",
                    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                    "mg_coarse_assembled_pc_type": "lu",
                    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                    }

problem = NonlinearVariationalProblem(F, u)
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)


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

vtkfile = VTKFile(f"Mashallah/u_t.pvd")
vtkfile.write(u)
from firedrake import *
from firedrake.petsc import PETSc
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver
)

time_partition = [2, 2, 2, 2]

ensemble = create_ensemble(time_partition, comm=COMM_WORLD)
############################################################
# mesh = SquareMesh(nx=32, ny=32, L=1, comm=ensemble.comm)

# x, y = SpatialCoordinate(mesh)
# V = FunctionSpace(mesh, "CG", 1)

# u0 = Function(V)
# u0.interpolate(sin(0.25*pi*x)*cos(2*pi*y))
############################################################
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base_mesh = UnitSquareMesh(nx = 5, ny = 5, distribution_parameters=distribution_parameters, comm = ensemble.comm)
spatial_mesh = MeshHierarchy(base_mesh, refinement_levels = 2)

N = 20
dt = 0.001
mesh_hierarchy = ExtrudedMeshHierarchy(
    base_hierarchy= spatial_mesh, 
    height = N*dt, 
    base_layer = N,
    refinement_ratio = 1,
    extrusion_type = 'uniform'
    )

mesh = mesh_hierarchy[-1]
n = FacetNormal(mesh)

degree_space = 1
V = FunctionSpace(mesh, "CG", degree_space)

x, y, t = SpatialCoordinate(V.mesh())
u0 = Function(V)
u0.project(sin(pi*x)+cos(2*pi*y))
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
    'circulant_alpha': 1e-4
}

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
for i in range(6):
    aaosolver.solve()
    aaofunc.bcast_field(-1, aaofunc.initial_condition)
    aaofunc.assign(aaofunc.initial_condition)



from firedrake import *
from firedrake.petsc import PETSc
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
)
import warnings
warnings.simplefilter("ignore", FutureWarning)

class ProblemParameters:
    def __init__(self):
        self.M = 4 # Number of elements in the base mesh
        self.mref = 1 # Number of mesh refinement levels for multigrid
        self.dt = 0.001
        self.R = Constant(1) # Reynolds number
        self.alpha = Constant(1) # Diffusion constant
        self.theta = 1.0 # Time-stepping parameter (1.0 for Backward Euler)
        self.space_degree = 2

parameters = ProblemParameters()

# 1. ENSEMBLE and MESH setup
# ----------------------------------------------------
time_partition = [4, 4]
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base_mesh = UnitSquareMesh(nx=parameters.M, ny=parameters.M,
                           distribution_parameters=distribution_parameters,
                           comm=ensemble.comm)
mesh_hierarchy = MeshHierarchy(base_mesh, parameters.mref)

# 2. FUNCTION SPACES and BOUNDARY CONDITIONS
# ---------------------------------------------
function_spaces = []
bcs_list = []

# Define a Constant to hold the current time for the BCs.
# Can be updated by the solver using a callback?
t_const = Constant(0.0)
x, y = SpatialCoordinate(mesh_hierarchy[-1])
ut_expr = as_vector((-cos(pi * x) * sin(pi * y) * exp(-2 * pi**2 * t_const),
                     sin(pi * x) * cos(pi * y) * exp(-2 * pi**2 * t_const)))

for mesh in mesh_hierarchy:
    V_el = VectorElement("CG", mesh.ufl_cell(), parameters.space_degree + 1)
    Q_el = FiniteElement("CG", mesh.ufl_cell(), parameters.space_degree)
    Z = FunctionSpace(mesh, MixedElement(V_el, Q_el))
    function_spaces.append(Z)

    # Create BCs for each level in the hierarchy
    bcs = [DirichletBC(Z.sub(0), ut_expr, "on_boundary")]
    bcs_list.append(bcs)


# 3. INITIAL CONDITION
# ----------------------
Z = function_spaces[-1]
z0 = Function(Z)
u0, p0 = z0.subfunctions

# Interpolate the initial velocity and set initial pressure
u0.interpolate(as_vector((-sin(pi * y) * cos(pi * x), cos(pi * y) * sin(pi * x))))
p0.assign(0.0) # Initial pressure can be zero

# 4. VARIATIONAL FORMS
# ----------------------------------

def form_mass(z, w):
    u, _ = split(z)
    phi, _ = split(w)
    return inner(u, phi) * dx

def form_function(z, w, t):
    u, p = split(z)
    phi, psi = split(w)
    
    convection = parameters.R * inner(dot(grad(u), u), phi) * dx
    pressure_term = p * div(phi) * dx
    diffusion_term = parameters.alpha * inner(grad(u), grad(phi)) * dx
    incompressibility = div(u) * psi * dx

    return convection + pressure_term + diffusion_term + incompressibility

# 5. asQ PROBLEM and SOLVER setup
# ---------------------------------
aaofunc = AllAtOnceFunction(ensemble, time_partition, Z)
aaofunc.initial_condition.assign(z0)

aaoform = AllAtOnceForm(aaofunc,
                        parameters.dt,
                        parameters.theta,
                        form_mass=form_mass,
                        form_function=form_function,
                        bcs=bcs_list[-1])

# Callback function to update the time constant for the BCs
def update_bcs_time(t):
    t_const.assign(t)

# The appctx dictionary passes necessary information to the solver,
# including callbacks and multigrid contexts.
app_context = {
    'mesh_hierarchy': mesh_hierarchy,
    'function_spaces': function_spaces,
    'bcs_list': bcs_list,
    'pre_function_callback': update_bcs_time,
    'pre_jacobian_callback': update_bcs_time,
}

solver_parameters = {
    'snes_type': 'ksponly',
    'mat_type': 'aij',
    'ksp_type': 'fgmres',
    "ksp_monitor_true_residual": None,
    "ksp_max_it": 100,
    "ksp_gmres_restart": 100,
    "ksp_atol": 1e-6,
    "ksp_rtol": 1e-6,
    'pc_type': 'python',
    'pc_python_type': 'CyclicReduction.asQMGPC',
    'asQMGPC_opts': {
        'ksp_type': 'chebyshev',
        'ksp_chebyshev_esteig': '0,0.25,0,1.05',
        'ksp_max_it': 2,
        'ksp_convergence_test': 'skip',
        'pc_type': 'python',
        'pc_python_type': 'CyclicReduction.CyclicReductionPC3',
        'cr_opts': {
            'patch_type': 'star',
            'construct_dim': 0,
            'exclude_subfunctions': "1",
            'mat_ordering_type': 'natural',
        }
    }
}


solver = AllAtOnceSolver(aaoform,
                         aaofunc,
                         solver_parameters=solver_parameters,
                         appctx=app_context)

# 6. SOLVE the problem
# --------------------
solver.solve()

PETSc.Sys.Print("Setup complete. The AllAtOnceSolver is ready to be used.")

# from firedrake import *
# from firedrake.petsc import PETSc
# from time import time
# from asQ import (
#     create_ensemble,
#     AllAtOnceFunction,
#     AllAtOnceForm,
#     AllAtOnceSolver,
# )
# import warnings
# warnings.simplefilter("ignore", FutureWarning)

# class ProblemParameters:
#     def __init__(self):
#         self.M = 4 # Number of spatial points
#         self.mref = 1
#         self.dt = 0.001
#         self.degree = {'space': 2,
#                        'time': 0} # DG degree 0 gives backward Euler
#         self.R = Constant(1)
#         self.alpha = Constant(1)
#         self.plot = False
#         self.theta = 1 # Theta parameter for the time-stepping scheme

# parameters = ProblemParameters()

# time_partition = [4,4]
# ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# # Define mesh
# distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
# base_mesh = UnitSquareMesh(nx = parameters.M, ny = parameters.M,
#                            distribution_parameters=distribution_parameters,comm = ensemble.comm)
# mesh_hierarchy = MeshHierarchy(base_mesh,parameters.mref)

# # Define function spaces
# function_spaces = []
# bcs_list = []
# for mesh in mesh_hierarchy:
#     # Append MG function spaces
#     space_element1 = FiniteElement("CG", triangle, parameters.degree['space']+1) # Velocity space
#     space_element2 = FiniteElement("CG", triangle, parameters.degree['space']) # Pressure space
#     Z = MixedFunctionSpace((VectorFunctionSpace(mesh,space_element1,dim=2),
#                            FunctionSpace(mesh,space_element2)))
#     function_spaces.append(Z)

#     # BC's for each level in the hierarchy
#     x, y = SpatialCoordinate(Z.mesh())
#     # PETSc.Sys.Print(f"x,y: {dir(x)}")
#     t = Constant(0.0)  # Do something here
#     ut = as_vector((-cos(pi*x)*sin(pi*y)*exp(-2*pi**2*t),
#                     sin(pi*x)*cos(pi*y)*exp(-2*pi**2*t)))
#     bcs = [DirichletBC(Z.sub(0),ut,"on_boundary")]
#     bcs_list.append(bcs)

# # Define initial condition on the finest mesh
# mesh = mesh_hierarchy[-1]
# Z = function_spaces[-1] 
# x, y = SpatialCoordinate(Z.mesh())

# z0 = Function(Z)
# u0, _ = z0.subfunctions
# u0 = interpolate(as_vector((-sin(pi*y)*cos(pi*x), cos(pi*y)*sin(pi*x))),Z.sub(0))

# bcs = bcs_list[-1]

# def form_mass(u, v):
#     return u*v*dx

# def form_function(u, p, phi, psi, t):
#     gradu = as_vector([u.dx(0),
#                        u.dx(1)])
    
#     gradphi = as_vector([phi.dx(0),
#                          phi.dx(1)])

#     F1 = parameters.R * inner(dot(u,gradu),phi) * dx \
#         + p * (phi[0].dx(0) + phi[1].dx(1)) * dx \
#         + parameters.alpha * inner(gradu,gradphi) * dx
    
#     F2 = (u[0].dx(0) + u[1].dx(1)) * psi * dx

#     return F1 + F2
# PETSc.Sys.Print(f"Function space Z: {Z.dim()}, u0 size: {u0.dat._vec.getSize()}, ownership ranges: {u0.dat._vec.getOwnershipRanges()}")


# aaofunc = AllAtOnceFunction(ensemble, time_partition, Z)
# aaofunc.initial_condition.assign(u0)

# aaoform = AllAtOnceForm(aaofunc, 
#                         parameters.dt, 
#                         parameters.theta, 
#                         form_mass,
#                         form_function, 
#                         bcs=bcs)

# #Set up solver parameters
# patch_parameters = {'patch_type': 'star',
#                     'construct_dim': 0, 
#                     'exclude_subfunctions': "1",
#                     'mat_ordering_type': 'natural',
#                    }

# mg_levels_parameters = {'ksp_type': 'chebyshev',
#                         'ksp_chebyshev_esteig': '0,0.25,0,1.05',
#                         'ksp_max_it': 2,
#                         'ksp_convergence_test': 'skip',
#                         'pc_type': 'python',
#                         'pc_python_type': 'CyclicReduction.CyclicReductionPC3',
#                         'cr_opts': patch_parameters
#                         }

# solver_parameters = {'snes_type': 'ksponly',
#                      'mat_type': 'aij',
#                      'ksp_type': 'fgmres',
#                      'ksp_monitor_true_residual': None,
#                      'ksp_max_it': 100,
#                      'ksp_gmres_restart': 100,
#                      'ksp_atol': 1e-6,
#                      'ksp_rtol': 1e-6,
#                      'pc_type': 'python',
#                      'pc_python_type': 'CyclicReduction.asQMGPC',
#                      'asQMGPC_opts': mg_levels_parameters
#                     }

# solver = AllAtOnceSolver(aaoform, 
#                          aaofunc, 
#                          solver_parameters,
#                          appctx={'mesh_hierarchy': mesh_hierarchy, 'function_spaces': function_spaces, 'bcs_list': bcs_list})

# Some useful prints
processes = COMM_WORLD.size
PETSc.Sys.Print(f"Running with {processes} MPI processes. {len(time_partition)} in time, each with {processes//len(time_partition)} in space")
PETSc.Sys.Print(f"Time partition: {time_partition}, total time steps: {sum(time_partition)}")

A,_ = solver.snes.ksp.getOperators()
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")

# J = mat_from_mesh()
# PETSc.Sys.Print(f"Size J: {J.getSize()}, with ownership ranges: {J.getOwnershipRanges()}")

space_dofs = U.dim()
time_dofs = sum(time_partition)
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's space: {space_dofs}, DOF's time: {time_dofs}, DOF's total: {total_dofs} \n")

start = time()
solver.solve()
PETSc.Sys.Print(f"Finished solve in {time()-start}s")

save_to_VTK = True
# --------------------------------------------------
# Output results to ParaView
# --------------------------------------------------
if save_to_VTK:
    import shutil
    import os
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
            w_local = Function(U)
            w_local.assign(aaofunc[local_step])
            w_local.rename("u")
            if ensemble.ensemble_comm.rank != 0:
                ensemble.send(w_local, dest=0, tag=step)
            else:
                w_recv = w_local.copy(deepcopy=True)  # if rank 0 owns it, just copy

        if ensemble.ensemble_comm.rank == 0:
            if not aaoform.layout.is_local(step):
                w_recv = Function(U, name="u")
                ensemble.recv(w_recv, source=MPI.ANY_SOURCE, tag=step)

            vtkfile.write(w_recv, time=step * parameters.dt)
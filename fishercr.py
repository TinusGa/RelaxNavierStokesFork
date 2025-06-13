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
        self.M = 4 # Number of elements in the base mesh
        self.mref = 1 # Number of mesh refinement levels for multigrid
        self.dt = 0.025
        self.R = Constant(1) # Reynolds number
        self.alpha = Constant(1) # Diffusion constant
        self.theta = 1.0 # Time-stepping parameter (1.0 for Backward Euler)
        self.space_degree = 2

parameters = ProblemParameters()

# 1. ENSEMBLE and MESH setup
# ----------------------------------------------------
time_partition = [4,4,4,4]
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base_mesh = PeriodicUnitSquareMesh(nx=parameters.M, ny=parameters.M, direction='both',
                                   distribution_parameters=distribution_parameters,
                                   comm=ensemble.comm)
mesh_hierarchy = MeshHierarchy(base_mesh, parameters.mref)

# 2. FUNCTION SPACES and BOUNDARY CONDITIONS
# ---------------------------------------------
function_spaces = []
bcs_list = []
for mesh in mesh_hierarchy:
    space_element = FiniteElement("CG", triangle, parameters.space_degree)
    U = FunctionSpace(mesh, space_element)
    function_spaces.append(U)
    bcs = []
    bcs_list.append(bcs)

# 3. INITIAL CONDITION
# ----------------------
U = function_spaces[-1]
u0 = Function(U)
x, y = SpatialCoordinate(U.mesh())
# g = exp(-((x-0.5)**2 + (y-0.5)**2)/0.01)
# u0.interpolate(g)
u0.interpolate(sin(2*pi*(x-1))*sin(2*pi*(y-1)))

# 4. VARIATIONAL FORMS
# ----------------------------------
D = Constant(0.01)  # Diffusion coefficient
r = Constant(1.0)  # Growth rate

def form_mass(u, v):
    return inner(u, v) * dx

def form_function(u, v, t):
    diffusion_term = D * inner(grad(u), grad(v)) * dx 
    reaction_term = - r * u * (1 - u) * v * dx
    return diffusion_term + reaction_term

# 5. asQ PROBLEM and SOLVER setup
# ---------------------------------
aaofunc = AllAtOnceFunction(ensemble, time_partition, U)
aaofunc.initial_condition.assign(u0)

aaoform = AllAtOnceForm(aaofunc,
                        parameters.dt,
                        parameters.theta,
                        form_mass=form_mass,
                        form_function=form_function,
                        bcs=bcs_list[-1])

# The appctx dictionary passes necessary information to the solver,
# including callbacks and multigrid contexts.
app_context = {
    'mesh_hierarchy': mesh_hierarchy,
    'function_spaces': function_spaces,
    'bcs_list': bcs_list,
}

solver_parameters = {
    'snes_type': 'newtonls',
    'snes_ksp_ew': None,
    'snes_monitor': None,
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
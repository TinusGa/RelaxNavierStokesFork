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

class ProblemParameters:
    def __init__(self):
        self.Nslice = 4
        self.Pt = 4
        self.time_partition = [self.Nslice] * self.Pt
        self.M = 9 # Number of elements in the base mesh
        self.mref = 1 # Number of mesh refinement levels for multigrid
        self.dt = 0.001
        self.R = Constant(1) # Reynolds number
        self.alpha = Constant(1) # Diffusion constant
        self.theta = 1.0 # Time-stepping parameter (1.0 for Backward Euler)
        self.space_degree = 2
        self.plot = True  # Whether to output results to ParaView
        self.write_metrics = True  # Whether to write metrics to file

parameters = ProblemParameters()

# def ReductionHeat(parameters = ProblemParameters()):

# 1. ENSEMBLE and MESH setup
# ----------------------------------------------------
start = time()
# time_partition = [120]
# time_partition = [60]*2
# time_partition = [30]*4
# time_partition = [15]*8
# time_partition = [5]*24

time_partition = [4,4,4,4]

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
for mesh in mesh_hierarchy:
    space_element = FiniteElement("CG", triangle, parameters.space_degree)
    U = FunctionSpace(mesh, space_element)
    function_spaces.append(U)

    bcs = []
    bcs_list.append(bcs)

# 3. INITIAL CONDITION
# ----------------------
U = function_spaces[-1]
# u0 = Function(U)
# x, y = SpatialCoordinate(U.mesh())
# # g = exp(-((x-0.5)**2 + (y-0.5)**2)/0.01)
# # u0.interpolate(g)
# u0.interpolate(sin(1*pi*(x-1))*sin(10*pi*(y-1)))

# # Define function spaces
# space_element = FiniteElement("CG", triangle, parameters.degree['space'])
# U = FunctionSpace(mesh,space_element)

# I.C and B.C
x, y = SpatialCoordinate(U.mesh())
u0 = Function(U)

# u0.interpolate(cos(pi*x)*cos(2*pi*y))
# bcs = []

u0.project(sin(0.25*pi*x)*cos(2*pi*y))
bcs = [DirichletBC(U, 0, sub_domain=1)]

# Define forms
def form_mass(u, v):
    return u*v*dx

def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx

PETSc.Sys.Print("Setting up AllAtOnceFunction...")
aaofunc = AllAtOnceFunction(ensemble, time_partition, U)
aaofunc.assign(u0)
PETSc.Sys.Print("AllAtOnceFunction setup complete.")

PETSc.Sys.Print("Setting up AllAtOnceForm...")
aaoform = AllAtOnceForm(aaofunc, 
                        parameters.dt, 
                        parameters.theta, 
                        form_mass,
                        form_function, 
                        bcs=bcs)
PETSc.Sys.Print("AllAtOnceForm setup complete.")

appctx = {
    'mesh_hierarchy': mesh_hierarchy,
    'function_spaces': function_spaces,
    'bcs_list': bcs_list,
    'time1': 0.0,  # Time to setup before reduction
    'time2': 0.0,  # Time to apply reduction
    'time3': 0.0,  # Time to fill solution
}

#Set up solver parameters. 
# solver_parameters = {
#     'snes_type': 'ksponly',
#     'mat_type': 'aij',
#     'ksp_type': 'fgmres',
#     "ksp_monitor_true_residual": None,
#     "ksp_max_it": 100,
#     "ksp_gmres_restart": 100,
#     "ksp_atol": 1e-8,
#     "ksp_rtol": 1e-8,
#     'pc_type': 'python',
#     'pc_python_type': 'CyclicReduction.asQMGPC',
#     'asQMGPC_opts': {
#         'ksp_type': 'chebyshev',
#         'ksp_chebyshev_esteig': '0,0.25,0,1.05',
#         'ksp_max_it': 2,
#         'ksp_initial_guess_nonzero': True,
#         'pc_type': 'python',
#         'pc_python_type': 'CyclicReduction.CyclicReductionPC3',
#     }
# }

solver_parameters = {
    'snes_type': 'ksponly',
    'mat_type': 'aij',
    'ksp_type': 'fgmres',
    "ksp_monitor_true_residual": None,
    "ksp_max_it": 100,
    "ksp_gmres_restart": 100,
    "ksp_atol": 1e-05,
    "ksp_rtol": 1e-05,
    'pc_type': 'python',
    'pc_python_type': 'CyclicReduction.CyclicReductionPC3',}

PETSc.Sys.Print("Setting up AllAtOnceSolver...")
# solver_parameters = {'snes_type': 'ksponly',
#                     'mat_type': 'aij',
#                     'ksp_type': 'fgmres',
#                     "ksp_monitor_true_residual": None,
#                     "ksp_max_it": 100,
#                     "ksp_atol": 1e-12,
#                     "ksp_rtol": 1e-12,
#                     'pc_type': 'python',
#                     'pc_python_type': 'firedrake.ASMStarPC',
#                     'pc_star_construct_dim': 0,
#                     'pc_star_sub_sub_pc_type': 'lu',
#                     'pc_star_sub_sub_pc_factor_mat_solver_type': 'umfpack',}
# solver = LinearSolver(aaoform,  
#                       solver_parameters,)
solver = AllAtOnceSolver(aaoform, 
                         aaofunc, 
                         solver_parameters,
                         appctx=appctx,)
PETSc.Sys.Print("AllAtOnceSolver setup complete.")
setup_time = time()-start
PETSc.Sys.Print(f"Setup time {int(setup_time//60)}m {int(setup_time%60)}s:")

# Some useful prints
processes = COMM_WORLD.size
PETSc.Sys.Print(f"Running with {processes} MPI processes. {len(time_partition)} in time, each with {processes//len(time_partition)} in space")
PETSc.Sys.Print(f"Time partition: {time_partition}, total time steps: {sum(time_partition)}")

A,_ = solver.snes.ksp.getOperators()
iterations = solver.snes.getLinearSolveIterations()
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")

space_dofs = U.dim()
time_dofs = sum(time_partition)
total_dofs = time_dofs*space_dofs
PETSc.Sys.Print(f"DOF's space: {space_dofs}, DOF's time: {time_dofs}, DOF's total: {total_dofs} \n")

start = time()
solver.solve()
solve_time = time() - start
PETSc.Sys.Print(f"Finished solve in {solve_time}s")

# PETSc.Sys.Print(f"Memory address of aaofunc {id(aaofunc)}")
# PETSc.Sys.Print(f"In apply_impl. Time to setup before reduction: {aaofunc.time1}. Time to apply reduction: {aaofunc.time2}. Time to fill solution: {aaofunc.time3}. Total time: {aaofunc.time1 + aaofunc.time2 + aaofunc.time3}")
if COMM_WORLD.rank == 0:
    if parameters.write_metrics:
        filename = "heatcr.csv"
        header = ["NT", "Nt", "Nx", "P", "Pt", "Px","mref", "Setup Time (s)", "Solve Time (s)", "Iterations","PC pre-reduction time (s)", "PC communication time (s)", "PC fill time (s)"]
        
        with open(filename, "a", encoding="utf-8") as file:
            writer = csv.writer(file)

            
            new_row = [total_dofs, time_dofs, space_dofs, processes, len(time_partition), processes//len(time_partition), parameters.mref, setup_time, solve_time,iterations, appctx['time1'], appctx['time2'], appctx['time3']]
            writer.writerow(new_row)


# --------------------------------------------------
# Output results to ParaView
# --------------------------------------------------
if parameters.plot:
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
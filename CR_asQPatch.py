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
        self.N = 3 # Number of time steps
        self.dt = 0.001 # Specified instead of end time
        self.M = 9 # Number of spatial points
        self.Mbase = 4 # Number of spatial points in base mesh
        self.Mref = 2 # Number of refinements in the mesh hierarchy
        self.degree = {'space': 2,
                       'time': 0} # DG degree 0 gives backward Euler
        self.plot = False
        self.solver = None
        self.Pt = 1 # Processors in time
        self.theta = 1 # Theta parameter for the time-stepping scheme

parameters = ProblemParameters()

# Create a time partition and an ensemble communicator
time_partition = [4,4,4,4]
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Define mesh
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
base_mesh = UnitSquareMesh(nx = parameters.Mbase, ny = parameters.Mbase,
                           distribution_parameters=distribution_parameters,comm = ensemble.comm)
mesh_hierarchy = MeshHierarchy(base_mesh,parameters.Mref)


mesh = mesh_hierarchy[-1] # This is the finest mesh

# Define function spaces
function_spaces = []
bcs_list = []
for mesh in mesh_hierarchy:
    space_element = FiniteElement("CG", triangle, parameters.degree['space'])
    U = FunctionSpace(mesh,space_element)
    function_spaces.append(U)

    # BC 1
    bcs = []

    # BC 2
    # bcs = [DirichletBC(U, 0, sub_domain=1)]
    bcs_list.append(bcs)


# Define initial condition
U = function_spaces[-1] 
x, y = SpatialCoordinate(U.mesh())
u0 = Function(U)

# Two different IC's and BC's to test

# IC to BC 1
u0.project(cos(pi*x)*cos(2*pi*y))

# IC to BC 2
# u0.project(sin(0.25*pi*x)*cos(2*pi*y))

bcs = bcs_list[-1]

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

#Set up solver parameters
patch_parameters = {'patch_type': 'star',
                 'construct_dim': 0, 
                 'mat_ordering_type': 'natural',
                }

mg_levels_parameters = {'ksp_type': 'chebyshev',
                        'ksp_chebyshev_esteig': '0,0.25,0,1.05',
                        'ksp_max_it': 2,
                        'ksp_convergence_test': 'skip',
                        'pc_type': 'python',
                        'pc_python_type': 'CyclicReduction.CyclicReductionPC4',
                        'cr_opts': patch_parameters
                        }

solver_parameters = {'snes_type': 'ksponly',
                     'mat_type': 'aij',
                     'ksp_type': 'fgmres',
                     'ksp_monitor_true_residual': None,
                     'ksp_max_it': 100,
                     'ksp_gmres_restart': 100,
                     'ksp_atol': 1e-6,
                     'ksp_rtol': 1e-6,
                     'pc_type': 'python',
                     'pc_python_type': 'CyclicReduction.asQMGPC2',
                     'asQMGPC_opts': mg_levels_parameters
                    }

solver = AllAtOnceSolver(aaoform, 
                         aaofunc, 
                         solver_parameters,
                         appctx={'mesh_hierarchy': mesh_hierarchy, 'function_spaces': function_spaces, 'bcs_list': bcs_list})

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
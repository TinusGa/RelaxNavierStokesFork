from firedrake import *
from firedrake.preconditioners import ASMStarPC
from mpi4py import MPI

# Parameters
nx = 9                     # spatial resolution
nt = 1                     # number of time steps
Lx = 1.0                    # spatial domain length
T = 1.0                     # final time

# Mesh in 1D space
base_mesh = IntervalMesh(nx, Lx)

# Extrude in time: layers=nt, height=T
mesh = ExtrudedMesh(base_mesh, layers=nt, layer_height=T/nt)

# Define function space on the extruded mesh
degree = 2
# V = FunctionSpace(mesh, FiniteElement("CG", interval, degree))

# If you want tensor product structure (space x time), use this:
V = FunctionSpace(mesh, TensorProductElement(FiniteElement("CG", interval, degree),
                                              FiniteElement("DG", interval, 0)))

# Define initial condition (only in space, apply to bottom layer)
x,t = SpatialCoordinate(mesh)
u0 = Function(V)
u0.interpolate(18*x)  # Note: Only spatially varying

u0.dat._vec.view()



# Define forms (e.g., mass and stiffness)
u = TrialFunction(V)
v = TestFunction(V)

mass_form = u*v*dx
stiffness_form = inner(grad(u), grad(v))*dx

# Apply BCs if needed
bcs = []

# Assembly example
M = assemble(mass_form).petscmat
K = assemble(stiffness_form).petscmat

star = ASMStarPC()
star.prefix = 'asm_star_'
patches = star.get_patches(V)
lgmap = V.dof_dset.lgmap

lgmap = V.dof_dset.lgmap
# Translate to global numbers
ises = tuple(lgmap.applyIS(iset) for iset in patches)

# set_size = V.node_set.size
# PETSc.Sys.Print(f"Rank: {COMM_WORLD.rank}. V dim : {V.dim()}, node set size: {set_size}", comm=COMM_SELF)
# PETSc.Sys.Print(f"V... {V.dof_dset.field_ises[0].indices}", comm=COMM_SELF)
# COMM_WORLD.Barrier()
# PETSc.Sys.Print(f"\n")
# if COMM_WORLD.rank == 1:
#     for iset in patches:
#         indices = lgmap.apply(iset.indices)  # Convert local indices to global indices
#         # indices = iset.indices # Already in global numbering
#         PETSc.Sys.Print(f"Rank: {COMM_WORLD.rank}. Patch indices: {indices}", comm=COMM_SELF)
for iset in ises:
    indices = iset.indices  # Already in global numbering
    PETSc.Sys.Print(f"Rank: {COMM_WORLD.rank}. Patch indices: {indices}", comm=COMM_SELF)


# Compute and stash patch statistics
mpi_comm = COMM_WORLD
max_local_patch = max(is_.getSize() for is_ in ises)
min_local_patch = min(is_.getSize() for is_ in ises)
sum_local_patch = sum(is_.getSize() for is_ in ises)
max_global_patch = mpi_comm.allreduce(max_local_patch, op=MPI.MAX)
min_global_patch = mpi_comm.allreduce(min_local_patch, op=MPI.MIN)
sum_global_patch = mpi_comm.allreduce(sum_local_patch, op=MPI.SUM)
avg_global_patch = sum_global_patch / mpi_comm.allreduce(len(ises) if sum_local_patch > 0 else 0, op=MPI.SUM)
msg = f"Minimum / average / maximum patch sizes : {min_global_patch} / {avg_global_patch} / {max_global_patch}\n"

# PETSc.Sys.Print(msg, comm=COMM_SELF)

PETSc.Sys.Print(f"Type M: {type(M)}, Type K: {type(K)}, ownership ranges: {M.getOwnershipRanges()}", comm=COMM_SELF)

print(type(ises))
print(type(ises[1:]), "patches created")
lower_ises = ises[1:]  # Exclude the first patch which is the whole mesh
submats_diag = M.createSubMatrices(ises,ises) # List of mats 
submats_sub = M.createSubMatrices(ises[1:],ises[:-1]) # List of mats
submats_sub = [0] + submats_sub
for sub,diag in zip(submats_sub, submats_diag):
    if sub == 0:
        PETSc.Sys.Print(f"Rank: {COMM_WORLD.rank}. Submat shape: {(0,0)}, Diag shape: {diag.getSize()}", comm=COMM_SELF)
    else:
        PETSc.Sys.Print(f"Rank: {COMM_WORLD.rank}. Submat shape: {sub.getSize()}, Diag shape: {diag.getSize()}", comm=COMM_SELF)


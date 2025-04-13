from firedrake import *
from asQ import (
    AllAtOnceFunction,
    create_ensemble,
    AllAtOnceForm,
    AllAtOnceSolver,
)
from asQ.pencil import Pencil, Subcomm
from CyclicReduction.check_setup import create_time_partition
import numpy as np
import warnings
warnings.simplefilter("ignore", FutureWarning)

n_timesteps = 9
temporal_processors = 4
nx = 4
ny = 4
degree_space = 1
theta = 1

time_partition = create_time_partition(n_timesteps-1, temporal_processors)
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Create a mesh with nx+1 and ny+1 vertices
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
mesh = UnitSquareMesh(nx = nx, ny = ny, distribution_parameters = distribution_parameters, comm = ensemble.comm)
n = FacetNormal(mesh)
V = FunctionSpace(mesh, "CG", degree_space)
x, y = SpatialCoordinate(V.mesh())
u0 = Function(V)
u0.project(cos(pi*x)*cos(2*pi*y))
bcs = []

def form_mass(u, v):
    return u*v*dx

def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx

aaofunc = AllAtOnceFunction(ensemble, time_partition, V)
aaofunc.initial_condition.assign(u0)
blockV = aaofunc.field_function_space 

srank = ensemble.comm.rank
trank = ensemble.ensemble_comm.rank

subcomm = Subcomm(ensemble.ensemble_comm, [0, 1])

# dimensions of space-time data in this ensemble_comm
nlocal = blockV.node_set.size
#PETSc.Sys.Print(f"nlocal for {trank,srank}: {nlocal}", comm=COMM_SELF)
NN = np.array([n_timesteps, nlocal], dtype=int)

# transfer pencil is aligned along axis 1
p0 = Pencil(subcomm, NN, axis=1)

# a0 is the local part of our fft working array
# has shape of (partition/P, nlocal)
PETSc.Sys.Print(f"p0 subshape for {trank,srank}: {p0.subshape}", comm=COMM_SELF)

PETSc.Sys.Print(f"")
a0 = np.zeros(p0.subshape,dtype=np.float64)
p1 = p0.pencil(0)
COMM_WORLD.Barrier()
# a0 is the local part of our other fft working array
PETSc.Sys.Print(f"p1 subshape for {trank,srank}: {p1.subshape}", comm=COMM_SELF)
a1 = np.zeros(p1.subshape,dtype=np.float64)
transfer = p0.transfer(p1,dtype=np.float64)

# with x.global_vec_ro() as xvec:
#     # PETSc.Sys.Print(f"Ownership range of xvec: {xvec.getOwnershipRange()}",comm=fd.COMM_SELF)
#     parray = xvec.array_r.reshape((self.nlocal_timesteps,
#                                     self.blockV.node_set.size))
#     # PETSc.Sys.Print(f"parray_shape: {parray.shape}",comm=fd.COMM_SELF)
# # This produces an array whose rows are time slices
# # and columns are finite element basis coefficients

# ######################
# # Diagonalise - scale, transfer, FFT, transfer, Copy
# # Scale
# # is there a better way to do this with broadcasting?
# parray = (1.0+0.j)*(self.Gam_slice*parray.T).T*np.sqrt(self.ntimesteps)
# # transfer forward
# self.a0[:] = parray[:]
# with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.transfer"):
#     self.transfer.forward(self.a0, self.a1)

# # FFT
# with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.fft"):
#     self.a1[:] = fft(self.a1, axis=0)

# # transfer backward
# with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.transfer"):
#     self.transfer.backward(self.a1, self.a0)

# # Copy into xfi, xfr
# parray[:] = self.a0[:]
# with self.xfr.function.dat.vec_wo as v:
#     v.array[:] = parray.real.reshape(-1)
# with self.xfi.function.dat.vec_wo as v:
#     v.array[:] = parray.imag.reshape(-1)
# #####################

# # Do the block solves

# with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.block_solves"):
#     for i in range(self.nlocal_timesteps):
#         # copy the data into solver input
#         cpx.set_real(self.xtemp, self.xfr[i])
#         cpx.set_imag(self.xtemp, self.xfi[i])

#         for cdat, xdat in zip(self.block_rhs.dat, self.xtemp.dat):
#             cdat.data[:] = xdat.data[:]

#         # solve the block system
#         self.block_sol.zero()
#         self.block_solvers[i].solve()

#         # copy the data from solver output
#         cpx.get_real(self.block_sol, self.xfr[i])
#         cpx.get_imag(self.block_sol, self.xfi[i])



# ######################
# # Undiagonalise - Copy, transfer, IFFT, transfer, scale, copy
# # get array of basis coefficients
# with self.xfi.function.dat.vec_ro as v:
#     parray = 1j*v.array_r.reshape((self.nlocal_timesteps,
#                                     self.blockV.node_set.size))
# with self.xfr.function.dat.vec_ro as v:
#     parray += v.array_r.reshape((self.nlocal_timesteps,
#                                     self.blockV.node_set.size))
# # transfer forward
# self.a0[:] = parray[:]
# with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.transfer"):
#     self.transfer.forward(self.a0, self.a1)

# # IFFT
# with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.fft"):
#     self.a1[:] = ifft(self.a1, axis=0)

# # transfer backward
# with PETSc.Log.Event("asQ.diag_preconditioner.CirculantPC.apply.transfer"):
#     self.transfer.backward(self.a1, self.a0)
# parray[:] = self.a0[:]

# # scale
# parray = ((1.0/self.Gam_slice)*parray.T).T
# # Copy into xfi, xfr

# with y.global_vec_wo() as yvec:
#     self.spatial_rank = self.ensemble.comm.rank
#     self.temporal_rank = self.ensemble.ensemble_comm.rank
#     #PETSc.Sys.Print(f"yvec ownership : {yvec.getOwnershipRange()}. parray shape : {parray.reshape(-1).real.shape}. Temporal rank: {self.temporal_rank}, Spatial rank: {self.spatial_rank}\n", comm=fd.COMM_SELF)
#     yvec.array[:] = parray.reshape(-1).real
# ################
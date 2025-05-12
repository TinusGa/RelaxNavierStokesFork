import faulthandler; faulthandler.enable()
import numpy as np
import os
from asQ.parallel_arrays import DistributedDataLayout1D

from firedrake import *
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
    LinearSolver,
    SharedArray,
)
from asQ.pencil import Pencil, Subcomm
from CyclicReduction.check_setup import check_setup, create_time_partition
import time
import warnings
warnings.simplefilter("ignore", FutureWarning)


class CRsolver():
    def __init__(self, time_partition, ensemble, form_mass, form_function, aaofunc, aaoform, dt):
        self.layout = DistributedDataLayout1D(time_partition, ensemble.ensemble_comm)
        self.ensemble = ensemble
        self.time_partition = self.layout.partition
        self.time_rank = ensemble.ensemble_comm.rank
        self.nlocal_timesteps = self.layout.local_size
        self.ntimesteps = self.layout.global_size
        
        self.time = tuple(fd.Constant(0) for _ in range(self.nlocal_timesteps))
        # for n in range((self.aaofunc.nlocal_timesteps)):
        #     self.time[n].assign(self.t0 + self.dt*(self.aaofunc.transform_index(n, from_range='slice', to_range='window') + 1))

        self.dt = dt

        self.form_mass = form_mass
        self.form_function = form_function
        self.aaofunc = aaofunc
        self.aaoform = aaoform
        

        # All-at-once reference state
        self.state_func = self.aaofunc.copy()
        self.field_function_space = self.aaofunc.field_function_space

        # This processor's spatial and temporal rank
        self.spatial_rank = self.ensemble.comm.rank
        self.temporal_rank = self.ensemble.ensemble_comm.rank

        # Zero out bc dofs
        self.block_bcs = tuple(
            fd.DirichletBC(self.field_function_space,
                            0*bc.function_arg,
                            bc.sub_domain)
            for bc in self.aaoform.field_bcs)
                
        self.diag_matrices = []
        self.lower_diag_matrices = []
        self.rhs = []

        dt1 = fd.Constant(1/self.dt)
        theta = fd.Constant(1)

        for i in range(self.nlocal_timesteps):
            # The reference states
            u0 = self.state_func[i]
            t = self.time[i]

            v = fd.TestFunction(self.field_function_space)
            u = fd.TrialFunction(self.field_function_space)

            M = self.form_mass(u, v)
            K = self.form_function(u, v, t)

            # Represents the linear system for timestep/row i. That is L*u[i] + D*u[i+1] = f[i+1]
            D = fd.assemble(dt1*M + theta*K, bcs=self.block_bcs).petscmat # Main diagonal block system
            L = fd.assemble(-1*dt1*M).petscmat # Lower/off - diagonal block system

            if self.temporal_rank == 0 and i == 0:
                RHS = (1/self.dt) * self.form_mass(u0, v)
                f = fd.assemble(RHS, bcs=self.block_bcs)
                f = f.dat._vec # f[1] = L*u[0]
            else:
                f = fd.Function(self.field_function_space).zero()
                f = f.dat._vec

            self.diag_matrices.append(D)
            self.lower_diag_matrices.append(L)
            self.rhs.append(f)

    def apply_impl(self, y):

        y.zero()
        
        # Define the pencil for the current rank for timestep ordering
        # p0 : Pencil describing spatial DOF distribution per timestep. E.g. If spatial rank 0, temporal rank 0
        # owns 3 timesteps of the global system, and owns 10 spatial DOFs in each then p0.subshape = (3,10)
        # This let's us use a0 to write to the global solution vector y 
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        nlocal = self.field_function_space.node_set.size # Spatial DOFs for this rank
        NN = np.array([self.ntimesteps, nlocal], dtype=int)
        self.p0 = Pencil(subcomm, NN, axis=1)
        self.a0 = np.zeros(self.p0.subshape,dtype=np.float64)

        # Temporal rank 0 owns the first timestep of the global system.
        # This is the first block of the system, which is the first diagonal matrix
        # and the first rhs vector.
        if self.temporal_rank == 0:
            F = self.get_factored_matrix(self.diag_matrices[0].copy(), self.ensemble.comm)
            u1 = self.rhs[0].duplicate()
            F.solve(self.rhs[0], u1)

            local_u1 = u1.getArray()
            self.a0[0,:] = local_u1[:]
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
        
        if self.temporal_rank > 0:
            # Receive u_prev from previous temporal rank. Sends and recieves must be done using Firedrake functions
            u_prev_function = fd.Function(self.field_function_space)
            source = self.temporal_rank - 1
            self.ensemble.recv(u_prev_function, source=source, tag=88)
            # Convert u_prev_function to a PETSc Vec
            with u_prev_function.dat.vec as v:
                u_prev = v.copy()
        else:
            # If this is the first temporal rank, we use u1 as u_prev
            u_prev = u1.copy() 

        offset = 1 if self.temporal_rank == 0 else 0
        
        for i in range(offset, self.nlocal_timesteps):
            L, D, f = self.diag_matrices[offset].copy(), lower_diag[offset].copy(), rhs[offset].copy()
            
            L_next = main_diag[i].copy()
            D_next = lower_diag[i].copy()
            f_next = rhs[i].copy()

            D_factored = self.get_factored_matrix(D, self.ensemble.comm)

            # Compute L <- L_next * D^{-1} * L, D <- -D_next and f <- L_next * D^{-1} * f - f_next
            L_dense = L.copy()
            L_dense.convert('dense')
            D_inv_L_dense = PETSc.Mat().createDense(size=L_dense.getSizes(), comm=self.ensemble.comm)
            D_inv_L_dense.setUp()
            D_inv_L_dense.assemble()

            # Compute L <- L_next * D^{-1} * L
            D_factored.matSolve(L_dense, D_inv_L_dense) # D^{-1} * L and stores in D_inv_L_dense
            L.convert('dense') # Convert L to dense matrix
            L.setUp()
            L.assemble()
            L_next.matMult(D_inv_L_dense, L) # L <- L_next * D^{-1} * L

            # Compute D <- - D_next
            D_next.scale(-1.0) # D_next <- -D_next
            D = D_next.copy() # D <- D_next

            # Compute f <- L_next * D^{-1} * f - f_next
            D_inv_f = f.duplicate()
            D_factored.solve(f, D_inv_f) # D^{-1} * f
            f_next.scale(-1.0) # f_next <- -f_next
            L_next.multAdd(D_inv_f, f_next, f) # f <- L_next * D^{-1} * f - f_next

            # Destruction
            D_factored.destroy()
            L_dense.destroy()
            D_inv_L_dense.destroy()
            D_inv_f.destroy()
            L_next.destroy()
            D_next.destroy()
            f_next.destroy()


        # ---------------------------------------------------------------------------
        # FORWARD REDUCTION
        # ---------------------------------------------------------------------------
        if COMM_WORLD.size > 1:
            L, D, f = self.forward_reduction(self.diag_matrices,
                                            self.lower_diag_matrices,
                                            self.rhs) 
        
            # y._vec.view()

            # ---------------------------------------------------------------------------
            # INTERFACE SOLVE (processor communication)
            # ---------------------------------------------------------------------------

            # Total number of temporal ranks
            n_temporal = self.ensemble.ensemble_comm.size

            # Cant we send/recieve from y instead?

            # Allocate buffers
            if self.temporal_rank > 0:
                # Receive u_prev from previous temporal rank. Sends and recieves must be done using Firedrake functions
                u_prev_function = fd.Function(self.field_function_space)
                source = self.temporal_rank - 1
                self.ensemble.recv(u_prev_function, source=source, tag=88)
                # Convert u_prev_function to a PETSc Vec
                with u_prev_function.dat.vec as v:
                    u_prev = v.copy()
            else:
                # If this is the first temporal rank, we use u1 as u_prev
                u_prev = u1.copy() 

            # Solve: u_next = D^{-1} (f - L * u_prev)
            rhs = f.duplicate()
            f.scale(-1.0) # Set f <- -f
            L.multAdd(u_prev, f, rhs)  # rhs <- L * u_prev - f
            rhs.scale(-1.0) # rhs <- f - L * u_prev

            u_next = rhs.duplicate()
            F = self.get_factored_matrix(D.copy(), self.ensemble.comm)
            F.solve(rhs, u_next)

            # PETSc.Sys.Print(f"u_next from rank {self.temporal_rank}:\n", u_next.getArray(),comm=fd.COMM_SELF)

            # Send to next temporal rank
            if self.temporal_rank < n_temporal - 1:
                dest = self.temporal_rank + 1
                # Convert u_next to a Firedrake Function
                u_next_function = fd.Function(self.field_function_space)
                with u_next_function.dat.vec as v:
                    u_next.copy(v) # Copies data from u_next to v
                self.ensemble.send(u_next_function, dest=dest, tag=88)
            
            # All ranks now own a u_prev and u_next. Most importantly, u_prev for each processor can be used 
            # to solve for all its owning rows of the global system.
            # PETSc.Sys.Print(f"u_prev from rank {self.temporal_rank}:\n", u_prev.getArray(),comm=fd.COMM_SELF)
            # PETSc.Sys.Print(f"u_next from rank {self.temporal_rank}:\n", u_next.getArray(),comm=fd.COMM_SELF)
        else:
            u_prev = u1.copy()
        
        # ---------------------------------------------------------------------------
        # BACKSUBSTITUTION (also writes to the global solution vector y)
        # ---------------------------------------------------------------------------
        self.forward_substitution(self.diag_matrices, self.lower_diag_matrices, self.rhs, u_prev, y)

        return y
        
    def forward_reduction(self, main_diag, lower_diag, rhs):
        """
        Perform the forward reduction step of the cyclic reduction algorithm.

        Expects main_diag, lower_diag and rhs to be lists of matrices/vectors
        representing the diagonal, lower diagonal and right-hand side of the
        system of equations respectively. Entries are expected to be in PETSc.Mat
        or PETSc.Vec format respectively.

        Notes
        -----
        - Input lists or it's entries should never be changed in any way during
          the reduction process.
        """

        # Make sure the input lists are of equal length
        if len(main_diag) != len(lower_diag) or len(main_diag) != len(rhs):
            raise ValueError("Input lists must be of equal length.")
        
        # Temporal rank 0 is offset from other ranks by 1
        offset = 1 if self.temporal_rank == 0 else 0
        
        # Reduce onto these variables
        L, D, f = main_diag[offset].copy(), lower_diag[offset].copy(), rhs[offset].copy()

        if len(main_diag) > 1: # this processor owns more than one timestep, so we reduce
            for i in range(offset + 1, self.nlocal_timesteps):
            
                L_next = main_diag[i].copy()
                D_next = lower_diag[i].copy()
                f_next = rhs[i].copy()

                D_factored = self.get_factored_matrix(D, self.ensemble.comm)

                # Compute L <- L_next * D^{-1} * L, D <- -D_next and f <- L_next * D^{-1} * f - f_next
                L_dense = L.copy()
                L_dense.convert('dense')
                D_inv_L_dense = PETSc.Mat().createDense(size=L_dense.getSizes(), comm=self.ensemble.comm)
                D_inv_L_dense.setUp()
                D_inv_L_dense.assemble()

                # Compute L <- L_next * D^{-1} * L
                D_factored.matSolve(L_dense, D_inv_L_dense) # D^{-1} * L and stores in D_inv_L_dense
                L.convert('dense') # Convert L to dense matrix
                L.setUp()
                L.assemble()
                L_next.matMult(D_inv_L_dense, L) # L <- L_next * D^{-1} * L

                # Compute D <- - D_next
                D_next.scale(-1.0) # D_next <- -D_next
                D = D_next.copy() # D <- D_next

                # Compute f <- L_next * D^{-1} * f - f_next
                D_inv_f = f.duplicate()
                D_factored.solve(f, D_inv_f) # D^{-1} * f
                f_next.scale(-1.0) # f_next <- -f_next
                L_next.multAdd(D_inv_f, f_next, f) # f <- L_next * D^{-1} * f - f_next

                # Destruction
                D_factored.destroy()
                L_dense.destroy()
                D_inv_L_dense.destroy()
                D_inv_f.destroy()

                L_next.destroy()
                D_next.destroy()
                f_next.destroy()
                
        return L, D, f

    def forward_substitution(self, main_diag, lower_diag, rhs, u_prev, y):
        """
        Perform the back substitution step of the cyclic reduction algorithm.
        """
        offset = 1 if self.temporal_rank == 0 else 0 # Since temporal rank 0 is offset from other ranks by 1. It has 1 more row than other ranks


        for i in range(offset, self.nlocal_timesteps):

            L = main_diag[i].copy()
            D = lower_diag[i].copy()
            f = rhs[i].copy()

            # Solve: u_next = D^{-1} (f - L * u_prev)
            tmp = f.duplicate()
            f.scale(-1.0) # Set f <- -f
            L.multAdd(u_prev, f, tmp)  # rhs_tmp <- L * u_prev - f
            tmp.scale(-1.0) # rhs_tmp <- f - L * u_prev
            F = self.get_factored_matrix(D, self.ensemble.comm)
            u_next = tmp.duplicate()
            F.solve(tmp, u_next)

            self.a0[i,:] = u_next.getArray()[:]
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
            
            u_prev = u_next.copy()

            # Destruction
            L.destroy()
            D.destroy()
            f.destroy()
            tmp.destroy()
            F.destroy()
            u_next.destroy()
            
        # u_prev.destroy()

    def get_factored_matrix(self, A, comm):
        """
        Factor a matrix using LU decomposition with MUMPS.

        This function creates a PETSc preconditioner (PC) object configured to
        perform LU factorization using the MUMPS solver. It sets the provided
        matrix `A` as the operator and configures solver parameters to enhance
        numerical stability and suppress MUMPS-specific output. The factored
        matrix is then returned, allowing reuse in subsequent solves.

        Parameters
        ----------
        A : PETSc.Mat
            The matrix to factor. Must be assembled, square and either of type MPIAIJ or MPIDENSE.
        comm : PETSc.Comm
            The MPI communicator over which to create the PETSc PC.

        Returns
        -------
        PETSc.Mat
            The LU-factored matrix configured with MUMPS.

        Notes
        -----
        - This function sets `ICNTL(24) = 1` to improve pivot detection and
        `ICNTL(13) = 0` to suppress MUMPS output. Both `ICNTL(13) = 1` and
        `ICNTL(13) = 0` are supported.
        - The pivot convergence tolerance `CNTL(3)` is set to `1e-7`.

        Examples. Solve Ax = b
        --------
        >>> F = get_factored_matrix(A, comm)
        >>> F.solve(b, x)
        """
        local_pc = PETSc.PC().create(comm=comm)
        local_pc.setType("lu")
        local_pc.setFactorSolverType("mumps")
        local_pc.setOperators(A)
        local_pc.getFactorMatrix().setMumpsIcntl(24, 1)
        local_pc.getFactorMatrix().setMumpsIcntl(13, 0)
        local_pc.getFactorMatrix().setMumpsCntl(3, 1e-14)
        local_pc.setUp()
        return local_pc.getFactorMatrix()

problem_parameters = {
    "Number of time windows": 1, # No functionality for this yet
    "Number of temporal processors": 4, # Optimal choice is the root of the number of time steps
    "Number of time steps": 9, # Number of time steps must fit into a list following [2^k+1, 2^k, ..., 2^k] where k is an integer and the list length is equal to the number of temporal processors.
    "dt": 0.001,
    "nx": 4,
    "ny": 4,
    "degree_space": 1,
    "theta": 1,
}

processors = COMM_WORLD.size
n_timesteps = problem_parameters["Number of time steps"]
temporal_processors = problem_parameters["Number of temporal processors"]
nx = problem_parameters['nx']
ny = problem_parameters['ny']
dt = problem_parameters['dt']
degree_space = problem_parameters['degree_space']
theta = problem_parameters['theta']

# Create a time partition and an ensemble communicator
time_partition = create_time_partition(n_timesteps-1, temporal_processors)
ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

# Create a mesh with nx+1 and ny+1 vertices
mesh = UnitSquareMesh(nx = nx, ny = ny, comm = ensemble.comm)

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

aaoform = AllAtOnceForm(aaofunc, 
                        dt, 
                        theta, 
                        form_mass,
                        form_function, 
                        bcs=bcs)

aaofunc.assign(u0)
CRsolver = CRsolver(time_partition, ensemble, form_mass, form_function, aaofunc, aaoform, dt)
res = CRsolver.apply_impl(aaofunc)

# res._vec.view()

import shutil
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
        w_local = Function(V)
        w_local.assign(aaofunc[local_step])
        w_local.rename("u")
        if ensemble.ensemble_comm.rank != 0:
            ensemble.send(w_local, dest=0, tag=step)
        else:
            w_recv = w_local.copy(deepcopy=True)  # if rank 0 owns it, just copy

    if ensemble.ensemble_comm.rank == 0:
        if not aaoform.layout.is_local(step):
            w_recv = Function(V, name="u")
            ensemble.recv(w_recv, source=MPI.ANY_SOURCE, tag=step)

        vtkfile.write(w_recv, time=step * dt)
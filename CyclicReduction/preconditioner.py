import numpy as np

import firedrake as fd
from firedrake.petsc import PETSc

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.preconditioners.base import AllAtOnceBlockPCBase
from asQ.parallel_arrays import SharedArray
from asQ.allatonce import time_average

__all__ = ['CyclicReductionPC']

class CyclicReductionPC(AllAtOnceBlockPCBase):

    prefix = 'cyclic_reduction_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))

    @profiler()
    def initialize(self,pc):
        # Initialize is called once per rank
        super().initialize(pc, final_initialize=False)

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
            t0 = self.time[i]

            v = fd.TestFunction(self.field_function_space)
            u = fd.TrialFunction(self.field_function_space)

            M = self.form_mass(u, v)
            K = self.form_function(u, v, t0)

            F1 = dt1*M + theta*K # Main diagonal block system
            F2 = -dt1*M # Lower/off - diagonal block system

            # Represents the linear system for timestep/row i. That is L*u[i] + D*u[i+1] = f[i+1]
            D = fd.assemble(F1, bcs=self.block_bcs).petscmat
            L = fd.assemble(F2, bcs=self.block_bcs).petscmat

            if self.temporal_rank == 0 and i == 0:
                RHS = (1/self.dt) * self.form_mass(u0, v)
                f = fd.assemble(RHS, bcs=self.block_bcs)
                f = f.dat._vec # f[1] = L*u[0]
                self.first_lhs = D.copy()
                self.first_rhs = f.copy()
            else:
                f = u0.dat._vec.copy()
                f.scale(0.0) # f[i+1] = 0
                self.diag_matrices.append(D)
                self.lower_diag_matrices.append(L)
                self.rhs.append(f)

        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm) # Not currently used for anything
        self.initialized = True


    @profiler()
    def _record_diagnostics(self):
        pass

    @profiler()
    def update(self, pc):
        pass

    @profiler()
    def apply_impl(self, pc, x, y):
        
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
            F = self.get_factored_matrix(self.first_lhs, self.ensemble.comm)
            u1 = self.first_rhs.duplicate()
            F.solve(self.first_rhs, u1)

            local_u1 = u1.getArray()
            self.a0[0,:] = local_u1[:]
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
            
        # ---------------------------------------------------------------------------
        # FORWARD REDUCTION
        # ---------------------------------------------------------------------------
        L, D, f = self.forward_reduction(self.diag_matrices,
                                         self.lower_diag_matrices,
                                         self.rhs) 

        # ---------------------------------------------------------------------------
        # INTERFACE SOLVE (processor communication)
        # ---------------------------------------------------------------------------

        # Total number of temporal ranks
        n_temporal = self.ensemble.ensemble_comm.size

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
            u_prev = u1 

        # Solve: u_next = D^{-1} (f - L * u_prev)
        rhs = f.duplicate()
        f.scale(-1.0) # Set f <- -f
        L.multAdd(u_prev, f, rhs)  # rhs <- L * u_prev - f
        rhs.scale(-1.0) # rhs <- f - L * u_prev

        u_next = rhs.duplicate()
        F = self.get_factored_matrix(D.copy(), self.ensemble.comm)
        F.solve(rhs, u_next)

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
        
        # ---------------------------------------------------------------------------
        # BACKSUBSTITUTION (also writes to the global solution vector y)
        # ---------------------------------------------------------------------------
        self.forward_substitution(self.diag_matrices, self.lower_diag_matrices, self.rhs, u_prev, y)

        #PETSc.Sys.Print(f"yvec = {y._vec.view()}")
        #y.copy(self.state_func)


    @profiler()
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
        
        # Reduce onto these variables
        L, D, f = main_diag[0].copy(), lower_diag[0].copy(), rhs[0].copy()

        if len(main_diag) > 1: # this processor owns more than one timestep, so we reduce
            for i in range(1, len(main_diag)):
                L_next, D_next, f_next = main_diag[i].copy(), lower_diag[i].copy(), rhs[i].copy()
                
                D_factored = self.get_factored_matrix(D.copy(), self.ensemble.comm)
                L_dense = L.copy()
                L_dense.convert('dense')
                D_inv_L_dense = PETSc.Mat().createDense(size=L_dense.getSizes(), comm=self.ensemble.comm)
                D_inv_L_dense.setUp()
                D_inv_L_dense.assemble()

                # Compute L <- L_next * D^{-1} * L
                D_factored.matSolve(L_dense, D_inv_L_dense) # D^{-1} * L and stores in D_inv_L_dense
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

    
    @profiler()
    def forward_substitution(self, main_diag, lower_diag, rhs, u_prev, y):
        """
        Perform the back substitution step of the cyclic reduction algorithm.
        """
        offset = 1 if self.temporal_rank == 0 else 0 # Since temporal rank 0 is offset from other ranks by 1. It has 1 more row than other ranks

        for i in range(offset, len(main_diag)):
            L, D, f = main_diag[i].copy(), lower_diag[i].copy(), rhs[i].copy()

            # Solve: u_next = D^{-1} (f - L * u_prev)
            rhs = u_prev.duplicate()
            f.scale(-1.0) # Set f <- -f
            L.multAdd(u_prev, f, rhs)  # rhs <- L * u_prev - f
            rhs.scale(-1.0) # rhs <- f - L * u_prev
            F = self.get_factored_matrix(D.copy(), self.ensemble.comm)
            u_next = rhs.duplicate()
            F.solve(rhs, u_next)

            u_next_array = u_next.getArray()
            self.a0[i,:] = u_next_array[:]
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
            
            u_prev = u_next.copy()

            # Destruction
            L.destroy()
            D.destroy()
            f.destroy()
            rhs.destroy()
            F.destroy()
            u_next.destroy()
            
        u_prev.destroy()

        
        
    @profiler()
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
        local_pc.getFactorMatrix().setMumpsCntl(3, 1e-7)
        local_pc.setUp()
        return local_pc.getFactorMatrix()
    
    @profiler()
    def back_substitution_indices(self, reduction_levels, offset=0):
        """
        Generate indices for back substitution in cyclic reduction.
        
        This convenience function computes and returns a list of lists containing
        the indices of the removed variables at each level of the reduction process.
        Each inner list corresponds to a specific reduction level.
        
        Parameters
        ----------
        reduction_levels : int
            The number of levels of reduction.
        offset : int
            The offset to apply to the indices.
        
        Returns
        -------
        list of list of int
            A list where each inner list contains the indices for back substitution.
        
        Examples
        --------
        >>> back_substitution_indices(3, 0)
        [[1, 3, 5, 7], [2, 6], [4]]
        
        >>> back_substitution_indices(3, 1)
        [[2, 4, 6, 7], [3, 7], [5]]
        """
        rows = int(2**reduction_levels)
        indices = np.arange(offset, rows + offset)
        index_list = []
        for _ in range(reduction_levels):
            even_indices = indices[::2]
            odd_indices = indices[1::2]
            index_list.append(odd_indices)
            indices = even_indices

        return index_list
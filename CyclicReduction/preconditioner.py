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

        # PETSc.Sys.Print(f"dir of aaofunc: {dir(self.aaofunc)}")
        # Do these exist? What are they?
        # PETSc.Sys.Print(f"dir of aaoform: {dir(self.aaoform)}")
        # PETSc.Sys.Print(f"type aaoform : {type(self.aaoform.form)}")

        self.spatial_rank = self.ensemble.comm.rank
        self.temporal_rank = self.ensemble.ensemble_comm.rank

        # Zero out bc dofs
        self.block_bcs = tuple(
            fd.DirichletBC(self.field_function_space,
                           0*bc.function_arg,
                           bc.sub_domain)
            for bc in self.aaoform.field_bcs)
        
        # PETSc.Sys.Print(f" state_func view: {self.state_func[0].dat._vec.view()}")

        
        self.diag_matrices = []
        self.lower_diag_matrices = []
        self.rhs = []

        derivate = False
        dt1 = fd.Constant(1/self.dt)
        theta = fd.Constant(1)

        for i in range(self.nlocal_timesteps):
            # The reference states
            u0 = self.state_func[i]
            t0 = self.time[i]

            if derivate:
                vs = fd.TestFunctions(self.field_function_space)
                us = fd.split(u0)

                M = self.form_mass(*us, *vs)
                K = self.form_function(*us, *vs, t0)

                F1 = dt1*M + theta*K # Main diagonal block system
                F2 = -dt1*M # Lower/off - diagonal block system

                F1 = fd.derivative(F1, u0)
                F2 = fd.derivative(F2, u0)

            else:
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
                f = f.dat._vec
                # ii = self.state_func.transform_index(i, from_range='slice', to_range='window')
                # PETSc.Sys.Print(f"f at timestep {ii} = {f.getArray()}",comm=fd.COMM_SELF)
            else:
                f = u0.dat._vec.copy()
                f.scale(0.0)

            # ii = self.state_func.transform_index(i, from_range='slice', to_range='window')
            # PETSc.Sys.Print(f"f at timestep {ii} = {self._x[i].dat._vec.getArray()}",comm=fd.COMM_SELF)

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

    def apply_impl(self, pc, x, y):
        """
        Solve with normal time-stepping. To check if build is correct.
        """
        y.zero()

        # Pencil to help store solutions in the global array y
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        nlocal = self.field_function_space.node_set.size # Spatial DOFs for this rank
        NN = np.array([self.ntimesteps, nlocal], dtype=int)
        self.p0 = Pencil(subcomm, NN, axis=1)
        self.a0 = np.zeros(self.p0.subshape,dtype=np.float64)

        n_temporal = self.ensemble.ensemble_comm.size

        if self.temporal_rank == 0:
            first_block = self.diag_matrices[0].copy()
            first_rhs = self.rhs[0].copy()
            #first_rhs = x[0].dat._vec.copy()
            F = self.get_factored_matrix(first_block, self.ensemble.comm)
            x0 = first_rhs.duplicate()
            F.solve(first_rhs, x0)

            local_x0 = x0.getArray()
            # PETSc.Sys.Print(f"(Rank, local_step) :  {fd.COMM_WORLD.rank,0} x0 = {local_x0[:]} \n",comm=fd.COMM_SELF)
            self.a0[0,:] = local_x0[:]
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
        

        if self.temporal_rank > 0:
            # Receive x_prev from previous temporal rank
            x_prev_function = fd.Function(self.field_function_space)
            source = self.temporal_rank - 1
            self.ensemble.recv(x_prev_function, source=source, tag=88)
            # Convert x_prev_function to a PETSc Vec
            with x_prev_function.dat.vec as v:
                x_prev = v.copy()
        else:
            x_prev = x0 
        
        offset = 1 if self.temporal_rank == 0 else 0 # Since temporal rank 0 is offset from other ranks by 1
        current_diag = self.diag_matrices[offset:]
        current_off_diag = self.lower_diag_matrices[offset:]
        current_rhs = self.rhs[offset:]

        for i in range(len(current_diag)):
            L = current_off_diag[i]
            D = current_diag[i]
            f = current_rhs[i]

            # Solve: x_next = D^{-1} (f - L * x_prev)
            rhs = x_prev.duplicate()
            f.scale(-1.0)
            L.multAdd(x_prev, f, rhs)  # f <- L * x_prev - f
            rhs.scale(-1.0) # rhs <- f - L * x_prev
            F = self.get_factored_matrix(D, self.ensemble.comm)
            x_next = rhs.duplicate()
            F.solve(rhs, x_next)
            
            x_next_array = x_next.getArray()
            # PETSc.Sys.Print(f"(Rank, local_step) :  {fd.COMM_WORLD.rank,i+offset} x0 = {x_next_array[:]} \n",comm=fd.COMM_SELF)
            self.a0[i+offset,:] = x_next_array[:]

            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
        
        # Send to next temporal rank
        if self.temporal_rank < n_temporal - 1:
            dest = self.temporal_rank + 1
            # Convert x_next to a Firedrake Function
            x_next_function = fd.Function(self.field_function_space)
            with x_next_function.dat.vec as v:
                x_next.copy(v) # Copies data from x_next to v
            self.ensemble.send(x_next_function, dest=dest, tag=88)
        
        
        #PETSc.Sys.Print(f"yvec = {y._vec.view()}")
        

    @profiler()
    def apply_impl2(self, pc, x, y):
        
        y.zero()

        # for i in range(self.nlocal_timesteps):
        #     ii = self.state_func.transform_index(i, from_range='slice', to_range='window')
        #     PETSc.Sys.Print(f"x at timestep {ii} = {x[i].dat._vec.getArray()}",comm=fd.COMM_SELF)

        # if self.temporal_rank == 0:
        #     x00 = x[0].dat._vec.copy()
        #     PETSc.Sys.Print(f"x[0] = {x00.getArray()}",comm=fd.COMM_SELF)

        # if self.temporal_rank == 0:
        #     with x.global_vec_wo as xvec:
        #         PETSc.Sys.Print(f"x array = {xvec.array[:]}",fd.COMM_SELF)

        # Store these before the reduction step as self.diag_matrices and self.rhs are modified
        # during the reduction step.
        if self.temporal_rank == 0:
            first_block = self.diag_matrices[0].copy()
            first_rhs = self.rhs[0].copy()
            
            # PETSc.Sys.Print(f"first_rhs = {first_rhs.getArray()}",comm=fd.COMM_SELF)
        
        # ---------------------------------------------------------------------------
        # FORWARD REDUCTION
        # ---------------------------------------------------------------------------
        L_k, D_k, f_k, L_s, D_s, f_s = self.forward_reduction() 

        if self.temporal_rank == 0: # Temporal rank 0 needs to solve for x0.
            F = self.get_factored_matrix(first_block, self.ensemble.comm)
            x0 = first_rhs.duplicate()
            F.solve(first_rhs, x0)
            #PETSc.Sys.Print(f"(Rank, local_step) :  {fd.COMM_WORLD.rank,0} x0 = {x0.getArray()[:]} \n",comm=fd.COMM_SELF)
            
        # Define the pencil for the current rank for timestep ordering
        # p0 : Pencil describing spatial DOF distribution per timestep. E.g. If spatial rank 0, temporal rank 0
        # owns 3 timesteps of the global system, and owns 10 spatial DOFs in each then p0.subshape = (3,10)
        # This let's us use a0 to write to the global solution vector y 
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        nlocal = self.field_function_space.node_set.size # Spatial DOFs for this rank
        NN = np.array([self.ntimesteps, nlocal], dtype=int)
        self.p0 = Pencil(subcomm, NN, axis=1)
        self.a0 = np.zeros(self.p0.subshape,dtype=np.float64)

        # We can write x0 to the global solution vector y
        if self.temporal_rank == 0:
            local_x0 = x0.getArray()
            self.a0[0,:] = local_x0[:]
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
        
        # PETSc.Sys.Print(f"yvec = {y._vec.view()}")
        
        # ---------------------------------------------------------------------------
        # INTERFACE SOLVE (processor communication)
        # ---------------------------------------------------------------------------

        # Total number of temporal ranks
        n_temporal = self.ensemble.ensemble_comm.size

        # Allocate buffers
        if self.temporal_rank > 0:
            # Receive x_prev from previous temporal rank
            x_prev_function = fd.Function(self.field_function_space)
            source = self.temporal_rank - 1
            self.ensemble.recv(x_prev_function, source=source, tag=88)
            # Convert x_prev_function to a PETSc Vec
            with x_prev_function.dat.vec as v:
                x_prev = v.copy()
        else:
            x_prev = x0 

        # Solve: x_next = D_k^{-1} (f_k - L_k * x_prev)
        rhs = f_k.duplicate()
        f_k.scale(-1.0)
        L_k.multAdd(x_prev, f_k, rhs)  # f_k <- L_k * x_prev - f_k
        rhs.scale(-1.0) # rhs <- f_k - L_k * x_prev
        x_next = rhs.duplicate()
        F = self.get_factored_matrix(D_k, self.ensemble.comm)
        F.solve(rhs, x_next)

        # PETSc.Sys.Print(f"temporal rank {self.temporal_rank} x_next = {x_next.getArray()} \n",comm=fd.COMM_SELF)

        # Send to next temporal rank
        if self.temporal_rank < n_temporal - 1:
            dest = self.temporal_rank + 1
            # Convert x_next to a Firedrake Function
            x_next_function = fd.Function(self.field_function_space)
            with x_next_function.dat.vec as v:
                x_next.copy(v) # Copies data from x_next to v
            self.ensemble.send(x_next_function, dest=dest, tag=88)
        
        # All ranks now own a x_prev and x_next
        
        # ---------------------------------------------------------------------------
        # BACKSUBSTITUTION (also writes to the global solution vector y)
        # ---------------------------------------------------------------------------
        self.back_substitution(y, x_prev, L_s, D_s, f_s)

        PETSc.Sys.Print(f"yvec = {y._vec.view()}")
        y.copy(self.state_func)


    @profiler()
    def forward_reduction(self):
        """
        Perform the forward reduction step of the cyclic reduction algorithm.

        Notes
        -----
            - This function is not self-contained. It relies on the class variables
              `self.diag_matrices`, `self.lower_diag_matrices`, and `self.rhs`
            - The function modifies these variables during the reduction process.
            - Should potentially be refactored to avoid modifying class variables.
        """

        # Store the matrices and vectors for backsubstitution
        L_s = [] # This will become a list of lists of matrices, i.e. L_s = [[L1,L3,L7],[L2,L6],[L4]] 
        D_s = [] # This will become a list of lists of matrices, i.e. D_s = [[D1,D3,D7],[D2,D6],[D4]]
        f_s = [] # This will become a list of lists of vectors, i.e. f_s = [[f1,f3,f7],[f2,f6],[f4]]
        
        offset = 1 if self.temporal_rank == 0 else 0 # Since temporal rank 0 is offset from other ranks by 1
        
        current_diag = self.diag_matrices[offset:]
        current_off_diag = self.lower_diag_matrices[offset:]
        current_rhs = self.rhs[offset:]

        # We don't need to store these anymore
        self.diag_matrices = None
        self.lower_diag_matrices = None
        self.rhs = None

        while len(current_diag) != 1:

            # Placeholders for the next level of reduction
            next_diag = []
            next_off_diag = []
            next_rhs = []
            
            # These lists will fill L_s, D_s and f_s respectively
            L_s_temp = [] 
            D_s_temp = []
            f_s_temp = []

            for i in range(0,len(current_diag),2): # Step twice at a time

                D1 = current_diag[i]
                L1 = current_off_diag[i]
                f1 = current_rhs[i]

                D2 = current_diag[i+1]
                L2 = current_off_diag[i+1]
                f2 = current_rhs[i+1]

                # Intermediate storing for backward step (maybe add as self. variables)
                L_s_temp.append(L1.copy())
                D_s_temp.append(D1.copy())
                f_s_temp.append(f1.copy())

                # Compute D1^{-1} * f1 
                F = self.get_factored_matrix(D1, self.ensemble.comm)
                D1inv_f1 = f1.duplicate()
                F.solve(f1, D1inv_f1)

                # Compute new_f1 <- L2 * D1^{-1} * f1 - f2 
                new_f1 = f2.copy()
                f2.scale(-1.0) # Set f2 <- -f2
                L2.multAdd(D1inv_f1, f2, new_f1) # A.multAdd(x,v,y) computes Ax + v and stores in y

                # Compute D1^{-1} * L1 
                L1.convert('dense')
                D1inv_L1 = PETSc.Mat().createDense(size=L1.getSizes(), comm=self.ensemble.comm)
                D1inv_L1.setUp()
                D1inv_L1.assemble()
                F.matSolve(L1, D1inv_L1) # F.matSolve(B, X) solves FX=B for factored matrix F. Stores in X
                
                # Compute new_L1 <- L2 * D1^{-1} * L1
                new_L1 = L2.matMult(D1inv_L1)
                D1inv_L1.destroy()

                # Set new D1 <- -D2 
                new_D1 = D2.copy()
                new_D1.scale(-1.0)

                # Store for next level of reduction
                next_diag.append(new_L1)
                next_off_diag.append(new_D1)
                next_rhs.append(new_f1)

                # Destruction
                D1.destroy()
                D2.destroy()
                L1.destroy()
                L2.destroy()
                D1inv_L1.destroy()
                D1inv_f1.destroy()
                F.destroy()
                f1.destroy()
                f2.destroy()
            
            # Arrange the next level of reduction
            current_diag = next_diag
            current_off_diag = next_off_diag
            current_rhs = next_rhs

            # Store the matrices and vectors for backsubstitution
            L_s.append(L_s_temp)
            D_s.append(D_s_temp)
            f_s.append(f_s_temp)

        return current_off_diag[0], current_diag[0], current_rhs[0], L_s, D_s, f_s
    
    @profiler()
    def back_substitution(self, y, x_prev, L_s, D_s, f_s):
        """
        Perform the back substitution step of the cyclic reduction algorithm.
        """

        # Length of L_s, D_s, f_s should be equal to the number of levels of reduction/substitution steps
        substitution_steps = len(L_s)

        offset = 1 if self.temporal_rank == 0 else 0 # Temporal rank 0 is offset from other ranks by 1
        idxs = self.back_substitution_indices(substitution_steps,offset=offset) # Tells self.a0 where to put the data

        # First time step for each temporal rank (except rank 0) is x_prev 
        # and is already computed. Therefore, we can immediately write to global array y
        self.a0[offset,:] = x_prev.getArray()[:]
        with y.global_vec_wo() as yvec:
            yvec.array[:] = self.a0.reshape(-1)[:]

        # We need to iterate backwards through the levels of reduction
        # L_s, D_s, f_s are lists of lists of matrices/vectors
        # L_s[i] is a list of the lower matrices for the i-th level of reduction etc.

        x_s = [x_prev.copy()] # Keep track of the x's we solve
        for i in range(substitution_steps-1,-1,-1):

            new_xs = [] # List of new x's to be computed

            for j, (L, D, f, idx) in enumerate(zip(L_s[i], D_s[i], f_s[i], idxs[i])):
                
                # Solve new_x = D^{-1} * (f - L * x[j])
                rhs = f.duplicate()
                f.scale(-1.0)
                L.multAdd(x_s[j], f, rhs)  # rhs <- L * x[j] - f
                rhs.scale(-1.0) # rhs <- f - L * x[j]

                x_new = f.duplicate()
                F = self.get_factored_matrix(D, self.ensemble.comm)
                F.solve(rhs, x_new)

                self.a0[idx,:] = x_new.getArray()[:]
                with y.global_vec_wo() as yvec:
                    yvec.array[:] = self.a0.reshape(-1)[:]
                new_xs.append(x_new.copy())

            # Due to the forward reduction step removing every other row, 
            # we need to interleave our current x's with the new x's
            # to restore the original ordering.
            # For example, if we have x_s = [x0,x2,x4] and new_xs = [x1,x3],
            # we want to interleave them to get x_s = [x0,x1,x2,x3,x4].
            x_s = [elem for pair in zip(x_s, new_xs) for elem in pair]
        
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
import numpy as np

import firedrake as fd
from firedrake.petsc import PETSc

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.preconditioners.base import AllAtOnceBlockPCBase, get_default_options, AllAtOncePCBase
from asQ.parallel_arrays import SharedArray

from CyclicReduction.utils import (
    back_substitution_indices
)

__all__ = ['CyclicReductionPC']


class CyclicReductionPC(AllAtOnceBlockPCBase):

    prefix = 'cyclic_reduction_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))
    default_theta = 1 # Backward Euler

    @profiler()
    def initialize(self,pc):
        # Initialize is called once per rank
        super().initialize(pc, final_initialize=False)

        aaofunc = self.aaofunc

        # All-at-once reference state
        self.state_func = aaofunc.copy()

        # Function space for a single time-step
        self.field_function_space = aaofunc.field_function_space 

        # Function space for the slice of the all-at-once system on this process
        function_space = aaofunc.function_space 

        # Building the nonlinear operator
        self.block_solvers = []

        # zero out bc dofs
        self.block_bcs = tuple(
            fd.DirichletBC(self.field_function_space,
                           0*bc.function_arg,
                           bc.sub_domain)
            for bc in self.aaoform.field_bcs)

        # user appctx for the blocks
        block_appctx = self.appctx.get('block_appctx', {})

        dt1 = fd.Constant(1/self.dt)
        tht = fd.Constant(self.theta)
        tht = fd.Constant(1)

        # Block i has prefix 'aaojacobi_block_{i}', but we want to be able
        # to set default options for all blocks using 'aaojacobi_block'.
        # LinearVariationalSolver will prioritise options it thinks are from
        # the command line (including those in the `inserted_options` database
        # of the AllAtOnceSolver) over the ones passed to __init__, so we pull
        # the default options off the global dict and pass these explicitly to LVS.
        default_block_prefix = f"{self.full_prefix}block_"
        default_block_options = get_default_options(
            default_block_prefix, range(self.ntimesteps))
        
        self.diag_matrices = []
        self.off_diag_matrices = []
        self.cr_rhs = []
        self.cr_sol = []

        self.spatial_rank = self.ensemble.comm.rank
        self.temporal_rank = self.ensemble.ensemble_comm.rank

        #PETSc.Sys.Print(f"Temporal rank {self.temporal_rank} with spatial rank {self.spatial_rank}", comm = COMM_SELF)
        # spatial_block_size = field_function_space.dim()
        # mT = (self.ensemble.ensemble_comm.size + 1)*spatial_block_size
        # self.intermediate_matrix = PETSc.Mat().createAIJ(size=(mT,mT),comm=self.ensemble.ensemble_comm)
        
        #PETSc.Sys.Print(f"Inter MATRIX ownershs : {self.intermediate_matrix.getOwnershipRanges()}")

        A, _ = pc.getOperators()
        
        # Pretty sure aaofunc[0] is the RHS of the first time-step of the global system
        nlocal_timesteps_start = 0

        if self.temporal_rank == 0:
            t0 = self.time[0]

            v = fd.TestFunction(self.field_function_space)
            u = fd.TrialFunction(self.field_function_space)

            M_mass = self.form_mass(u, v)
            K_stiff = self.form_function(u, v, t0)

            F1 = dt1 * M_mass + tht * K_stiff # Main diagonal block system

            A = fd.assemble(F1, bcs=self.block_bcs)

            A_petsc = fd.as_backend_type(A).mat().copy()

            self.first_block = A_petsc # Required for the intermediate step
            self.first_rhs = aaofunc[0].dat._vec.copy()
            # self.first_rhs = fd.as_backend_type(self._x[0].vector().copy()).vec()
            self.first_sol = fd.as_backend_type(self._y[0].vector().copy()).vec()

                    

            #PETSc.Sys.Print(f"view self.first_rhs : {self.first_rhs.view()}",comm=COMM_SELF)

            # PETSc.Sys.Print(f"first_block has size(s): {self.first_block.getSizes()}",comm=COMM_SELF)
            # PETSc.Sys.Print(f"first_rhs has size(s): {self.first_rhs.getSizes()}",comm=COMM_SELF)
            # PETSc.Sys.Print(f"first_sol has size(s): {self.first_rhs.getSizes()}",comm=COMM_SELF)

            nlocal_timesteps_start = 1

        for i in range(nlocal_timesteps_start, self.nlocal_timesteps):
            # the reference states
            u0 = self.state_func[i]
            t0 = self.time[i]

            v = fd.TestFunction(self.field_function_space)
            u = fd.TrialFunction(self.field_function_space)

            M_mass = self.form_mass(u, v)
            K_stiff = self.form_function(u, v, t0)

            F1 = dt1 * M_mass + tht * K_stiff # Main diagonal block system
            F2 = -dt1 * M_mass # Lower/off - diagonal block system

            A = fd.assemble(F1, bcs=self.block_bcs)
            B = fd.assemble(F2, bcs=self.block_bcs)

            A_petsc = fd.as_backend_type(A).mat().copy()
            B_petsc = fd.as_backend_type(B).mat().copy()

            self.diag_matrices.append(A_petsc)
            self.off_diag_matrices.append(B_petsc)

            rhs_vec = fd.as_backend_type(self._x[i].vector().copy()).vec()
            sol_vec = fd.as_backend_type(self._y[i].vector().copy()).vec()

            self.cr_sol.append(sol_vec)
            self.cr_rhs.append(rhs_vec)


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
        # x rhs, y sol
        # Applies the action of A * y = x, should return y = A^{-1} * x

        # x and y are already the rhs and solution of the blocks
        # with x.global_vec_ro() as xvec:
        #     x_vec = xvec
        #     #PETSc.Sys.Print(f"Type x_vec : {type(x_vec)}") # <class 'petsc4py.PETSc.Vec'>
        #     #PETSc.Sys.Print(f"Type x_vec.array : {type(x_vec.array)}") # <class 'numpy.ndarray'>
        
        # with y.global_vec_wo() as yvec:
        #     y_vec = yvec
        
        # self.ksp.solve(x_vec,y_vec)
        #PETSc.Sys.Print(f"Whose calling? Rank {self.temporal_rank} subrank {self.spatial_rank}",comm=COMM_SELF)

        B_k, A_k, f_k, B_s, A_s, f_s = self.forward_reduction(pc,x,y) # We still need to deal with A_0. Currently only have A_i, B_i for i = 1,...,n+1

        # N Temporal processes/ranks have finished their forward reduction step. Each own M spatial ranks.

        if self.temporal_rank == 0: # Temporal rank 0 (with M spatial ranks) needs to solve for x0.
            local_pc = PETSc.PC().create(comm=self.ensemble.comm)
            local_pc.setType("lu")
            local_pc.setFactorSolverType("mumps") # cholesky also ok :)
            local_pc.setOperators(self.first_block)
            local_pc.getFactorMatrix().setMumpsIcntl(24, 1)
            local_pc.getFactorMatrix().setMumpsIcntl(13, 0) # both (13,1) and (13,0) works!
            local_pc.getFactorMatrix().setMumpsCntl(3, 1e-7)
            local_pc.setUp()
            F = local_pc.getFactorMatrix() # F is the factored matrix of A1
            x0 = self.first_rhs.duplicate()
            F.solve(self.first_rhs, x0) # x0 is now solved for temporal rank 0, distributed overs its M spatial ranks.

        # The solution is a motherfucking PENCIL BITCH!
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        nlocal = self.field_function_space.node_set.size # DOFs for this rank
        NN = np.array([self.ntimesteps, nlocal], dtype=int)

        # p0 : Pencil describing spatial DOF distribution per timestep. E.g. If patial rank 0, temporal rank 0
        # owns 3 timesteps of the global system, and owns 10 spatial DOFs in each then
        # p0.subshape = (3,10)
        self.p0 = Pencil(subcomm, NN, axis=1)
        self.a0 = np.zeros(self.p0.subshape,dtype=np.float64)

        # p1 : Redescribes p0 to a pencil over all timesteps.
        # p1 = self.p0.pencil(0)
        # a1 = np.zeros(p1.subshape,dtype=np.float64)
        # transfer = self.p0.transfer(p1,dtype=np.float64)

        if self.temporal_rank == 0:
            local_x0 = x0.getArray()
            self.a0[0,:] = local_x0[:]
            
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
        
        PETSc.Sys.Print(f"Rank {self.temporal_rank, self.spatial_rank} a0.shape : {self.a0.shape}", comm=fd.COMM_SELF)
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

        # Solve: x_next = A_k^{-1} (f_k - B_k * x_prev)
        rhs = f_k.duplicate()
        f_k.scale(-1.0)
        B_k.multAdd(x_prev, f_k, rhs)  # f_k <- B_k * x_prev - f_k
        rhs.scale(-1.0) # rhs <- f_k - B_k * x_prev

        x_next = rhs.duplicate()
        local_pc = PETSc.PC().create(comm=self.ensemble.comm)
        local_pc.setType("lu") # can also do 'cholesky' here?
        local_pc.setFactorSolverType("mumps")
        local_pc.setOperators(A_k)
        local_pc.getFactorMatrix().setMumpsIcntl(24, 1)
        local_pc.getFactorMatrix().setMumpsIcntl(13, 0) # both (13,1) and (13,0) works!
        local_pc.getFactorMatrix().setMumpsCntl(3, 1e-7)
        local_pc.setUp()
        F = local_pc.getFactorMatrix()
        F.solve(rhs, x_next)

        # Send to next temporal rank
        if self.temporal_rank < n_temporal - 1:
            dest = self.temporal_rank + 1
            # Convert x_next to a Firedrake Function
            x_next_function = fd.Function(self.field_function_space)
            with x_next_function.dat.vec as v:
                x_next.copy(v) # Copies data from x_next to v
            self.ensemble.send(x_next_function, dest=dest, tag=88)
        
        # All ranks now own a x_prev and x_next
        # Now we need to do the backward reduction step
        self.back_substitution(pc, y, x_prev, x_next, B_s, A_s, f_s)

        # local_x = x_prev.getArray()
        # self.a0[1,:] = local_x[:]
        # with y.global_vec_wo() as yvec:
        #     yvec.array[:] = self.a0.reshape(-1)[:]
        
        PETSc.Sys.Print(f"view y {y._vec.view()}")


    @profiler()
    def back_substitution(self, pc, y, x_prev, x_next, B_s, A_s, f_s):
        # length of B_s, A_s, f_s should be equal to the number of levels of reduction
        substitution_steps = len(B_s)

        offset = 1 if self.temporal_rank == 0 else 0 # Temporal rank 0 is offset from other ranks by 1
        idxs = back_substitution_indices(substitution_steps,offset=offset) # Tells self.a0 where to put the data

        # First time step (except x0) for each temporal rank is x_prev and is already computed. Write to global array y
        self.a0[offset,:] = x_prev.getArray()[:]
        with y.global_vec_wo() as yvec:
            yvec.array[:] = self.a0.reshape(-1)[:]

        # We need to go backwards through the levels of reduction
        # B_s, A_s, f_s are lists of lists of matrices/vectors
        # B_s[i] is a list of the lower matrices for the i-th level of reduction etc.
        x_s = [x_prev.copy()]
        for i in range(substitution_steps-1,-1,-1):
            new_xs = []
            for j, (B, A, f, idx) in enumerate(zip(B_s[i], A_s[i], f_s[i], idxs[i])):
                # Do something
                # B*x_before + A*x_after = f
                # Solve: x[j+1] = A^{-1} (f - B * x[j])
                rhs = f.duplicate()
                f.scale(-1.0)
                B.multAdd(x_s[j], f, rhs)  # f_k <- B_k * x_prev - f_k
                rhs.scale(-1.0) # rhs <- f_k - B_k * x_prev

                x_new = f.duplicate()
                local_pc = PETSc.PC().create(comm=self.ensemble.comm)
                local_pc.setType("lu") # can also do 'cholesky' here?
                local_pc.setFactorSolverType("mumps")
                local_pc.setOperators(A)
                local_pc.getFactorMatrix().setMumpsIcntl(24, 1)
                local_pc.getFactorMatrix().setMumpsIcntl(13, 0) # both (13,1) and (13,0) works!
                local_pc.getFactorMatrix().setMumpsCntl(3, 1e-7)
                local_pc.setUp()
                F = local_pc.getFactorMatrix()
                F.solve(rhs, x_new)

                self.a0[idx,:] = x_new.getArray()[:]
                with y.global_vec_wo() as yvec:
                    yvec.array[:] = self.a0.reshape(-1)[:]
                new_xs.append(x_new.copy())

            x_s = [elem for pair in zip(x_s, new_xs) for elem in pair]
        


        

    @profiler()
    def forward_reduction(self,pc,x,y):

        B_s = [] # This should be list of lists of matrices, i.e. [[B1,B3,B7],[B2,B6],[B4]] 
        A_s = [] # This should be list of lists of matrices, i.e. [[A1,A3,A7],[A2,A6],[A4]]
        f_s = [] # This should be list of lists of vectors, i.e. [[f1,f3,f7],[f2,f6],[f4]]
        
        time_steps = self.nlocal_timesteps
        if self.temporal_rank == 0:
            time_steps -= 1
        
        steps_of_reduction = int(np.log2(time_steps))

        current_diag = self.diag_matrices.copy()
        current_off_diag = self.off_diag_matrices.copy()
        current_rhs = self.cr_rhs.copy()

        while time_steps != 1:

            next_diag = []
            next_off_diag = []
            next_rhs = []
            
            B_s_temp = [] 
            A_s_temp = []
            f_s_temp = []

            for i in range(0,time_steps,2): # Step twice at a time
                A1 = current_diag[i]
                A2 = current_diag[i+1]

                B1 = current_off_diag[i]
                B2 = current_off_diag[i+1]

                f1 = current_rhs[i]
                f2 = current_rhs[i+1]

                # === Store for backward step (maybe add as self. variables) ===
                B_s_temp.append(B1.copy())
                A_s_temp.append(A1.copy())
                f_s_temp.append(f1.copy())

                # === Factor A1 ===
                #A1_dense = A1.convert('dense')
                local_pc = PETSc.PC().create(comm=self.ensemble.comm)
                local_pc.setType("lu") # can also do 'cholesky' here?
                local_pc.setFactorSolverType("mumps")
                local_pc.setOperators(A1)
                local_pc.getFactorMatrix().setMumpsIcntl(24, 1)
                local_pc.getFactorMatrix().setMumpsIcntl(13, 0) # both (13,1) and (13,0) works!
                local_pc.getFactorMatrix().setMumpsCntl(3, 1e-7)
                local_pc.setUp()
                F = local_pc.getFactorMatrix() # F is the factored matrix of A1

                # === Solve A^{-1} * f1 ===
                A1inv_f1 = f1.duplicate()
                F.solve(f1, A1inv_f1)

                # === Compute new_f1 <- B2 * A^{-1} * f1 - f2 ===
                new_f1 = f2.copy()
                f2.scale(-1.0) # Set f2 <- -f2
                B2.multAdd(A1inv_f1, f2, new_f1) # A.multAdd(x,v,y) computes Ax + v and stores in y

                # === Solve A^{-1} * B1 ===
                B1.convert('dense')
                #PETSc.Sys.Print(f"B1_dense view: {B1_dense.view()} \n ",comm=COMM_SELF)
                A1inv_B1 = PETSc.Mat().createDense(size=B1.getSizes(), comm=self.ensemble.comm)
                A1inv_B1.setUp()
                A1inv_B1.assemble()
                F.matSolve(B1, A1inv_B1) # F.matSolve(B, X) solves FX=B for factored matrix F. Stores in X
                
                # === Compute new_B1 <- B2 * A1^{-1} * B1 ===
                new_B1 = B2.matMult(A1inv_B1)
                A1inv_B1.destroy()

                # === Set new A1 <- -A2  ===
                new_A1 = A2.copy()
                new_A1.scale(-1.0)

                # === Store for next level of reduction ===
                next_diag.append(new_B1)
                next_off_diag.append(new_A1)
                next_rhs.append(new_f1)

                # === Destruction ===
                A1.destroy()
                A2.destroy()
                B1.destroy()
                B2.destroy()
                A1inv_B1.destroy()
                A1inv_f1.destroy()
                F.destroy()
                f1.destroy()
                f2.destroy()
            
            current_diag = next_diag
            current_off_diag = next_off_diag
            current_rhs = next_rhs
            time_steps = time_steps//2

            B_s.append(B_s_temp)
            A_s.append(A_s_temp)
            f_s.append(f_s_temp)

        return current_off_diag[0], current_diag[0], current_rhs[0], B_s, A_s, f_s
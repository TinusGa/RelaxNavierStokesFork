import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import COMM_SELF, COMM_WORLD

import time

from warnings import warn
import numpy as np
from scipy.fft import fft, ifft

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.common import get_option_from_list, get_deprecated_option

from asQ.allatonce.function import time_average as time_average_function
from asQ.preconditioners.base import AllAtOnceBlockPCBase, get_default_options

from asQ.ensemble import split_ensemble

from asQ.parallel_arrays import SharedArray

from functools import partial

__all__ = ['CyclicReductionPC','CyclicReductionPC1']

class CyclicReductionPC(AllAtOnceBlockPCBase):

    prefix = 'cyclic_reduction_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))
    default_theta = 1 # Backward Euler

    @profiler()
    def initialize(self,pc):
        # Initialize is called once per MPI processors
        super().initialize(pc, final_initialize=False)

        aaofunc = self.aaofunc

        # all-at-once reference state
        self.state_func = aaofunc.copy()

        # single timestep function space
        field_function_space = aaofunc.field_function_space

        # Building the nonlinear operator
        self.block_solvers = []

        # zero out bc dofs
        self.block_bcs = tuple(
            fd.DirichletBC(field_function_space,
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
        spatial_block_size = field_function_space.dim()
        mT = (self.ensemble.ensemble_comm.size + 1)*spatial_block_size
        self.intermediate_matrix = PETSc.Mat().createAIJ(size=(mT,mT),comm=self.ensemble.ensemble_comm)
        
        PETSc.Sys.Print(f"Inter MATRIX ownershs : {self.intermediate_matrix.getOwnershipRanges()}")

        _, A = pc.getOperators()

        # Building the block problem solvers
        if self.temporal_rank == 0:
            self.nlocal_timesteps -= 1

        for i in range(self.nlocal_timesteps):
            # the reference states
            u0 = self.state_func[i]
            t0 = self.time[i]

            v = fd.TestFunction(field_function_space)
            u = fd.TrialFunction(field_function_space)

            M_mass = self.form_mass(u, v)
            K_stiff = self.form_function(u, v, t0)

            F1 = dt1 * M_mass + tht * K_stiff # Main diagonal block system
            F2 = -dt1 * M_mass # Lower/off - diagonal block system

            A = fd.assemble(F1, bcs=self.block_bcs)
            B = fd.assemble(F2, bcs=self.block_bcs)

            A_petsc = fd.as_backend_type(A).mat()
            B_petsc = fd.as_backend_type(B).mat()

            self.first_block = A_petsc # Required for the intermediate step

            self.diag_matrices.append(A_petsc)
            self.off_diag_matrices.append(B_petsc)

            sol_vec = fd.as_backend_type(self._y[i].vector()).vec()
            rhs_vec = fd.as_backend_type(self._x[i].vector()).vec()

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
        # x and y are already the rhs and solution of the blocks
        # with x.global_vec_ro() as xvec:
        #     x_vec = xvec
        #     #PETSc.Sys.Print(f"Type x_vec : {type(x_vec)}") # <class 'petsc4py.PETSc.Vec'>
        #     #PETSc.Sys.Print(f"Type x_vec.array : {type(x_vec.array)}") # <class 'numpy.ndarray'>
        
        # with y.global_vec_wo() as yvec:
        #     y_vec = yvec
        
        # self.ksp.solve(x_vec,y_vec)
        #PETSc.Sys.Print(f"Whose calling? Rank {self.temporal_rank} subrank {self.spatial_rank}",comm=COMM_SELF)

        start = time.time()
        B_k, A_k, sol_k, f_k, B_s, A_s, f_s = self.forward_reduction(pc,x,y) # We still need to deal with A_0. Currently only have A_i, B_i for i = 1,...,n+1
        end = time.time()

        #PETSc.Sys.Print(f"T_rank, spat_rank {self.temporal_rank, self.spatial_rank} spent {end-start}s in forward reduction", comm=COMM_SELF)
        # Block all but one temporal_processor?? Do sequential work
        # PETSc.Sys.Print(f"Ownership range sol_k {sol_k.getOwnershipRanges()}", comm = COMM_SELF)
        # PETSc.Sys.Print(f"Ownership range y {y._vec.getOwnershipRanges()}", comm = COMM_SELF)
        # PETSc.Sys.Print(f"Ownership range x {x._vec.getOwnershipRanges()}", comm = COMM_SELF)

        if self.temporal_rank == 0:
            pass

        #COMM_WORLD.Barrier()
        

        # Begin all processes again
        # self.backward_solution()....

        # ksp = PETSc.KSP().create()
        # ksp.setOperators(self.first_block) # [A_0, ..., 0] [x_0] = [f_0 - B_0 i.c.]
        # ksp.setOptionsPrefix(self.full_prefix + "cyclic_reduction_")
        # ksp.setFromOptions()
        # ksp.solve(f1,A1inv_f1)

        # First solve for x0 for first proc, send result to next proc. Solve on next proc, send to the one after. Repeat.



        # self._y.zero()
        # for i in range(self.nlocal_timesteps):
        #     self.block_solvers[i].solve()

    @profiler()
    def forward_reduction(self,pc,x,y):

        B_s = []
        A_s = []
        f_s = []
        
        time_steps = self.nlocal_timesteps
        current_diag = self.diag_matrices.copy()
        current_off_diag = self.off_diag_matrices.copy()
        current_rhs = self.cr_rhs.copy()
        current_sol = self.cr_sol.copy()

        while time_steps != 1:

            next_diag = []
            next_off_diag = []
            next_rhs = []
            next_sol = []

            for i in range(0,time_steps,2): # Step twice at a time
                A1 = current_diag[i]
                A2 = current_diag[i+1]

                B1 = current_off_diag[i]
                B2 = current_off_diag[i+1]

                u1 = current_sol[i]
                u2 = current_sol[i+1]

                f1 = current_rhs[i]
                f2 = current_rhs[i+1]

                # === Store for backward step (maybe add as self. variables) ===
                B_s.append(B1.copy())
                A_s.append(A1.copy())
                f_s.append(f1.copy())

                # === Factor A1 ===
                #A1_dense = A1.convert('dense')
                local_pc = PETSc.PC().create(comm=self.ensemble.comm)
                local_pc.setType("cholesky")
                local_pc.setFactorSolverType("mumps")
                local_pc.setOperators(A1)
                local_pc.getFactorMatrix().setMumpsIcntl(24, 1)
                local_pc.getFactorMatrix().setMumpsIcntl(13, 1)
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

                # === Set-up dense matrices for computing A^{-1} * B1 ===
                B1.convert('dense')
                #PETSc.Sys.Print(f"B1_dense view: {B1_dense.view()} \n ",comm=COMM_SELF)
                A1inv_B1 = PETSc.Mat().createDense(size=B1.getSizes(), comm=self.ensemble.comm)
                A1inv_B1.setUp()
                A1inv_B1.assemble()
                #F = local_pc.getFactorMatrix()
                F.matSolve(B1, A1inv_B1) # F.matSolve(B, X) solves FX=B for factored matrix F. Stores in X
                # === Solve A^{-1} * B1 ===
                # try:
                #     F.matSolve(B1_dense, A1inv_B1) # F.matSolve(B, X) solves FX=B for factored matrix F. Stores in X
                # except Exception as e:
                #     # A_diag = A1.getDiagonal()
                #     # PETSc.Sys.Print(f"A's diag: {A_diag.array}. At time step {time_steps} of total steps {self.nlocal_timesteps}", comm=COMM_SELF)
                #     PETSc.Sys.Print(f"Error: {e}", comm=COMM_SELF)


                # A1inv_B1 = PETSc.Mat().createDense(size=B1.getSizes(), comm=self.ensemble.comm) # or COMM_SELF??
                # A1inv_B1.setUp()

                # row_start, row_end = A1inv_B1.getOwnershipRange()
                # col_start, col_end = A1inv_B1.getOwnershipRangeColumn()

                # #PETSc.Sys.Print(f"Rows: {row_start, row_end}, Cols: {col_start, col_end}",comm=COMM_SELF)
                # #PETSc.Sys.Print(f"Got here mr!. status: {self.temporal_rank, self.spatial_rank}. Rows: {row_start, row_end}, Cols: {col_start, col_end}",comm=COMM_SELF)

                # COMM_WORLD.Barrier()
                # for j in range(col_start,col_end):
                #     # Extract the j-th column from B1_dense into a vector
                #     b_col = B1.getColumnVector(j)
                #     x_col = b_col.duplicate()
                #     F.solve(b_col, x_col)
                #     # Now, update the j-th column of A1inv_B1 using setValue.
                #     #PETSc.Sys.Print(f"x_col owns: {x_col.getOwnershipRange()} and {x_col.getOwnershipRanges()}",comm=COMM_SELF)
                #     for i in range(row_start,row_end):
                #         # Get the i-th entry of x_col and set it in A1inv_B1 at (i, j)
                #         A1inv_B1.setValue(i, j, x_col.getValue(i))

                #     PETSc.Sys.Print(f"First barrier (j = {j-col_start}). Status: {self.temporal_rank, self.spatial_rank}",comm=COMM_SELF)
                #     COMM_WORLD.Barrier()
                #     PETSc.Sys.Print(f"\n")

                # PETSc.Sys.Print(f"Fourth barrier. Status: {self.temporal_rank, self.spatial_rank}",comm=COMM_SELF)
                # COMM_WORLD.Barrier()
                # PETSc.Sys.Print(f"Got here 1. status: {self.temporal_rank, self.spatial_rank}",comm=COMM_SELF)
                # A1inv_B1.assemblyBegin()
                # #PETSc.Sys.Print(f"Got here betweenz",comm=COMM_SELF)
                # A1inv_B1.assemblyEnd()
                # PETSc.Sys.Print(f"Got here 2",comm=COMM_SELF)

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
                next_sol.append(u2.duplicate())  # Placeholder for next solution

                # === Destruction ===
                A1.destroy()
                A2.destroy()
                B1.destroy()
                B2.destroy()
                #B1_dense.destroy()
                A1inv_B1.destroy()
                A1inv_f1.destroy()
                F.destroy()
                f1.destroy()
                f2.destroy()
            
            current_diag = next_diag
            current_off_diag = next_off_diag
            current_rhs = next_rhs
            current_sol = next_sol
            time_steps = time_steps//2
        
        return current_off_diag[0], current_diag[0], current_sol[0], current_rhs[0], B_s, A_s, f_s




class CyclicReductionPC1(AllAtOnceBlockPCBase):

    prefix = 'cyclic_reduction_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))
    default_theta = 1 # Backward Euler

    @profiler()
    def initialize(self,pc):
        # Initialize is called once per MPI processors
        super().initialize(pc, final_initialize=False)

        aaofunc = self.aaofunc

        # all-at-once reference state
        self.state_func = aaofunc.copy()

        # single timestep function space
        field_function_space = aaofunc.field_function_space

        # Building the nonlinear operator
        self.block_solvers = []

        # zero out bc dofs
        self.block_bcs = tuple(
            fd.DirichletBC(field_function_space,
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
        
        self.diag_solver = []
        self.off_diag_solver = []
        self.cr_rhs = []
        self.cr_sol = []

        self.spatial_rank = self.ensemble.comm.rank
        self.temporal_rank = self.ensemble.ensemble_comm.rank

        #PETSc.Sys.Print(f"Temporal rank {self.temporal_rank} with spatial rank {self.spatial_rank}", comm = COMM_SELF)
        spatial_block_size = field_function_space.dim()
        mT = (self.ensemble.ensemble_comm.size + 1)*spatial_block_size
        self.intermediate_matrix = PETSc.Mat().createAIJ(size=(mT,mT),comm=self.ensemble.ensemble_comm)
        
        # PETSc.Sys.Print(f"Inter MATRIX ownershs : {self.intermediate_matrix.getOwnershipRanges()}")

        #_, A = pc.getOperators()

        # Building the block problem solvers
        if self.temporal_rank == 0:
            self.nlocal_timesteps -= 1

        # Let's instead build LVP's and LVS's
        for i in range(self.nlocal_timesteps):
            # the reference states
            u0 = self.state_func[i]
            t0 = self.time[i]

            v = fd.TestFunction(field_function_space)
            u = fd.TrialFunction(field_function_space)

            M_mass = self.form_mass(u, v)
            K_stiff = self.form_function(u, v, t0)

            F1 = dt1 * M_mass + tht * K_stiff # Main diagonal block system
            F2 = -dt1 * M_mass # Lower/off - diagonal block system

            A = fd.assemble(F1)
            B = fd.assemble(F2)

            # pass parameters into PC:
            appctx_h = {
                "dt": self.dt,
                "theta": self.theta,
                "tref": t0,
                "uref": u0,
                "bcs": self.block_bcs,
                "form_mass": self.form_mass,
                "form_function": self.form_function,
            }

            appctx_h.update(block_appctx)

            ii = aaofunc.transform_index(i, from_range='slice', to_range='window')

            # The block rhs/solution are the timestep i of the
            # input/output AllAtOnceCofunction/Function
            diag_problem = fd.LinearVariationalProblem(A, self._x[i], self._y[i],
                                                        bcs=self.block_bcs,
                                                        constant_jacobian=True)
            off_diag_problem = fd.LinearVariationalProblem(B, self._x[i], self._y[i],
                                                        bcs=self.block_bcs,
                                                        constant_jacobian=True)
            diag_solver = fd.LinearVariationalSolver(
                diag_problem, appctx=appctx_h,
                options_prefix=default_block_prefix+str(ii),
                solver_parameters=default_block_options)
            
            self.diag_solver.append(diag_solver)

            off_diag_solver = fd.LinearVariationalSolver(
                off_diag_problem, appctx=appctx_h,
                options_prefix=default_block_prefix+str(ii),
                solver_parameters=default_block_options)
            
            self.off_diag_solver.append(off_diag_solver)

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

        start = time.time()
        B_k, A_k, B_s, A_s, f_s = self.forward_reduction(pc,x,y) # We still need to deal with A_0. Currently only have A_i, B_i for i = 1,...,n+1
        end = time.time()

        if self.temporal_rank == 0:
            pass

        #COMM_WORLD.Barrier()

    @profiler()
    def forward_reduction(self,pc,x,y):

        B_s = []
        A_s = []
        
        time_steps = self.nlocal_timesteps
        current_diag = self.diag_solver
        current_off_diag = self.off_diag_solver

        while time_steps != 1:

            next_diag = []
            next_off_diag = []

            for i in range(0,time_steps,2): # Step twice at a time
                # Solve the blocks I suppose?
                current_diag[i].solve() # A1 x = f
                current_diag[i+1].solve() # A2 x = f

                current_off_diag[i].solve() # B1 x = f
                current_off_diag[i+1].solve() # B2 x = f

                PETSc.Sys.Print(f"current diag {type(current_diag[i])}",comm=COMM_SELF)
                PETSc.Sys.Print(f"current diag dir {dir(current_diag[i])} \n",comm=COMM_SELF)


            current_diag = next_diag
            current_off_diag = next_off_diag
            # current_rhs = next_rhs
            # current_sol = next_sol
            time_steps = time_steps//2
        
        return current_off_diag[0], current_diag[0], B_s, A_s
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
from asQ.preconditioners.base import AllAtOnceBlockPCBase, get_default_options, AllAtOncePCBase

from asQ.ensemble import split_ensemble

from asQ.parallel_arrays import SharedArray

from functools import partial

__all__ = ['CyclicReductionPC']

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
        
        #PETSc.Sys.Print(f"Inter MATRIX ownershs : {self.intermediate_matrix.getOwnershipRanges()}")

        _, A = pc.getOperators()

        # Building the block problem solvers
        nlocal_timesteps_start = 0

        if self.temporal_rank == 0:
            t0 = self.time[0]

            v = fd.TestFunction(field_function_space)
            u = fd.TrialFunction(field_function_space)

            M_mass = self.form_mass(u, v)
            K_stiff = self.form_function(u, v, t0)

            F1 = dt1 * M_mass + tht * K_stiff # Main diagonal block system

            A = fd.assemble(F1, bcs=self.block_bcs)

            A_petsc = fd.as_backend_type(A).mat()

            self.first_block = A_petsc # Required for the intermediate step
            self.first_rhs = fd.as_backend_type(self._x[0].vector()).vec()
            self.first_sol = fd.as_backend_type(self._y[0].vector()).vec()

            # PETSc.Sys.Print(f"first_block has size(s): {self.first_block.getSizes()}",comm=COMM_SELF)
            # PETSc.Sys.Print(f"first_rhs has size(s): {self.first_rhs.getSizes()}",comm=COMM_SELF)
            # PETSc.Sys.Print(f"first_sol has size(s): {self.first_rhs.getSizes()}",comm=COMM_SELF)

            nlocal_timesteps_start = 1

        for i in range(nlocal_timesteps_start, self.nlocal_timesteps):
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
        start = time.time()
        B_k, A_k, f_k, B_s, A_s, f_s = self.forward_reduction(pc,x,y) # We still need to deal with A_0. Currently only have A_i, B_i for i = 1,...,n+1
        end = time.time()

        #PETSc.Sys.Print(f"T_rank, spat_rank {self.temporal_rank, self.spatial_rank} spent {end-start}s in forward reduction", comm=COMM_SELF)
        # Block all but one temporal_processor?? Do sequential work
        # PETSc.Sys.Print(f"Ownership range sol_k {sol_k.getOwnershipRanges()}", comm = COMM_SELF)
        # PETSc.Sys.Print(f"Ownership range y {y._vec.getOwnershipRanges()}", comm = COMM_SELF)
        # PETSc.Sys.Print(f"Ownership range x {x._vec.getOwnershipRanges()}", comm = COMM_SELF)

        if self.temporal_rank == 0: # Solve for x0
            local_pc = PETSc.PC().create(comm=self.ensemble.comm)
            local_pc.setType("lu")
            local_pc.setFactorSolverType("mumps")
            local_pc.setOperators(self.first_block)
            local_pc.getFactorMatrix().setMumpsIcntl(24, 1)
            local_pc.getFactorMatrix().setMumpsIcntl(13, 0) # both (13,1) and (13,0) works!
            local_pc.getFactorMatrix().setMumpsCntl(3, 1e-7)
            local_pc.setUp()
            F = local_pc.getFactorMatrix() # F is the factored matrix of A1

            x0 = self.first_rhs.duplicate()
            F.solve(self.first_rhs, x0) # x0 is now solved for temporal rank 0, distributed overs its X spatial ranks.

            # We need to write x0 to the global RHS y. 
            # bcast_field(step, u). step: window index of field to broadcast. u: fd.Function to place field into.

            ranges = x0.getOwnershipRange()
            ranges_global = x0.getOwnershipRanges()
            PETSc.Sys.Print(f"ranges x : {ranges, ranges_global}", comm = COMM_SELF)
            with y.global_vec() as yvec: # write to global array?
                # yvec is <class 'petsc4py.PETSc.Vec'>
                ranges_yvec = yvec.getOwnershipRange()
                ranges_global_yvec = yvec.getOwnershipRanges()
                PETSc.Sys.Print(f"ranges yvec: {ranges_yvec, ranges_global_yvec}", comm = COMM_SELF)
                local_x0 = x0.getArray()
                #local_x0 = np.zeros(len(yvec.array))
                yvec.array[ranges[0]:ranges[1]] = local_x0[:]
                #yvec.array[ranges[0]:ranges[1]] = local_x0
                #PETSc.Sys.Print(f"Checking contents of y {yvec.array}",comm=COMM_SELF)

   

            # PETSc.Sys.Print(f"x0 solved. ownership: {x0.getOwnershipRange()}",comm=COMM_SELF)
        with y.global_vec() as yvec:
            PETSc.Sys.Print(f"Checking contents of y {yvec.view()}",comm=COMM_WORLD)
            yvec.copy(yvec)

        COMM_WORLD.Barrier()
        

        # Begin all processes again
        # self.backward_solution() ...

        # ksp = PETSc.KSP().create()
        # ksp.setOperators(self.first_block) # [A_0, ..., 0] [x_0] = [f_0 - B_0 i.c.]
        # ksp.setOptionsPrefix(self.full_prefix + "cyclic_reduction_")
        # ksp.setFromOptions()
        # ksp.solve(f1,A1inv_f1)

        # First solve for x0 for first proc, send result to next proc. Solve on next proc, send to the one after. Repeat.

        # x = self._x, y = self._y : Types are AllAtOnceCofunction and AllAtOnceFunction respectively.

        # with self._y[0].global_vec_wo() as yvec:
        #     shape = yvec.array.shape
        #     PETSc.Sys.Print(f"y's shape: {shape}", comm = COMM_SELF)
        #     cus_array = np.ones(shape)*3
        #     yvec.array[:] = cus_array

        # for i in range(self.nlocal_timesteps):
        #     self.block_solvers[i].solve()

    @profiler()
    def forward_reduction(self,pc,x,y):

        B_s = []
        A_s = []
        f_s = []
        
        time_steps = self.nlocal_timesteps
        if self.temporal_rank == 0:
            time_steps -= 1
        current_diag = self.diag_matrices.copy()
        current_off_diag = self.off_diag_matrices.copy()
        current_rhs = self.cr_rhs.copy()
        # current_sol = self.cr_sol.copy()

        while time_steps != 1:

            next_diag = []
            next_off_diag = []
            next_rhs = []
            # next_sol = []

            for i in range(0,time_steps,2): # Step twice at a time
                A1 = current_diag[i]
                A2 = current_diag[i+1]

                B1 = current_off_diag[i]
                B2 = current_off_diag[i+1]

                # u1 = current_sol[i]
                # u2 = current_sol[i+1]

                f1 = current_rhs[i]
                f2 = current_rhs[i+1]

                # === Store for backward step (maybe add as self. variables) ===
                B_s.append(B1.copy())
                A_s.append(A1.copy())
                f_s.append(f1.copy())

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
                # next_sol.append(u2.duplicate())  # Placeholder for next solution

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
            # current_sol = next_sol
            time_steps = time_steps//2
        
        # return current_off_diag[0], current_diag[0], current_sol[0], current_rhs[0], B_s, A_s, f_s
        return current_off_diag[0], current_diag[0], current_rhs[0], B_s, A_s, f_s
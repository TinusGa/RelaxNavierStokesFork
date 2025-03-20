import firedrake as fd
from firedrake.petsc import PETSc

from warnings import warn
import numpy as np
from scipy.fft import fft, ifft

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.common import get_option_from_list, get_deprecated_option

from asQ.allatonce.function import time_average as time_average_function
from asQ.preconditioners.base import AllAtOnceBlockPCBase, get_default_options

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

        # building the block problem solvers
        for i in range(self.nlocal_timesteps):

            # the reference states
            u0 = self.state_func[i]
            t0 = self.time[i]

            # the form
            vs = fd.TestFunctions(field_function_space)
            ##########################################
            v = fd.TestFunction(field_function_space)
            u = fd.TrialFunction(field_function_space)

            F1 = dt1 * self.form_mass(u, v) + tht * self.form_function(u, v, t0)
            F2 = -dt1 * self.form_mass(u, v)

            A = fd.assemble(F1)
            B = fd.assemble(F2)

            self.diag_matrices.append(A)
            self.off_diag_matrices.append(B)

            #block_problem = fd.LinearVariationalProblem(A, self._x[i], self._y[i], bcs=self.block_bcs)
            #block_solver = fd.LinearVariationalSolver(block_problem)
            #self.block_solvers.append(block_solver)


            ##########################################
            us = fd.split(u0)

            M = self.form_mass(*us, *vs)
            K = self.form_function(*us, *vs, t0)


            F = dt1*M + tht*K # (dt1*M + tht*K)u^{n+1} - (dt1*M)u^{n} = b^{n+1}

            F1 = dt1*M + tht*K
            F2 = - dt1*M

            A = fd.derivative(F, u0)

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

            # the global index of this block
            ii = aaofunc.transform_index(i, from_range='slice', to_range='window')

            # The block rhs/solution are the timestep i of the
            # input/output AllAtOnceCofunction/Function
            block_problem = fd.LinearVariationalProblem(A, self._x[i], self._y[i],
                                                        bcs=self.block_bcs,
                                                        constant_jacobian=True)

            self.cr_sol.append(self._y[i])
            self.cr_rhs.append(self._x[i])

            block_solver = fd.LinearVariationalSolver(
                block_problem, appctx=appctx_h,
                options_prefix=default_block_prefix+str(ii),
                solver_parameters=default_block_options)

            self.block_solvers.append(block_solver)

        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm)
        self.initialized = True

    
    @profiler()
    def _record_diagnostics(self):
        pass

    @profiler()
    def update(self, pc):
        # """
        # Update the state to linearise around according to aaojacobi_state.
        # """

        # aaofunc = self.aaofunc
        # aaoform = self.aaoform
        # state_func = self.state_func
        # jacobian_state = self.jacobian_state

        # for st, ft in zip(self.time, aaoform.time):
        #     st.assign(ft)

        # if jacobian_state == 'linear':
        #     return

        # elif jacobian_state == 'current':
        #     state_func.assign(aaofunc)

        # elif jacobian_state in ('window', 'slice'):
        #     time_average_function(aaofunc, state_func.initial_condition,
        #                  state_func.uprev, average=jacobian_state)
        #     state_func.assign(state_func.initial_condition)

        #     for t in self.time:
        #         if jacobian_state == 'window':
        #             t.assign(aaoform.t0 + self.dt*(self.ntimesteps + 1)/2)
        #         elif jacobian_state == 'slice':
        #             i1 = aaofunc.transform_index(0, from_range='slice',
        #                                          to_range='window')
        #             t1 = aaoform.t0 + i1*self.dt
        #             t.assign(t1 + self.dt*(self.nlocal_timesteps + 1)/2)

        # elif jacobian_state == 'initial':
        #     state_func.assign(aaofunc.initial_condition)
        #     for t in self.time:
        #         t.assign(self.aaoform.t0)

        # elif jacobian_state == 'reference':
        #     aaofunc.assign(self.jacobian.reference_state)

        # elif jacobian_state == 'user':
        #     pass

        # for block in self.block_solvers:
        #     block.invalidate_jacobian()
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
        self.forward_reduction(pc,x,y)
        

        # self._y.zero()
        # for i in range(self.nlocal_timesteps):
        #     self.block_solvers[i].solve()
    @profiler()
    def forward_reduction(self,pc,x,y):
        
        print(f"len diags {len(self.diag_matrices)}")
        print(f"len off diags {len(self.off_diag_matrices)}")
        for i in range(self.nlocal_timesteps,2): # Step twice at a time
            main_block_i_0 = self.diag_matrices[i].copy()
            main_block_i_1 = self.diag_matrices[i+1].copy()

            lower_block_i_0 = self.off_diag_matrices[i].copy()
            lower_block_i_1 = self.off_diag_matrices[i+1].copy()

            u_i_0 = self.cr_sol[i].copy()
            u_i_1 = self.cr_sol[i+1].copy()

            rhs_i_0 = self.cr_rhs[i].copy()
            rhs_i_1 = self.cr_rhs[i+1].copy()





        pass

class CyclicReductionPC1(AllAtOnceBlockPCBase):
   
    prefix = "circulant_"
    deprecated_prefix = "diagfft_"
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))

    default_alpha = 1e-3

    @profiler()
    def initialize(self, pc):
        # Initialize is called once per MPI processors
        super().initialize(pc, final_initialize=False)

        # these were setup by super
        prefix = self.full_prefix
        aaofunc = self.aaofunc
        appctx = self.appctx

        # basic model function space
        self.blockV = aaofunc.field_function_space # What is block V? Is it V s.t. C_j = V D_j V^{-1} ??

        # Input/Output wrapper Functions for all-at-once residual being acted on
        self.yf = fd.Function(aaofunc.function_space)  # output. So this is b^{tilde} ?? I.e. Au = b^{tilde}
        PETSc.Sys.Print(f"self.yf : {type(self.yf)}")

        self.alpha = get_deprecated_option(
            PETSc.Options().getReal, prefix, self.deprecated_prefix,
            "alpha", default=self.default_alpha)

        dt = self.dt # ∆t
        self.t_average = fd.Constant(self.aaoform.t0 + (self.aaofunc.ntimesteps + 1)*self.dt/2)
        theta = self.theta
        alpha = self.alpha
        nt = self.ntimesteps # N_t

        # Gamma coefficients
        exponents = np.arange(nt)/nt
        self.Gam = alpha**exponents # Gamma = diag( alpha^{n-1}/N_t )

        slice_begin = aaofunc.transform_index(0, from_range='slice', to_range='window')
        slice_end = slice_begin + self.nlocal_timesteps
        self.Gam_slice = self.Gam[slice_begin:slice_end]

        # circulant eigenvalues
        C1col = np.zeros(nt)
        C2col = np.zeros(nt)

        C1col[:2] = np.array([1, -1])/dt
        C2col[:2] = np.array([theta, 1-theta])

        # Build C_1 and C_2 used in C_j = V D_j V^{-1}. 
        # These are just constructed for the first column, i.e., C1col = [1/dt, -1/dt, 0, ... , 0]^T.

        self.D1 = np.sqrt(nt)*fft(self.Gam*C1col)
        self.D2 = np.sqrt(nt)*fft(self.Gam*C2col)

        # D_j = diag( Gamma * Fourier * c_j ), where c_j is the first column of C_j

        # Block system setup
        # First need to build the complex function space version of blockV
        valid_cpx_type = ['vector', 'mixed']

        cpx_type = get_option_from_list(
            prefix, "complex_proxy", valid_cpx_type,
            default_index=0, deprecated_prefix=self.deprecated_prefix)

        if cpx_type == 'vector':
            import asQ.complex_proxy.vector as cpx
            self.cpx = cpx
        elif cpx_type == 'mixed':
            import asQ.complex_proxy.mixed as cpx
            self.cpx = cpx

        self.CblockV = cpx.FunctionSpace(self.blockV)

        # set the boundary conditions to zero for the residual
        self.block_bcs = tuple((cb
                                for bc in self.aaoform.field_bcs
                                for cb in cpx.DirichletBC(self.CblockV, self.blockV,
                                                          bc, 0*bc.function_arg)))

        # function to do global reduction into for average block jacobian
        if self.jacobian_state in ('window', 'slice'):
            self.ureduce = fd.Function(self.blockV)
            self.uwrk = fd.Function(self.blockV)

        # input and output functions to the block solve
        self.block_sol = fd.Function(self.CblockV)
        self.block_rhs = fd.Cofunction(self.CblockV.dual())

        # input for the cofunc rhs map
        self.xtemp = fd.Function(self.CblockV)

        # A place to store the real/imag components of the all-at-once residual after fft
        self.xfi = aaofunc.copy()
        self.xfr = aaofunc.copy()

        # setting up the FFT stuff
        # construct simply dist array and 1d fftn:
        subcomm = Subcomm(self.ensemble.ensemble_comm, [0, 1])
        # dimensions of space-time data in this ensemble_comm
        nlocal = self.blockV.node_set.size
        NN = np.array([nt, nlocal], dtype=int)
        # transfer pencil is aligned along axis 1
        self.p0 = Pencil(subcomm, NN, axis=1)
        # a0 is the local part of our fft working array
        # has shape of (partition/P, nlocal)
        self.a0 = np.zeros(self.p0.subshape, complex)
        self.p1 = self.p0.pencil(0)
        # a0 is the local part of our other fft working array
        self.a1 = np.zeros(self.p1.subshape, complex)
        self.transfer = self.p0.transfer(self.p1, complex)

        # building the Jacobian of the nonlinear term
        # what we want is a block diagonal matrix in the 2x2 system
        # coupling the real and imaginary parts.
        # We achieve this by copying w_all into both components of u0
        # building the nonlinearity separately for the real and imaginary
        # parts and then linearising.
        # This is constructed by cpx.derivative

        #  Building the nonlinear operator
        self.block_solvers = []

        # which form to linearise around
        form_mass = self.form_mass
        form_function = partial(self.form_function, t=self.t_average)

        # Now need to build the block solver
        self.u0 = fd.Function(self.CblockV)  # time average to linearise around

        # user appctx for the blocks
        block_appctx = appctx.get('block_appctx', {})

        # Block i has prefix 'circulant_block_{i}', but we want to be able
        # to set default options for all blocks using 'circulant_block'.
        # LinearVariationalSolver will prioritise options it thinks are from
        # the command line (including those in the `inserted_options` database
        # of the AllAtOnceSolver) over the ones passed to __init__, so we pull
        # the default options off the global dict and pass these explicitly to LVS.

        default_block_prefix = f"{prefix}block_"
        deprecated_block_prefix = f"{self.deprecated_prefix}block_"
        for k, v in PETSc.Options().getAll().items():
            if k.startswith(f"{deprecated_block_prefix}"):
                msg = "Prefix 'diagfft' is deprecated and will be removed in the future. Use 'circulant' instead."
                warn(msg)
                default_block_prefix = deprecated_block_prefix

        default_block_options = get_default_options(
            default_block_prefix, range(self.ntimesteps)) # {'pc_type': 'lu'}
        
        
        # building the block problem solvers. This yields a 2x2 system i think!
        for i in range(self.nlocal_timesteps):
            ii = aaofunc.transform_index(i, from_range='slice', to_range='window')
            d1 = self.D1[ii]
            d2 = self.D2[ii]

            M, D1r, D1i = cpx.BilinearForm(self.CblockV, d1, form_mass, return_z=True)
            K, D2r, D2i = cpx.derivative(d2, form_function, self.u0, return_z=True)

            A = M + K

            PETSc.Sys.Print(f"Type A: {type(A)}")

            # The rhs
            L = self.block_rhs

            # pass parameters into PC:
            appctx_h = {
                "d1": d1,
                "d2": d2,
                "cpx": cpx,
                "uref": self.u0,
                "tref": self.t_average,
                "bcs": self.block_bcs,
                "form_mass": self.form_mass,
                "form_function": self.form_function,
            }

            appctx_h.update(block_appctx)

            block_problem = fd.LinearVariationalProblem(A, L, self.block_sol,
                                                        bcs=self.block_bcs,
                                                        constant_jacobian=True)

            block_solver = fd.LinearVariationalSolver(
                block_problem, appctx=appctx_h,
                options_prefix=default_block_prefix+str(ii),
                solver_parameters=default_block_options)

            self.block_solvers.append(block_solver)

        self.initialized = True

    @profiler()
    def _record_diagnostics(self):
        """
        Update diagnostic information from block linear solvers.

        Must be called exactly once at the end of each apply().
        """
        for i in range(self.nlocal_timesteps):
            its = self.block_solvers[i].snes.getLinearSolveIterations()
            self.block_iterations.dlocal[i] += its

    @profiler()
    def update(self, pc):
        '''
        we need to update u0 according to the diagfft_state option.
        we copy the state into both the "real" and "imaginary" parts
        of u0. this is so that when we linearise the nonlinearity,
        we get an operator that is block diagonal in the 2x2 system
        coupling real and imaginary parts.
        '''
        cpx = self.cpx

        # default to time at centre of window
        self.t_average.assign(self.aaoform.t0 + self.dt*(self.ntimesteps + 1)/2)

        jacobian_state = self.jacobian_state
        if jacobian_state == 'linear':
            return

        elif jacobian_state == 'initial':
            ustate = self.aaofunc.initial_condition
            self.t_average.assign(self.aaoform.t0)

        elif jacobian_state == 'reference':
            ustate = self.jacobian.reference_state

        elif jacobian_state in ('window', 'slice'):
            time_average_function(self.aaofunc, self.ureduce,
                                  self.uwrk, average=jacobian_state)
            ustate = self.ureduce

            if jacobian_state == 'slice':
                i1 = self.aaofunc.transform_index(0, from_range='slice',
                                                  to_range='window')
                t1 = self.aaoform.t0 + i1*self.dt
                self.t_average.assign(t1 + self.dt*(self.nlocal_timesteps + 1)/2)

        cpx.set_real(self.u0, ustate)
        cpx.set_imag(self.u0, ustate)

        for block in self.block_solvers:
            block.invalidate_jacobian()
        return

    @profiler()
    def apply_impl(self, pc, x, y):
        cpx = self.cpx

        # get array of basis coefficients
        with x.global_vec_ro() as xvec:
            parray = xvec.array_r.reshape((self.nlocal_timesteps,
                                           self.blockV.node_set.size))
        # This produces an array whose rows are time slices
        # and columns are finite element basis coefficients

        ######################
        # Diagonalise - scale, transfer, FFT, transfer, Copy
        # Scale
        # is there a better way to do this with broadcasting?
        parray = (1.0+0.j)*(self.Gam_slice*parray.T).T*np.sqrt(self.ntimesteps)
        # transfer forward
        self.a0[:] = parray[:]
        with PETSc.Log.Event("CyclicReduction.CyclicReductionPC.apply.transfer"):
            self.transfer.forward(self.a0, self.a1)

        # FFT
        with PETSc.Log.Event("CyclicReduction.CyclicReductionPC.apply.fft"):
            self.a1[:] = fft(self.a1, axis=0)

        # transfer backward
        with PETSc.Log.Event("CyclicReduction.CyclicReductionPC.apply.transfer"):
            self.transfer.backward(self.a1, self.a0)

        # Copy into xfi, xfr
        parray[:] = self.a0[:]
        with self.xfr.function.dat.vec_wo as v:
            v.array[:] = parray.real.reshape(-1)
        with self.xfi.function.dat.vec_wo as v:
            v.array[:] = parray.imag.reshape(-1)
        #####################

        # Do the block solves

        with PETSc.Log.Event("CyclicReduction.CyclicReductionPC.apply.block_solves"):
            PETSc.Sys.Print(f"nlocal_timesteps: {self.nlocal_timesteps}")
            for i in range(self.nlocal_timesteps):
                # copy the data into solver input
                cpx.set_real(self.xtemp, self.xfr[i])
                cpx.set_imag(self.xtemp, self.xfi[i])

                for cdat, xdat in zip(self.block_rhs.dat, self.xtemp.dat):
                    cdat.data[:] = xdat.data[:]

                # solve the block system
                self.block_sol.zero()
                self.block_solvers[i].solve()

                # copy the data from solver output
                cpx.get_real(self.block_sol, self.xfr[i])
                cpx.get_imag(self.block_sol, self.xfi[i])

        ######################
        # Undiagonalise - Copy, transfer, IFFT, transfer, scale, copy
        # get array of basis coefficients
        with self.xfi.function.dat.vec_ro as v:
            parray = 1j*v.array_r.reshape((self.nlocal_timesteps,
                                           self.blockV.node_set.size))
        with self.xfr.function.dat.vec_ro as v:
            parray += v.array_r.reshape((self.nlocal_timesteps,
                                         self.blockV.node_set.size))
        # transfer forward
        self.a0[:] = parray[:]
        with PETSc.Log.Event("CyclicReduction.CyclicReductionPC.apply.transfer"):
            self.transfer.forward(self.a0, self.a1)

        # IFFT
        with PETSc.Log.Event("CyclicReduction.CyclicReductionPC.apply.fft"):
            self.a1[:] = ifft(self.a1, axis=0)

        # transfer backward
        with PETSc.Log.Event("CyclicReduction.CyclicReductionPC.apply.transfer"):
            self.transfer.backward(self.a1, self.a0)
        parray[:] = self.a0[:]

        # scale
        parray = ((1.0/self.Gam_slice)*parray.T).T
        # Copy into xfi, xfr

        with y.global_vec_wo() as yvec:
            yvec.array[:] = parray.reshape(-1).real
        ################
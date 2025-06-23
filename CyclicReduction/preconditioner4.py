import numpy as np
from time import time

import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.preconditioners import ASMPatchPC
from firedrake.logging import warning
import firedrake.preconditioners 


from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.preconditioners.base import AllAtOnceBlockPCBase
from asQ.parallel_arrays import SharedArray
from asQ.allatonce import LinearSolver, time_average

from asQ import (
    AllAtOnceFunction,
    AllAtOnceCofunction,
    AllAtOnceForm,
    AllAtOnceSolver,
)


__all__ = ['CyclicReductionPC4','ApproxCyclicReductionPC4','asQMGPC2', 'DirectSolvePC']

def order_points(mesh_dm, points, ordering_type, prefix):
    '''Order the points (topological entities) of a patch based
    on the adjacency graph of the mesh.

    :arg mesh_dm: the `mesh.topology_dm`
    :arg points: array with point indices forming the patch
    :arg ordering_type: a `PETSc.Mat.OrderingType`
    :arg prefix: the prefix associated with additional ordering options

    :returns: the permuted array of points
    '''
    # Order points by decreasing topological dimension (interiors, faces, edges, vertices)
    points = points[::-1]
    if ordering_type == "natural":
        return points
    subgraph = [np.intersect1d(points, mesh_dm.getAdjacency(p), return_indices=True)[1] for p in points]
    ia = np.cumsum([0] + [len(neigh) for neigh in subgraph]).astype(PETSc.IntType)
    ja = np.concatenate(subgraph).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ((len(points), )*2, csr=(ia, ja, np.ones(ja.shape, PETSc.RealType)), comm=PETSc.COMM_SELF)
    A.setOptionsPrefix(prefix)
    rperm, cperm = A.getOrdering(ordering_type)
    indices = points[rperm.getIndices()]
    A.destroy()
    rperm.destroy()
    cperm.destroy()
    return indices

def order_points_vanka(mesh_dm, points, ordering_type, prefix):
    '''Order a the points (topological entities) of a patch based on the adjacency graph of the mesh.
    :arg mesh_dm: the `mesh.topology_dm`
    :arg points: array with point indices forming the patch
    :arg ordering_type: a `PETSc.Mat.OrderingType`
    :arg prefix: the prefix associated with additional ordering options
    :returns: the permuted array of points                                                                        
    '''
    if ordering_type == "natural":
        return points
    subgraph = [np.intersect1d(points, mesh_dm.getAdjacency(p), return_indices=True)[1] for p in points]
    ia = np.cumsum([0] + [len(neigh) for neigh in subgraph]).astype(PETSc.IntType)
    ja = np.concatenate(subgraph).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ((len(points), )*2, csr=(ia, ja, np.ones(ja.shape, PETSc.RealType)), comm=PETSc.COMM_SELF)
    A.setOptionsPrefix(prefix)
    rperm, _ = A.getOrdering(ordering_type)
    A.destroy()
    return points[rperm.getIndices()]

class ASMVankaStarPC(ASMPatchPC):
    '''Patch-based PC using closure of star of mesh entities implemented as an
    :class:`ASMPatchPC`.
    ASMVankaStarPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the closure of the star of the mesh entity
    specified by `pc_vanka_construct_dim` (or codim).
    This version includes the star of the "exclude_subfunctions" in the patch
    '''

    _prefix = "pc_vankastar_"

    def get_patches(self, V):
        mesh = V._mesh
        mesh_dm = mesh.topology_dm
        if mesh.layers:
            warning("applying ASMVankaPC on an extruded mesh")
        
        # Obtain the topological entities to use to construct the stars
        depth = PETSc.Options().getInt(self.prefix + "construct_dim", default=-1)
        height = PETSc.Options().getInt(self.prefix + "construct_codim", default=-1)
        if (depth == -1 and height == -1) or (depth != -1 and height != -1):
            raise ValueError(f"Must set exactly one of {self.prefix}construct_dim or {self.prefix}construct_codim")

        exclude_subfunctions = [int(subspace) for subspace in PETSc.Options().getString(self.prefix+"exclude_subfunctions", default="-1").split(",")]
        ordering = PETSc.Options().getString(self.prefix+"mat_ordering_type", default="natural")
        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = []
        for (i, W) in enumerate(V):
            V_local_ises_indices.append(V.dof_dset.local_ises[i].indices)

        # Build index sets for the patches
        ises = []
        if depth != -1:
            (start, end) = mesh_dm.getDepthStratum(depth)
        else:
            (start, end) = mesh_dm.getHeightStratum(height)

        for seed in range(start, end):
            # Only build patches over owned DoFs
            if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
                continue

            # Create point list from mesh DM
            star, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
            pt_array_star = order_points_vanka(mesh_dm, star, ordering, self.prefix)
            
            pt_array_vanka = set()
            for pt in star.tolist():
                closure, _ = mesh_dm.getTransitiveClosure(pt, useCone=True)
                pt_array_vanka.update(closure.tolist())

            pt_array_vanka = order_points_vanka(mesh_dm, pt_array_vanka, ordering, self.prefix)
            # Get DoF indices for patch
            indices = []
            for (i, W) in enumerate(V):
                section = W.dm.getDefaultSection()
                if i in exclude_subfunctions:
                    loop_list = pt_array_star
                else:
                    loop_list = pt_array_vanka
                for p in loop_list:
                    dof = section.getDof(p)
                    if dof <= 0:
                        continue
                    off = section.getOffset(p)
                    # Local indices within W
                    W_indices = slice(off*W.value_size, W.value_size * (off + dof))
                    indices.extend(V_local_ises_indices[i][W_indices])
            iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
            ises.append(iset)

        return ises

class ASMStarPC(ASMPatchPC):
    '''Patch-based PC using Star of mesh entities implmented as an
    :class:`ASMPatchPC`.

    ASMStarPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the topological star of the mesh entity
    specified by `pc_star_construct_dim`.
    '''

    _prefix = "pc_star_"

    def get_patches(self, V):
        mesh = V._mesh
        mesh_dm = mesh.topology_dm
        if mesh.cell_set._extruded:
            warning("applying ASMStarPC on an extruded mesh")

        # Obtain the topological entities to use to construct the stars
        opts = PETSc.Options(self.prefix)
        depth = opts.getInt("construct_dim", default=0)
        ordering = opts.getString("mat_ordering_type", default="natural")
        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = tuple(iset.indices for iset in V.dof_dset.local_ises)

        # Build index sets for the patches
        ises = []
        (start, end) = mesh_dm.getDepthStratum(depth)
        for seed in range(start, end):
            # Only build patches over owned DoFs
            if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
                continue

            # Create point list from mesh DM
            pt_array, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
            pt_array = order_points(mesh_dm, pt_array, ordering, self.prefix)

            # Get DoF indices for patch
            indices = []
            for (i, W) in enumerate(V):
                section = W.dm.getDefaultSection()
                for p in pt_array.tolist():
                    dof = section.getDof(p)
                    if dof <= 0:
                        continue
                    off = section.getOffset(p)
                    # Local indices within W
                    W_indices = slice(off*W.block_size, W.block_size * (off + dof))
                    indices.extend(V_local_ises_indices[i][W_indices])
            iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
            ises.append(iset)
        return ises

class asQMGPC2(AllAtOnceBlockPCBase):
    prefix = 'opts_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))

    @profiler()
    def initialize(self, pc):
        start = time()
        pc.setOptionsPrefix('asQMGPC_')
        super().initialize(pc, final_initialize=False)

        self.jacobian_state = self.jacobian.jacobian_state
        
        self.sub_solver_parameters = PETSc.Options(self.full_prefix).getAll()

        self.field_function_space = self.aaofunc.field_function_space
        self.function_space = self.aaofunc.function_space

        self.mesh_hierarchy = self.appctx.get('mesh_hierarchy')
        self.function_spaces = self.appctx.get('function_spaces')
        self.bcs_list = self.appctx.get('bcs_list')

        self.us = [] # List of solutions as AllAtOnceFunction's
        self.rs = [] # List of residuals as AllAtOnceCofunction's

        self.forms = []
        self.solvers = []
        self.corrections = []

        # Create AllAtOnce objects for each function space in the hierarchy, except the coarsest one
        for V, bcs in zip(self.function_spaces[1:],self.bcs_list[1:]):

            u = AllAtOnceFunction(self.ensemble, self.time_partition, V)
            r = AllAtOnceCofunction(self.ensemble, self.time_partition, u.field_function_space.dual())
            form = AllAtOnceForm(u, self.dt, self.theta,
                                 self.form_mass, self.form_function,
                                 bcs=bcs)  
            solver = LinearSolver(form, solver_parameters=self.sub_solver_parameters, 
                                  options_prefix=self.full_prefix+'pc_python_',)
            
            self.us.append(u)
            self.rs.append(r)
            self.forms.append(form)
            self.solvers.append(solver)
            self.corrections.append(u.copy())


        # Reverse the order so that we get the finest mesh is first
        self.function_spaces = list(reversed(self.function_spaces))
        self.us = list(reversed(self.us))
        self.rs = list(reversed(self.rs))
        self.forms = list(reversed(self.forms))
        self.solvers = list(reversed(self.solvers))
        self.corrections = list(reversed(self.corrections))

        # Transfer manager for restriction and prolongation
        self.tm = fd.TransferManager(use_averaging=False)

        # Set-up the coarse solver
        coarse_parameters = {
            'snes_type': 'ksponly',
            'mat_type': 'matfree',
            'ksp_type': 'richardson',
            # 'ksp_monitor_true_residual': None,
            'ksp_rtol': 1e-14,
            'pc_type': 'python',
            'pc_python_type': 'asQ.CirculantPC',
            'circulant_block': {'pc_type': 'lu'},
            'circulant_alpha': 1e-4}

        coarse_prefix = 'coarse_'

        V_c, bcs_c = self.function_spaces[-1], self.bcs_list[-1]
        u = AllAtOnceFunction(self.ensemble, self.time_partition, V_c)
        r = AllAtOnceCofunction(self.ensemble, self.time_partition, u.field_function_space.dual())
        form = AllAtOnceForm(u, self.dt, self.theta,
                                self.form_mass, self.form_function,
                                bcs=bcs_c)  
        solver = LinearSolver(form, solver_parameters=coarse_parameters, 
                                options_prefix=coarse_prefix)
        self.us.append(u)  
        self.rs.append(r)
        self.forms.append(form)
        self.solvers.append(solver)
        self.corrections.append(u.copy())

        self.initialized = True

    def _record_diagnostics(self):
        pass
    
    @profiler()
    def update(self, pc):
        # We need to ping the LinearSolvers to have them update their Jacobian based on the current state,
        # i.e. we need this update() to trigger the update() in CyclicReductionPC3
        PETSc.Sys.Print(f"asQMGPC2: Updating solvers with jacobian state: {self.jacobian_state}")
        for solver in self.solvers:
            solverPC = solver.ksp.getPC()
            pcctx = solverPC.getPythonContext()
            # Add the jacobian state to the appctx
            pcctx.jac_state = self.jacobian_state
            pcctx.setUp(solverPC)

    @profiler()
    def apply_impl(self, pc, x, y):
        u_fine = self.us[0]  
        r_fine = self.rs[0]  

        u_fine.assign(y)  
        r_fine.assign(x) 

        # Coarsening
        for i in range(len(self.mesh_hierarchy) - 1):
            u_coarse = self.us[i+1]  
            r_coarse = self.rs[i+1]

            # PETSc.Sys.Print(f"Pre-smooting")
            solver = self.solvers[i]     
            solver.solve(r_fine, u_fine)
            
            # Transfer the residual to the next coarser mesh
            for j in range(self.nlocal_timesteps):
                self.tm.restrict(r_fine[j], r_coarse[j])
            
            # u_coarse and r_coarse will be the fine solution and residual in the next iteration
            u_fine = u_coarse
            r_fine = r_coarse
        
        # Coarsest solve
        solver = self.solvers[-1]
        solver.solve(r_fine, u_fine)

        u_coarse = u_fine
        r_coarse = r_fine

        # Refine the solution back to the finest mesh
        # First fine should be the second to last elements in the lists, and we want to loop backwards
        for i in range(len(self.mesh_hierarchy) - 2, -1, -1):
            u_coarse = self.us[i+1]
            u_fine = self.us[i]  
            r_fine = self.rs[i]  
            correction = self.corrections[i] 

            # Transfer the solution back to the finer mesh
            for j in range(self.nlocal_timesteps):
                self.tm.prolong(u_coarse[j], correction[j])
            
            u_fine.axpy(1.0, correction)  # Update the fine solution with the correction

            # PETSc.Sys.Print(f"Post-smooting")
            solver = self.solvers[i]
            solver.solve(r_fine, u_fine)


        y.assign(self.us[0])  # Final solution is in the finest mesh's AllAtOnceFunction
        

class CyclicReductionPC4(AllAtOnceBlockPCBase):

    prefix = 'cr_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))

    @profiler()
    def initialize(self,pc):
        super().initialize(pc, final_initialize=False)

        aaofunc = self.aaofunc
        # aaofunc.update_time_halos()
        self.state_func = aaofunc.copy()
        field_function_space = aaofunc.field_function_space

        self.spatial_rank = self.ensemble.comm.rank
        self.temporal_rank = self.ensemble.ensemble_comm.rank

        self.block_solvers = []

        self.block_bcs = tuple(
            fd.DirichletBC(field_function_space,
                           0*bc.function_arg,
                           bc.sub_domain)
            for bc in self.aaoform.field_bcs)

        dt1 = fd.Constant(1/self.dt)
        tht = fd.Constant(self.theta)

        # Get solver parameters from the options
        first_solve_parameters = PETSc.Options(self.full_prefix+'first_solve_parameters_').getAll()
        bigstep_solve_parameters = PETSc.Options(self.full_prefix+'bigstep_solve_parameters_').getAll()
        forward_solve_parameters = PETSc.Options(self.full_prefix+'forward_solve_parameters_').getAll()

        PETSc.Sys.Print(f"big solve parameters: {bigstep_solve_parameters}")

        # Construct the solvers for each step in forward substitution
        for i in range(self.nlocal_timesteps):
            # State and time to linearise around
            u0 = self.state_func[i] 
            t0 = self.time[i]

            # Symbolic test and trial functions for defining forms
            vs = fd.TestFunctions(field_function_space)
            gs = fd.TrialFunctions(field_function_space)
            
            # The residual F (a linear form) is built using the concrete state u0
            us = fd.split(u0)
            M_form = self.form_mass(*us, *vs)
            K_form = self.form_function(*us, *vs, t0)
            F = dt1*M_form + tht*K_form

            # The mass matrix operator L (a bilinear form) is built using a TrialFunction
            M_op = self.form_mass(*gs, *vs)

            # These form the linear system L * u[i] + D * u[i+1] = f[i+1]
            L = -dt1*M_op
            D = fd.derivative(F, u0)

            if i == 0 and self.temporal_rank == 0:
                # First slice, first time-step. Only a diagonal block exist here
                residual = self._x[i] 
                solver_parameters = first_solve_parameters
            elif i == 0:
                # First time-step of a slice, but not the first slice. Need to use the last time-step from the previous slice
                residual = self._x[i] - L * self._y.uprev
                solver_parameters = forward_solve_parameters
            else:
                # Else just use previous time-step local to this slice
                residual = self._x[i] - L * self._y[i-1]
                solver_parameters = forward_solve_parameters
            
            block_problem = fd.LinearVariationalProblem(D, residual, self._y[i], 
                                                        bcs=self.block_bcs, 
                                                        constant_jacobian=True)
            
            block_solver = fd.LinearVariationalSolver(block_problem,
                                                      options_prefix=self.full_prefix + f'timestep_{i}_',
                                                      solver_parameters=solver_parameters)
            self.block_solvers.append(block_solver)
        
        # Now we compute a BIG STEP. 
        # We approximate the solution at the last time-step of the slice using the last time-step of the previous slice.
        # For the first time-slice, we define the first time-step as the last time-step of the 'previous slice', even though 
        # there is no previous slice.
        offset = 1 if self.temporal_rank == 0 else 0
        step_size = self.nlocal_timesteps - offset

        bigdt1 = fd.Constant(step_size/(self.dt))

        u0 = self.state_func[-1]
        t0 = self.time[-1]
        us = fd.split(u0)

        vs = fd.TestFunctions(field_function_space)
        gs = fd.TrialFunctions(field_function_space)
        
        M_form = self.form_mass(*us, *vs)
        K_form = self.form_function(*us, *vs, t0)
        F = bigdt1*M_form + bigdt1*K_form
        M_op = self.form_mass(*gs, *vs)

        L_big = -bigdt1*M_op
        D_big = fd.derivative(F, u0)
        
        residual = self._x[-1] - L_big * self._y.uprev
        
        self.big_problem = fd.LinearVariationalProblem(D_big, residual, self._y[-1],
                                                        bcs=self.block_bcs, 
                                                        constant_jacobian=True)
        
        self.big_solver = fd.LinearVariationalSolver(self.big_problem,
                                                     options_prefix=self.full_prefix + 'big_step_',
                                                     solver_parameters=bigstep_solve_parameters)

        self.initialized = True

    @profiler()
    def _record_diagnostics(self):
        pass

    @profiler()
    def apply_impl(self, pc, x, y):
        self._y.zero()

        if self.temporal_rank == 0:
            # self.first_solver.solve()
            self.block_solvers[0].solve()
        
        # Communicate across time-slices to corresponding spatial subcommunicators
        size = self.ensemble.ensemble_comm.size
        rank = self.ensemble.ensemble_comm.rank
        dst = (rank+1) % size
        src = (rank-1) % size

        if self.temporal_rank > 0:
            # If this is not the first temporal rank, we must wait for the previous rank to send its solution
            self.ensemble.recv(self._y.uprev, source=src, tag=src)
        else:
            # If this is the first temporal rank, we set uprev as the first time-step, and do the big step starting from the second time-step
            self._y.uprev.assign(self._y[0])
        
        self.big_solver.solve()
        self._y.unext.assign(self._y[-1]) 
        
        if self.temporal_rank < size - 1:
            # All ranks except the last one send their solution to the next rank
            self.ensemble.send(self._y.unext, dest=dst, tag=rank)
        
        offset = 1 if self.temporal_rank == 0 else 0

        for i in range(offset, self.nlocal_timesteps):
            # Solve the block problem for each time-step. Each slice now has a uprev to reference and can work independently
            self.block_solvers[i].solve()
    
            
    @profiler()
    def update(self, pc):
        """
        Update the state to linearise around according to aaojacobi_state.
        """
        aaofunc = self.aaofunc
        aaoform = self.aaoform
        state_func = self.state_func
        jacobian_state = self.jacobian.jacobian_state
        PETSc.Sys.Print(f"Updating state to linearise around: {jacobian_state}")

        for st, ft in zip(self.time, aaoform.time):
            st.assign(ft)

        if jacobian_state == 'linear':
            return

        elif jacobian_state == 'current':
            state_func.assign(aaofunc)

        elif jacobian_state in ('window', 'slice'):
            time_average(aaofunc, state_func.initial_condition,
                         state_func.uprev, average=jacobian_state)
            state_func.assign(state_func.initial_condition)

            for t in self.time:
                if jacobian_state == 'window':
                    t.assign(aaoform.t0 + self.dt*(self.ntimesteps + 1)/2)
                elif jacobian_state == 'slice':
                    i1 = aaofunc.transform_index(0, from_range='slice',
                                                 to_range='window')
                    t1 = aaoform.t0 + i1*self.dt
                    t.assign(t1 + self.dt*(self.nlocal_timesteps + 1)/2)

        elif jacobian_state == 'initial':
            state_func.assign(aaofunc.initial_condition)
            for t in self.time:
                t.assign(self.aaoform.t0)

        elif jacobian_state == 'reference':
            aaofunc.assign(self.jacobian.reference_state)

        elif jacobian_state == 'user':
            pass

        return   

class DirectSolvePC(AllAtOnceBlockPCBase):

    prefix = 'direct_opts_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))

    @profiler()
    def initialize(self,pc):
        super().initialize(pc, final_initialize=False)
        self.some_parameters_sent_by_KSP = PETSc.Options(self.full_prefix).getAll()

        # Obtain the time-slice local Jacobian matrix and the contribution from the previous time-slice. self.prevmat is 'None' for temporal rank 0.
        self.mat = self.jacobian.mat 
        self.prevmat = self.jacobian.prevmat 

        self.field_function_space = self.aaofunc.field_function_space
        self.function_space = self.aaofunc.function_space

        self.temporal_rank = self.ensemble.ensemble_comm.rank
        self.spatial_rank = self.ensemble.comm.rank
    
    @profiler()
    def apply_impl(self, pc, x, y):
        with x.global_vec_ro() as xvec, y.global_vec_wo() as yvec:
            # if self.prevmat is None:
                # F = self.get_factored_matrix(self.mat, self.function_space.mesh().comm)
                # F.solve(xvec, yvec)
            PETSc.Sys.Print(f"Rank {fd.COMM_WORLD.rank}. self.mat size : {self.mat.getSize()}, ownership range: {self.mat.getOwnershipRange()}",comm=fd.COMM_SELF)
            PETSc.Sys.Print(f"Rank {fd.COMM_WORLD.rank}. xvec,yvec size {xvec.getSize(),yvec.getSize()}, with ranges {xvec.getOwnershipRange(),yvec.getOwnershipRange()} ",comm=fd.COMM_SELF)
            # else:
            #     y.update_time_halos(blocking=True)
            #     F = self.get_factored_matrix(self.prevmat, self.ensemble.comm)
            #     rhs = xvec.duplicate()



            
        
        y.update_time_halos(blocking=True)
        y.uprev # Prev time-step contribution, if any.

        PETSc.Sys.Print(f"apply in DIRECTSOLVEPC")

        with y.global_vec_wo() as yvec:
            # yvec is a PETSc Vec. Solution
            yvec.set(0.0)
            pass
    
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
    def _record_diagnostics(self):
        pass

    @profiler()
    def update(self, pc):
        """
        No need to update. The method should only be called once per rank.
        """
        pass

    

class ApproxCyclicReductionPC4(CyclicReductionPC4):
    """
    Approximate cyclic reduction preconditioner.
    
    This class implements an approximate cyclic reduction preconditioner for
    solving linear systems of equations. It is a subclass of the CyclicReductionPC
    class and provides additional functionality for approximating the solution
    using a reduced set of variables.
    
    Parameters
    ----------
    A : PETSc.Mat
        The matrix to be preconditioned.
    comm : PETSc.Comm
        The MPI communicator over which to create the preconditioner.
    """
    @profiler()
    def initialize(self, pc):
        super().initialize(pc)

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
            F = self.get_factored_matrix(self.diag_matrices[0].copy(), self.ensemble.comm)
            u1 = self.rhs[0].duplicate()
            F.solve(self.rhs[0], u1)

            local_u1 = u1.getArray()
            self.a0[0,:] = local_u1[:]
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
        
        # ---------------------------------------------------------------------------
        # FORWARD REDUCTION
        # ---------------------------------------------------------------------------
        # start = time()
        L, D, f = self.approx_forward_reduction(self.lower_diag_matrices,
                                         self.diag_matrices,
                                         self.rhs) 
        # PETSc.Sys.Print(f"Time taken for approx_forward_reduction: {time() - start} seconds")

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
        F = self.get_factored_matrix(D, self.ensemble.comm)
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
        self.forward_substitution(self.lower_diag_matrices, self.diag_matrices, self.rhs, u_prev, y)
    
    @profiler()
    def approx_forward_reduction(self, lower_diag, main_diag, rhs):
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

        if len(main_diag) > 1: # This processor owns more than one timestep, so we reduce
            # Reduce onto these variables. L and D are the lower diagonal and main diagonal
            # matrices respectively of type 'mpiaij'.
            L, D, f = lower_diag[offset].copy(), main_diag[offset].copy(), rhs[offset].copy()
            for i in range(offset + 1, self.nlocal_timesteps):
                L_next = lower_diag[i].copy()
                D_next = main_diag[i].copy()
                f_next = rhs[i].copy()

                # Factor using LU decomposition with MUMPS
                D_factored = self.get_factored_matrix(D, self.ensemble.comm)

                # Compute f <- L_next * D^{-1} * f - f_next
                D_inv_f = f.duplicate()
                D_factored.solve(f, D_inv_f) # D^{-1} * f
                f_next.scale(-1.0) # f_next <- -f_next
                L_next.multAdd(D_inv_f, f_next, f) # f <- L_next * D^{-1} * f - f_next

                # Destruction
                D_factored.destroy()
                D_inv_f.destroy()

                L_next.destroy()
                D_next.destroy()
                f_next.destroy()

            L.destroy()
            D.destroy()
            # f is now correct rhs

            # Let's do a BIIIG timestep

            # How many big steps should we do?
            factor = fd.Constant(len(main_diag)-1)

            t = self.time[0]
            v = fd.TestFunction(self.field_function_space)
            u = fd.TrialFunction(self.field_function_space)
            M = self.form_mass(u, v)
            K = self.form_function(u, v, t)
            D = fd.assemble(factor*self.dt1*M + self.theta*K, bcs=self.block_bcs).petscmat
            L = fd.assemble(-1*factor*self.dt1*M, bcs=self.block_bcs).petscmat
        else: 
            # If we only own one timestep, we just return the first diagonal and lower diagonal matrices
            L, D, f = lower_diag[0].copy(), main_diag[0].copy(), rhs[0].copy()

        return L, D, f
    
    @profiler()
    def forward_substitution(self, lower_diag, main_diag, rhs, u_prev, y):
        """
        Perform the back substitution step of the cyclic reduction algorithm.
        """
        offset = 1 if self.temporal_rank == 0 else 0 # Since temporal rank 0 is offset from other ranks by 1. It has 1 more row than other ranks
        L, D = lower_diag[0].copy(), main_diag[0].copy()
        F = self.get_factored_matrix(D, self.ensemble.comm)

        for i in range(offset, self.nlocal_timesteps):

            # L = lower_diag[i].copy()
            # D = main_diag[i].copy()
            f = rhs[i].copy()

            # Solve: u_next = D^{-1} (f - L * u_prev)
            tmp = f.duplicate()
            f.scale(-1.0) # Set f <- -f
            L.multAdd(u_prev, f, tmp)  # rhs_tmp <- L * u_prev - f
            tmp.scale(-1.0) # rhs_tmp <- f - L * u_prev
            u_next = tmp.duplicate()
            F.solve(tmp, u_next)

            self.a0[i,:] = u_next.getArray()[:]
            with y.global_vec_wo() as yvec:
                yvec.array[:] = self.a0.reshape(-1)[:]
            
            u_prev = u_next.copy()

            # Destruction
            # L.destroy()
            # D.destroy()
            # f.destroy()
            # tmp.destroy()
            # F.destroy()
            # u_next.destroy()
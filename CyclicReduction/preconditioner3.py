import numpy as np
from time import time

import firedrake as fd
from firedrake.petsc import PETSc
from CyclicReduction.ASMPatchPCs import ASMStarPC

from asQ import (
    AllAtOnceFunction,
    AllAtOnceCofunction,
    AllAtOnceForm,
)

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.preconditioners.base import AllAtOnceBlockPCBase
from asQ.allatonce import LinearSolver, time_average


__all__ = ['CyclicReductionPC3','ApproxCyclicReductionPC3','asQMGPC', 'DirectSolvePC']


class asQMGPC(AllAtOnceBlockPCBase):
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
                                  options_prefix=self.full_prefix, appctx=self.appctx)
            
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
            'ksp_rtol': 1e-12,
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
        # PETSc.Sys.Print("Updating in asQMGPC")
        if self.jacobian_state != 'linear':
            for solver in self.solvers:
                solverPC = solver.ksp.getPC()
                pcctx = solverPC.getPythonContext()
                # Add the jacobian state to the appctx
                pcctx.jac_state = self.jacobian_state
                pcctx.setUp(solverPC)
        else:
            # If the jacobian state is linear, we don't need to update the solvers
            # but we still need to set the jacobian state in the appctx
            for solver in self.solvers:
                solverPC = solver.ksp.getPC()
                pcctx = solverPC.getPythonContext()
                pcctx.jac_state = self.jacobian_state

    @profiler()
    def apply_impl(self, pc, x, y):
        # x is the input residual from the outer KSP solver (e.g., FGMRES)
        # y is the output vector where we will compute the correction
        # For a preconditioner, the initial guess for the correction is always zero.
        y.zero()

        # --- SETUP ---
        # Set the residual on the finest level (level 0) to be the input vector x.
        # The solution on the finest level (the correction we are computing) starts at 0.
        self.rs[0].assign(x)
        self.us[0].zero() # us[0] will accumulate the correction on the finest level

        # --- GO DOWN THE V-CYCLE (Pre-smoothing and Restriction) ---
        for i in range(len(self.mesh_hierarchy) - 1):
            # Get objects for the current "fine" level i
            u_i = self.us[i]
            r_i = self.rs[i]
            smoother_i = self.solvers[i]
            jacobian_i = smoother_i.jacobian

            # 1. PRE-SMOOTHING
            # Apply the pre-configured smoother to get an initial approximation for the correction.
            # This updates u_i in place.
            smoother_i.solve(r_i, u_i)

            # 2. COMPUTE THE POST-SMOOTHING RESIDUAL: r_new = r_i - A_i * u_i
            # This is the first critical fix.
            r_after_smoothing = r_i.copy() # Make a temporary copy to do the math
            Au = u_i.copy() # Temporary vector to store the mat-vec product
            with u_i.global_vec_ro() as u_i_vec, Au.global_vec_wo() as Au_vec:
                jacobian_i.mult(None, u_i_vec, Au_vec)
            
            with r_after_smoothing.global_vec_ro() as r_after_smoothing_vec, Au.global_vec_ro() as Au_vec:
                r_after_smoothing_vec.axpy(-1.0, Au_vec)  # r_after_smoothing = r_i - A_i * u_i
            # r_after_smoothing.axpy(-1.0, Au) # r_after_smoothing = r_i - A*u_i

            # 3. RESTRICT THE NEW RESIDUAL
            # Get the residual vector for the next coarser level
            r_coarse = self.rs[i+1]
            # Restrict the correct residual (r_after_smoothing)
            for j in range(self.nlocal_timesteps):
                self.tm.restrict(r_after_smoothing[j], r_coarse[j])
            
            # The next level down now has the correct residual to work on.

        # --- COARSEST GRID SOLVE ---
        # The initial guess for the coarsest correction must be zero.
        u_coarse = self.us[-1]
        r_coarse = self.rs[-1]
        u_coarse.zero()
        
        coarse_solver = self.solvers[-1]
        # This is an "exact" solve as configured in your initialize method
        coarse_solver.solve(r_coarse, u_coarse)

        # --- GO UP THE V-CYCLE (Correction and Post-smoothing) ---
        for i in range(len(self.mesh_hierarchy) - 2, -1, -1):
            # Get objects for the current "fine" level i and the "coarse" level i+1
            u_i = self.us[i]
            r_i = self.rs[i]
            # This now holds the computed correction from the level below
            u_coarse = self.us[i+1]
            smoother_i = self.solvers[i]
            
            # 4. PROLONGATION AND CORRECTION
            # This is the second critical fix.
            correction = self.corrections[i] # Use a temporary vector
            for j in range(self.nlocal_timesteps):
                self.tm.prolong(u_coarse[j], correction[j])
            
            # Add the coarse grid correction to the existing fine grid solution
            u_i.axpy(1.0, correction)

            # 5. POST-SMOOTHING
            # Smooth the newly corrected solution u_i
            smoother_i.solve(r_i, u_i)
            
        # The final computed correction is now in self.us[0]. Assign it to the output vector y.
        y.assign(self.us[0])

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

            # Pre-smooting and residual update
            solver = self.solvers[i]     
            solver.solve(r_fine, u_fine)
            jacobian = solver.jacobian
            tmp = r_fine.copy()
            with tmp.global_vec_ro() as tmpvec, u_fine.global_vec_ro() as uvec:
                jacobian.mult(None, uvec, tmpvec)
            tmp.scale(-1.0)
            r_fine.axpy(1.0, tmp)
            
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
        

class CyclicReductionPC3(AllAtOnceBlockPCBase):

    prefix = 'cr_opts_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))

    @profiler()
    def initialize(self,pc):
        super().initialize(pc, final_initialize=False)
        # start = time()
        self.patch_parameters = PETSc.Options(self.full_prefix).getAll()

        # Obtain the time-slice local Jacobian matrix and the contribution from the previous time-slice. self.prevmat is 'None' for temporal rank 0.
        self.mat = self.jacobian.mat 
        self.prevmat = self.jacobian.prevmat 
        self.state_func = self.aaofunc.copy()

        # We have to cheese the system here a bit as all ranks must have a prevmat.
        if self.prevmat is None:
            self.prevmat = PETSc.Mat().create(comm=fd.COMM_SELF)
            self.prevmat.setSizes([[1, 1], [1, 1]])
            self.prevmat.setType('mpiaij') 
            self.prevmat.setUp()
            self.prevmat.setValue(0, 0, 0)
            self.prevmat.assemble()
                
        # The function space for the time-slice
        self.function_space = self.aaofunc.function_space 
        self.field_function_space = self.aaofunc.field_function_space

        # This processor's spatial and temporal rank
        self.spatial_rank = self.ensemble.comm.rank
        self.temporal_rank = self.ensemble.ensemble_comm.rank
        
        # Initialize the desired patch PC. 
        # TODO: Currently only supports 'star' patch type. Should make this dynamic through e.g. appctx.
        # self.star_pc = ASMStarPC()

        self.patch_pc = self.appctx.get('patch_class', ASMStarPC)()
        self.patch_pc.prefix = self.full_prefix

        # Get patch IS in rank local numbering, split each into nlocal_timesteps
        rank_local_isets = self.patch_pc.get_patches(self.function_space)
        rank_local_isets = list(np.array_split(iset.getIndices(), self.nlocal_timesteps) for iset in rank_local_isets)
        rank_local_isets = list(iset for iset_list in rank_local_isets for iset in iset_list)

        # Local-to-global mapping from rank local numbering to time-slice local numbering
        lgmap = self.function_space.dof_dset.lgmap 

        # Slice local IS's and arrays. We need IS's for the submatrices, and arrays for the scatter/communication operations.
        self.slice_local_arrays = list(lgmap.apply(array) for array in rank_local_isets)  
        self.slice_local_isets = list(PETSc.IS().createGeneral(array, comm=fd.COMM_SELF) for array in self.slice_local_arrays)

        # We have a dependency in time, which means the indices of the last time-step of each patch 
        # corresponds to the columns we need to index on the next time-slice. Normally, we would have to communicate these indices
        # but due to how asQ builds forms we can just use the field_function_space to get the indices of the patches

        if self.temporal_rank > 0:
            self.isrows = self.keep_every_n(self.slice_local_isets, self.nlocal_timesteps,0)
            field_patches = self.patch_pc.get_patches(self.field_function_space)
            field_lgmap = self.field_function_space.dof_dset.lgmap
            self.iscols = list(field_lgmap.applyIS(iset) for iset in field_patches)
        else:
            # Make IS rows and columns with (0,0) as the only entry.
            self.isrows = np.zeros(len(self.slice_local_arrays)//self.nlocal_timesteps)
            self.isrows = list(PETSc.IS().createGeneral(array, comm=fd.COMM_SELF) for array in self.isrows)
            self.iscols = self.isrows
        
        # PETSc.Sys.Print(f"Time to set-up isets: {time()-start:.2f}s")

        # start = time()

        # Obtain the diagonal and off-diagonal matrices for the time-slice including communication
        self.diag_mats = self.mat.createSubMatrices(self.slice_local_isets, 
                                                    self.slice_local_isets)
        self.offdiag_mats = self.mat.createSubMatrices(self.skip_every_n(self.slice_local_isets, self.nlocal_timesteps, 0),
                                                       self.skip_every_n(self.slice_local_isets, self.nlocal_timesteps, self.nlocal_timesteps-1))
        self.prev_offdiag_mats = self.prevmat.createSubMatrices(self.isrows, 
                                                                self.iscols) 
        
        # We reshape for easier insert, then reshape back to the original structure
        self.offdiag_mats = self.reshape_list(self.offdiag_mats, len(self.offdiag_mats)//(self.nlocal_timesteps-1), self.nlocal_timesteps-1)
        for i, prev_offdiag_mat in enumerate(self.prev_offdiag_mats):
            self.offdiag_mats[i].insert(0, prev_offdiag_mat)
        
        self.offdiag_mats = list(mat for mat_list in self.offdiag_mats for mat in mat_list)

        # Factor the diagonal matrices for the diagonal solve version of apply_impl
        self.factored_diag_mats = list(self.get_factored_matrix(mat, comm=fd.COMM_SELF) for mat in self.diag_mats)
        
        # PETSc.Sys.Print(f"Time to set-up submats: {time()-start:.2f}s")

        #-----------------------------------------------
        # Scatters for apply_impl
        #-----------------------------------------------
        # start = time()

        # In apply_impl we will work with the global indices of the patches,
        # therefore we must shift the indices of the IS's to global numbering.
        first_timestep = self.aaoform.layout.transform_index(0,'l','g') # Get the first timestep for this ranks time-slice in global numbering
        cumulative_time_partition = np.cumsum(self.time_partition) - 1 # -1 to account for zero indexing
        current_timeslice = self.find_index(first_timestep, cumulative_time_partition) # Find which time-slice this rank belongs to
        global_start = self.function_space.dim()*current_timeslice # The offset for the global indices of this time-slice

        self.global_isets = list(self.shiftIS(iset, global_start, comm=fd.COMM_SELF) for iset in self.slice_local_isets)

        # 1. Master vector setup (covers all DoFs for subdomains on this rank)
        unique_global_ids_on_rank_list = sorted(list(set(idx for iset in self.global_isets for idx in iset.getIndices())))

        self.master_is_global_numbering = PETSc.IS().createGeneral(unique_global_ids_on_rank_list, comm=PETSc.COMM_SELF)
        self.master_is_local_numbering = PETSc.IS().createStride(len(unique_global_ids_on_rank_list), 0, 1, comm=PETSc.COMM_SELF)
            
        # Create master vectors (like osm->lx and osm->ly)
        self.master_rhs_vec = PETSc.Vec().createSeq(len(unique_global_ids_on_rank_list), comm=PETSc.COMM_SELF)
        self.master_sol_vec = PETSc.Vec().createSeq(len(unique_global_ids_on_rank_list), comm=PETSc.COMM_SELF)

        # 2. Scatter from global x to master_rhs_vec
        with self._x.global_vec_ro() as xvec:
            self.scatter_x_to_master = PETSc.Scatter().create(xvec, self.master_is_global_numbering, 
                                                                self.master_rhs_vec, self.master_is_local_numbering)
        
        # 3. Scatter from master_sol_vec to global y
        with self._y.global_vec_wo() as yvec:
            self.scatter_master_to_y = PETSc.Scatter().create(self.master_sol_vec, self.master_is_local_numbering, 
                                                                yvec, self.master_is_global_numbering)

        # 4. Scatters for each subdomain from/to master vectors
        self.scatters_master_to_sub_rhs = []
        self.scatters_sub_sol_to_master = []
        self.sub_rhs_vecs = [] # To store sub_RHS vectors
        self.sub_sol_vecs = [] # To store sub_solution vectors

        global_to_master_local_map = {gid: lidx for lidx, gid in enumerate(unique_global_ids_on_rank_list)}

        for i, global_is in enumerate(self.global_isets):
            sub_mat = self.diag_mats[i]
            sub_rhs_v = sub_mat.createVecRight()
            sub_sol_v = sub_mat.createVecLeft()
            self.sub_rhs_vecs.append(sub_rhs_v)
            self.sub_sol_vecs.append(sub_sol_v)

            indices_in_master_for_sub_i = [global_to_master_local_map[gid] for gid in global_is.getIndices()]
            
            is_from_master_for_sub_i = PETSc.IS().createGeneral(indices_in_master_for_sub_i, comm=PETSc.COMM_SELF)
            is_to_sub_i_local_stride = PETSc.IS().createStride(len(indices_in_master_for_sub_i), 0, 1, comm=PETSc.COMM_SELF)

            sc_master_to_sub = PETSc.Scatter().create(self.master_rhs_vec, is_from_master_for_sub_i, 
                                                        sub_rhs_v, is_to_sub_i_local_stride)
            self.scatters_master_to_sub_rhs.append(sc_master_to_sub)

            # For scatter from sub_sol_v to master_sol_vec (used in REVERSE)
            self.scatters_sub_sol_to_master.append(sc_master_to_sub) # We reuse the same scatter object
        # PETSc.Sys.Print(f"Time to set-up scatters: {time()-start:.2f}s")

        self.initialized = True

    def reshape_list(self, data, rows, cols):
            if len(data) != rows * cols:
                raise ValueError("Total elements do not match target shape.")
            return [data[i*cols:(i+1)*cols] for i in range(rows)]

    def find_index(self, number, bounds):
        for i, b in enumerate(bounds):
            if number <= b:
                return i
        return len(bounds)
    
    def skip_every_n(self, data, n, offset):
        return [v for i, v in enumerate(data) if (i - offset) % n != 0]

    def keep_every_n(self, data, n, offset):
        # return data[offset::n]  # This is a more concise way to do it
        return data[offset::n] if n > 0 else data
    
    def shiftIS(self, iset, shift, comm=fd.COMM_SELF):
        """
        Shift the indices of a PETSc IS by a given amount.
        """
        new_indices = iset.getIndices() + shift
        # iset.destroy()  # Destroy the old IS to free memory
        return PETSc.IS().createGeneral(new_indices, comm=comm)
    
    def splitIS(self, iset, n, comm=fd.COMM_SELF):
        """
        Split a PETSc IS into n equal parts.
        """
        indices = iset.getIndices()
        size = len(indices)
        if size % n != 0:
            raise ValueError("The IS cannot be evenly split into the specified number of parts.")
        
        chunks = np.array_split(indices, n)
        return [PETSc.IS().createGeneral(chunk, comm=comm) for chunk in chunks]

    def shift_and_split_IS(self, iset, shift, n, comm=fd.COMM_SELF):
        """
        Shift the indices of a PETSc IS and split it into n equal parts.

        :param iset: A PETSc IS object.
        :param shift: Integer value to add to all indices.
        :param n: Number of equal parts to split the IS into.
        :param comm: MPI communicator (default: fd.COMM_SELF).
        :return: A list of PETSc IS objects.
        """
        indices = iset.getIndices()
        indices = indices + shift

        size = len(indices)
        if size % n != 0:
            raise ValueError("The IS cannot be evenly split into the specified number of parts.")

        chunks = np.array_split(indices, n)
        iset.destroy()
        return tuple(PETSc.IS().createGeneral(chunk, comm=comm) for chunk in chunks)

    @profiler()
    def _record_diagnostics(self):
        pass
    
    @profiler()
    def update(self, pc):
        """
        Update the state to linearise around according to aaojacobi_state.
        """

        aaofunc = self.aaofunc
        aaoform = self.aaoform
        state_func = self.state_func
        # jacobian_state = self.jac_state

        # for st, ft in zip(self.time, aaoform.time):
        #     st.assign(ft)

        # if jacobian_state == 'linear':
        #     return

        # elif jacobian_state == 'current':
        #     state_func.assign(aaofunc)

        # elif jacobian_state in ('window', 'slice'):
        #     time_average(aaofunc, state_func.initial_condition,
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

        # aaofunc.update_time_halos()

        # Recompute the Jacobian matrices with the updated state
        # if jacobian_state != 'linear':
        #     self.mat = fd.assemble(self.jacobian.form, bcs=self.jacobian.bcs).petscmat
        #     self.prevmat = fd.assemble(self.jacobian.form_prev).petscmat if self.jacobian._useprev else self.prevmat

        #     self.diag_mats = self.mat.createSubMatrices(self.slice_local_isets, 
        #                                                 self.slice_local_isets)
        #     self.offdiag_mats = self.mat.createSubMatrices(self.skip_every_n(self.slice_local_isets, self.nlocal_timesteps, 0),
        #                                                 self.skip_every_n(self.slice_local_isets, self.nlocal_timesteps, self.nlocal_timesteps-1))
        #     self.prev_offdiag_mats = self.prevmat.createSubMatrices(self.isrows, 
        #                                                             self.iscols)
            
        #     # We reshape for easier insert, then reshape back to the original structure
        #     self.offdiag_mats = self.reshape_list(self.offdiag_mats, len(self.offdiag_mats)//(self.nlocal_timesteps-1), self.nlocal_timesteps-1)
        #     for i, prev_offdiag_mat in enumerate(self.prev_offdiag_mats):
        #         self.offdiag_mats[i].insert(0, prev_offdiag_mat)
            
        #     self.offdiag_mats = list(mat for mat_list in self.offdiag_mats for mat in mat_list)

        #     # Factor the diagonal matrices for the diagonal solve version of apply_impl
        #     self.factored_diag_mats = list(self.get_factored_matrix(mat, comm=fd.COMM_SELF) for mat in self.diag_mats)

        return

    @profiler()
    def apply_impl_1(self, pc, x, y):
        """
        Test apply_impl, using only the main diag and doing a simple solve, like PCASM
        To ensure that the patch distribution is correct, and initialize does what it should.
        """
        # 1. Scatter global RHS x to master_rhs_vec (local "ghosted" vector)

        with x.global_vec_ro() as xvec:
            self.scatter_x_to_master.scatter(xvec, self.master_rhs_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # 2. Solve on subdomains
        self.master_sol_vec.set(0.0) # Initialize local solution accumulator

        for i in range(len(self.global_isets)):
            scatter_master_to_sub = self.scatters_master_to_sub_rhs[i] # The same scatter object is used for reverse direction later.
            
            sub_rhs_vec_i = self.sub_rhs_vecs[i]
            sub_sol_vec_i = self.sub_sol_vecs[i]

            indices = self.global_isets[i].getIndices()
            if len(indices) != sub_rhs_vec_i.getSize():
                PETSc.Sys.Print(f"Rank {fd.COMM_WORLD.rank}. sub_rhs_vec_i size: {sub_rhs_vec_i.getSize()}, iset size {len(indices)}", comm=fd.COMM_SELF)

            # Scatter from master_rhs_vec to sub_rhs_vec_i
            scatter_master_to_sub.scatter(self.master_rhs_vec, sub_rhs_vec_i, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            
            F = self.factored_diag_mats[i]  # Get the factored matrix for this subdomain
            F.solve(sub_rhs_vec_i, sub_sol_vec_i)  # Solve the local system for this subdomain

            # Add local solution sub_sol_vec_i to master_sol_vec
            # This is SCATTER_REVERSE with ADD_VALUES using the same scatter object
            scatter_master_to_sub.scatter(sub_sol_vec_i, self.master_sol_vec, addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # 3. Scatter accumulated master_sol_vec to global y
        with y.global_vec_wo() as yvec:
            yvec.set(0.0) # Crucial for additive methods.
            self.scatter_master_to_y.scatter(self.master_sol_vec, yvec, addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.FORWARD) # This scatter is defined from master to global.

    @profiler()
    def apply_impl(self, pc, x, y):
        """
        Custom additive Schwarz-style preconditioner:
        - Assumes self.patches: tuple of PETSc IS (global indices for patches)
        - Assumes self.submats: tuple of sequential patch matrices (one per patch)
        """
        @profiler()
        def startup():
            # Scatter global RHS to local ghosted vector
            with x.global_vec_ro() as xvec:
                self.scatter_x_to_master.scatter(xvec, self.master_rhs_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            self.master_sol_vec.set(0.0) # Initialize local solution accumulator
            # Scatter from local ghosted vector to subdomain vectors
            for i in range(len(self.global_isets)):
                scatter_master_to_sub = self.scatters_master_to_sub_rhs[i] # The same scatter object is used for reverse direction later.
                sub_rhs_vec_i = self.sub_rhs_vecs[i]
                scatter_master_to_sub.scatter(self.master_rhs_vec, sub_rhs_vec_i, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    
            # We reshape for convenience, such that each entry corresponds to a list of timesteps for a single patch 
            rows = len(self.global_isets) // self.nlocal_timesteps
            cols = self.nlocal_timesteps

            # We reshape to a more convenient structure unless they
            # have already been reshaped in a previous call to apply_impl()
            if len(self.diag_mats) == len(self.global_isets): self.diag_mats = self.reshape_list(self.diag_mats, rows, cols)
            if len(self.factored_diag_mats) == len(self.global_isets): self.factored_diag_mats = self.reshape_list(self.factored_diag_mats, rows, cols)
            if len(self.offdiag_mats) == len(self.global_isets): self.offdiag_mats = self.reshape_list(self.offdiag_mats, rows, cols)
            if len(self.sub_rhs_vecs) == len(self.global_isets): self.sub_rhs_vecs = self.reshape_list(self.sub_rhs_vecs, rows, cols)
            if len(self.sub_sol_vecs) == len(self.global_isets): self.sub_sol_vecs = self.reshape_list(self.sub_sol_vecs, rows, cols)
        startup()
        
        
        # ---------------------------------------------------------------------------
        # Solve the first block for each patch system on temporal rank 0
        # ---------------------------------------------------------------------------
        if self.temporal_rank == 0:
            u1s = []
            for i in range(len(self.diag_mats)):
                F = self.factored_diag_mats[i][0]
                F.solve(self.sub_rhs_vecs[i][0], self.sub_sol_vecs[i][0])
                u1s.append(self.sub_sol_vecs[i][0].copy())
        # ---------------------------------------------------------------------------
        # For each patch system, apply forward reduction to get L, D, f
        # ---------------------------------------------------------------------------
        Ls, Ds, fs = [], [], []
        for offdiags, diags, rhss in zip(self.offdiag_mats, self.diag_mats, self.sub_rhs_vecs):
            L, D, f = self.forward_reduction(offdiags, diags, rhss)
            Ls.append(L)
            Ds.append(D)
            fs.append(f)
        
        # ---------------------------------------------------------------------------
        # INTERFACE SOLVE (processor communication)
        # ---------------------------------------------------------------------------
        # u_prevs = None
        @profiler()
        def interface_solve():

            # Total number of temporal ranks
            n_temporal = self.ensemble.ensemble_comm.size

            # Ring communication
            size = self.ensemble.ensemble_comm.size
            rank = self.ensemble.ensemble_comm.rank
            dst = (rank+1) % size
            src = (rank-1) % size

            if self.temporal_rank > 0:
                u_prevs = self.ensemble.ensemble_comm.recv(source=src, tag=0)
                # Convert received u_prevs to PETSc Vecs
                for i,u_prev in enumerate(u_prevs):
                    u_vec = PETSc.Vec().createSeq(len(u_prev), comm=PETSc.COMM_SELF)
                    u_vec.setArray(u_prev)  # Set the array data
                    u_prevs[i] = u_vec
            else:
                # If this is the first temporal rank, we use u1 as u_prev
                u_prevs = u1s

            u_nexts = []
            for u_prev, L, D, f in zip(u_prevs, Ls, Ds, fs):
                # Solve: u_next = D^{-1} (f - L * u_prev)
                rhs = f.duplicate()
                f.scale(-1.0)
                L.multAdd(u_prev, f, rhs)  # rhs <- L * u_prev - f
                rhs.scale(-1.0)  # rhs <- f - L * u_prev

                u_next = rhs.duplicate()
                F = self.get_factored_matrix(D, comm = fd.COMM_SELF)
                F.solve(rhs, u_next)
                u_nexts.append(u_next)
            
            # Obviouslt can't send a list of PETSc vecs, but we can convert them to a list of np.ndarrays
            if self.temporal_rank < n_temporal - 1:
                u_nexts = [u_next.getArray() for u_next in u_nexts]
                self.ensemble.ensemble_comm.send(u_nexts, dst, 0)
            
            return u_prevs
        
        u_prevs = interface_solve()
        
        # All ranks now own a u_prev and u_next. Most importantly, u_prev for each processor can be used 
        # to solve for all its owning rows of the global system.

        # ---------------------------------------------------------------------------
        # Forward substitution to solve for each patch system 
        # ---------------------------------------------------------------------------
        for offdiags, factored_diags, rhss, sols, u_prev in zip(self.offdiag_mats, self.factored_diag_mats, self.sub_rhs_vecs, self.sub_sol_vecs, u_prevs):
            self.forward_substitution(offdiags, factored_diags, rhss, sols, u_prev)

        
        @profiler()
        def solution_filling():      
            self.sub_sol_vecs = list(sol for sublist in self.sub_sol_vecs for sol in sublist)
            self.sub_rhs_vecs = list(rhs for sublist in self.sub_rhs_vecs for rhs in sublist)

            for i in range(len(self.global_isets)):
                sub_sol_vec_i = self.sub_sol_vecs[i]
                scatter_master_to_sub = self.scatters_master_to_sub_rhs[i]
                scatter_master_to_sub.scatter(sub_sol_vec_i, self.master_sol_vec, addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            
            # 3. Scatter accumulated master_sol_vec to global y
            with y.global_vec_wo() as yvec:
                yvec.set(0.0)
                self.scatter_master_to_y.scatter(self.master_sol_vec, yvec, addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.FORWARD) # This scatter is defined from master to global.
        solution_filling()


    @profiler()
    def forward_reduction(self, lower_diag, main_diag, rhs):
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

        if len(main_diag) <= 1:
            return lower_diag[0].copy(), main_diag[0].copy(), rhs[0].copy()

        # Pre-allocate buffers
        L, D, f = lower_diag[offset].copy(), main_diag[offset].copy(), rhs[offset].copy()
        
        # Matrix-matrix products will be dense, better to convert only once
        L.convert('dense')
        L.setUp()
        L.assemble()

        D_inv_L = PETSc.Mat().createDense(size=L.getSizes(), comm=fd.COMM_SELF)
        D_inv_L.setUp()
        D_inv_L.assemble()

        D_inv_f = f.duplicate()
        

        for i in range(offset + 1, self.nlocal_timesteps):
            # Factor using LU decomposition with MUMPS. Factored mat is required for any solve() routine
            D_factored = self.get_factored_matrix(D, comm=fd.COMM_SELF)

            L_next, D_next, f_next = lower_diag[i], main_diag[i].copy(), rhs[i].copy()

            D_factored.matSolve(L, D_inv_L) # D⁻¹ * L and stores the result in D_inv_L
            L_next.matMult(D_inv_L,L) # L_next * D⁻¹ * L and stores the result in L. L is now correctly updated for the next iteration of reduction

            D_next.scale(-1.0) # D_next <- -D_next
            D = D_next.copy() # D <- D_next

            # Compute f <- L_next * D^{-1} * f - f_next
            D_factored.solve(f, D_inv_f) # D^{-1} * f
            f_next.scale(-1.0) # f_next <- -f_next
            L_next.multAdd(D_inv_f, f_next, f) # f <- L_next * D^{-1} * f - f_next

            # L_next = lower_diag[i].copy()
            # D_next = main_diag[i].copy()
            # f_next = rhs[i].copy()

            # # Compute L <- L_next * D^{-1} * L, D <- -D_next and f <- L_next * D^{-1} * f - f_next
            # L_dense = L.copy()
            # L_dense.convert('dense')
            # D_inv_L_dense = PETSc.Mat().createDense(size=L_dense.getSizes(), comm=fd.COMM_SELF) # Create a dense matrix for D^{-1} * L
            # D_inv_L_dense.setUp()
            # D_inv_L_dense.assemble()

            # # Compute L <- L_next * D^{-1} * L
            # D_factored.matSolve(L_dense, D_inv_L_dense) # D^{-1} * L and stores in D_inv_L_dense
            # L.convert('dense') # Convert L to dense matrix
            # L.setUp()
            # L.assemble()
            # L_next.matMult(D_inv_L_dense, L) # L <- L_next * D^{-1} * L

            # # Compute D <- - D_next
            # D_next.scale(-1.0) # D_next <- -D_next
            # D = D_next.copy() # D <- D_next

            # # Compute f <- L_next * D^{-1} * f - f_next
            # D_inv_f = f.duplicate()
            # D_factored.solve(f, D_inv_f) # D^{-1} * f
            # f_next.scale(-1.0) # f_next <- -f_next
            # L_next.multAdd(D_inv_f, f_next, f) # f <- L_next * D^{-1} * f - f_next

            # # Destruction
            # D_factored.destroy()
            # L_dense.destroy()
            # D_inv_L_dense.destroy()
            # D_inv_f.destroy()

            # L_next.destroy()
            # D_next.destroy()
            # f_next.destroy()
                
        return L, D, f

    @profiler()
    def forward_substitution(self, lower_diag, factored_main_diag, rhs, sol, u_prev):
        """
        Perform the back substitution step of the cyclic reduction algorithm.
        """
        offset = 1 if self.temporal_rank == 0 else 0 # Since temporal rank 0 is offset from other ranks by 1. It has 1 more row than other ranks

        for i in range(offset, self.nlocal_timesteps):
            L = lower_diag[i]
            F = factored_main_diag[i]
            f = rhs[i]

            # Solve: u_next = D^{-1} (f - L * u_prev)
            tmp = f.duplicate()
            f.scale(-1.0) # Set f <- -f
            L.multAdd(u_prev, f, tmp)  # rhs_tmp <- L * u_prev - f
            tmp.scale(-1.0) # rhs_tmp <- f - L * u_prev
            F.solve(tmp, sol[i])
            u_prev = sol[i].copy()

            # Destruction
            tmp.destroy()

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

    

class ApproxCyclicReductionPC3(CyclicReductionPC3):
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
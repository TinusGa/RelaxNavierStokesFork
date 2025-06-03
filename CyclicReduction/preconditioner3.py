import numpy as np
from time import time

import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.preconditioners import ASMPatchPC
from firedrake.logging import warning

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.preconditioners.base import AllAtOnceBlockPCBase
from asQ.parallel_arrays import SharedArray
from asQ.allatonce import LinearSolver

from asQ import (
    AllAtOnceFunction,
    AllAtOnceCofunction,
    AllAtOnceForm,
    AllAtOnceSolver,
)


__all__ = ['CyclicReductionPC3','ApproxCyclicReductionPC3','asQMGPC']

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

class asQMGPC(AllAtOnceBlockPCBase):
    prefix = 'opts_'
    valid_jacobian_states = tuple(('window', 'slice', 'linear', 'initial', 'reference'))

    @profiler()
    def initialize(self, pc):
        pc.setOptionsPrefix('asQMGPC_')
        super().initialize(pc, final_initialize=False)
        
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

        # Create AllAtOnce objects for each function space in the hierarchy
        for V, bcs in zip(self.function_spaces,self.bcs_list):

            u = AllAtOnceFunction(self.ensemble, self.time_partition, V)
            r = AllAtOnceCofunction(self.ensemble, self.time_partition, u.field_function_space.dual())
            form = AllAtOnceForm(u, self.dt, self.theta,
                                 self.form_mass, self.form_function,
                                 bcs=bcs)  
            solver = LinearSolver(form, solver_parameters=self.sub_solver_parameters, 
                                  options_prefix=self.full_prefix)
            
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

        self.initialized = True

    def _record_diagnostics(self):
        pass
    
    @profiler()
    def update(self, pc):
        pass
    
    @profiler()
    def apply_impl(self, pc, x, y):
        # Might need to move the v-cycle here
        # No, we can solve in initialize and just pass the solution to the apply function
        # y.assign(self.somesolution) for example
        # x is the residual AllAtOnceCofunction
        # y is the solution AllAtOnceFunction
        # use these for the MG solve

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
        # PETSc.Sys.Print(f"Coarse solve")
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
        self.patch_parameters = PETSc.Options(self.full_prefix).getAll()

        self.mat = self.jacobian.mat # Obtain the monolithic matrix from the Jacobian. Of type 'aij' or 'mpiaij'. 
        # Turns out this mat is not monolithic but time-sliced
        # PETSc.Sys.Print(f"Jacobian mat {self.mat.getSize()} with ownership range {self.mat.getOwnershipRanges()}")

        # All-at-once reference state
        self.state_func = self.aaofunc.copy()
        self.field_function_space = self.aaofunc.field_function_space # This is currently not compatible with how MG is set up
        self.function_space = self.aaofunc.function_space

        # This processor's spatial and temporal rank
        self.spatial_rank = self.ensemble.comm.rank
        self.temporal_rank = self.ensemble.ensemble_comm.rank

        # Zero out bc dofs
        self.block_bcs = tuple(
            fd.DirichletBC(self.field_function_space,
                           0*bc.function_arg,
                           bc.sub_domain)
            for bc in self.aaoform.field_bcs)
        
        # Initialize the desired patch PC. There's probably a better way to do this, but this works for now
        self.star_pc = ASMStarPC()
        self.star_pc.prefix = 'star'
        self.patches = self.star_pc.get_patches(self.function_space) # Get path IS in local numbering
        lgmap = self.function_space.dof_dset.lgmap # Local to global map for the function space
        self.patches = list(lgmap.applyIS(iset) for iset in self.patches) # Convert local indices to global indices

        # All indices are correct, but only for the first time-slice. Therefore, we must shift the IS's 
        # of ranks not belonging to this time-slice to their own time-slice.
        # first_timestep = self.aaoform.layout.transform_index(0,'l','g') # Get the first timestep for this ranks time-slice in global numbering
        # cumulative_time_partition = np.cumsum(self.time_partition) - 1 # -1 to account for zero indexing
        # current_timeslice = self.find_index(first_timestep, cumulative_time_partition) # Find which time-slice this rank belongs to
        # global_start = self.function_space.dim()*current_timeslice # The offset for the global indices of this time-slice
        global_start = 0 # When the Jacobian mat is time-sliced, the global start is always 0. If it was monolithic, we would need to shift the indices to the correct time-slice.

        # self.patches is now a list of lists of PETSc IS's
        # For each spatial dof (specified by construct_dim) i, self.patches[i] holds a list of IS's for each time-step on this time-slice.
        # For each self.patches[i] we want to create an LBD system where the main diagonal block correspond to
        # the rows and columns (self.patches[i], self.patches[i]) and the lower diagonal block corresponds to
        # the rows and columns (self.patches[i][1:], self.patches[i][:-1])
        
    
        # PETSc.Sys.Print(f"len(self.patches) = {len(self.patches)}",comm=fd.COMM_SELF)
        self.patches = list(self.shift_and_split_IS(iset, global_start, self.nlocal_timesteps) for iset in self.patches) # Shift the indices of the IS's to the correct time-slice and split them into nlocal_timesteps parts
        # PETSc.Sys.Print(f"len(self.patches) after split = {len(self.patches)}, each el of len {len(self.patches[0])}",comm=fd.COMM_SELF)
        # We have a dependency in time, which means the last diagonal block in each self.diag_submats[i] 
        # corresponds to the first self.offdiag_submats[i] block, on the next time-slice.
        # Therefore, we must communicate. Communication must occur between consecutive temporal ranks which share the same spatial rank.
        if self.temporal_rank == self.ensemble.ensemble_comm.size - 1:
            to_send = [0]*len(self.patches) # Last temporal rank sends nothing, as it has no next rank to communicate with
        else:
            to_send = list(self.patches[i][-1].indices.tolist() for i in range(len(self.patches)))

        # Recieve buffer
        to_recv = []
        # Ring communication
        size = self.ensemble.ensemble_comm.size
        rank = self.ensemble.ensemble_comm.rank

        # ring communication
        dst = (rank+1) % size
        src = (rank-1) % size
        
        to_recv = self.ensemble.ensemble_comm.sendrecv(sendobj=to_send, dest=dst, sendtag=0, source=src, recvtag=0)
        # PETSc.Sys.Print(f"len(to_recv) = {len(to_recv)}", comm=fd.COMM_SELF)
        

        self.flattened_patches = tuple(iset for sub in self.patches for iset in sub)
        self.diagmats = self.mat.createSubMatrices(self.flattened_patches, self.flattened_patches) # This gives the correct diagonal

        def skip_every_n(data, n, offset):
            return [v for i, v in enumerate(data) if (i - offset) % n != 0]

        self.offdiagmats = self.mat.createSubMatrices(skip_every_n(self.flattened_patches, self.nlocal_timesteps, 0),
                                                      skip_every_n(self.flattened_patches, self.nlocal_timesteps, self.nlocal_timesteps-1))

        def keep_every_n(data, n, offset):
            return [v for i, v in enumerate(data) if (i - offset) % n == 0]
        
        # Replace the list of indices with the IS
        for i, indices in enumerate(to_recv):
            recv_is = PETSc.IS().createGeneral(indices, comm=fd.COMM_SELF)
            to_recv[i] = recv_is 
        
        # These deinfe the rows
        if self.temporal_rank != 0:
            isrows = keep_every_n(self.flattened_patches, self.nlocal_timesteps,0)
        else:
            # First temporal rank has no previous rank to communicate with, and therefore no extra off-diagonal blocks,
            # but it must still participate in the call to createSubMatrices, so we use the same IS's as the diagonal blocks.
            isrows = to_recv 
        
        # These define the columns
        iscols = to_recv
        
        self.communicated_offdiags = self.mat.createSubMatrices(isrows, iscols) # Create the off-diagonal blocks for the communicated patches

        def reshape_list(data, rows, cols):
            if len(data) != rows * cols:
                raise ValueError("Total elements do not match target shape.")
            return [data[i*cols:(i+1)*cols] for i in range(rows)]
        
        self.diagmats = reshape_list(self.diagmats, len(self.diagmats)//self.nlocal_timesteps, self.nlocal_timesteps)
        self.offdiagmats = reshape_list(self.offdiagmats, len(self.offdiagmats)//(self.nlocal_timesteps-1), self.nlocal_timesteps-1)

        for i, comm_offdiag in enumerate(self.communicated_offdiags):
            self.offdiagmats[i].insert(0, comm_offdiag)


        # PETSc.Sys.Print(f"len(self.diagmats) = {len(self.diagmats)}, len(self.offdiagmats) = {len(self.offdiagmats)}",comm=fd.COMM_SELF)

        # Now we need the rhs working vectors for each submatrix.
        # since we are working with the global rhs self._x, the IS's
        # need to be in global numbering, not time-slice local numbering.
        first_timestep = self.aaoform.layout.transform_index(0,'l','g') # Get the first timestep for this ranks time-slice in global numbering
        cumulative_time_partition = np.cumsum(self.time_partition) - 1 # -1 to account for zero indexing
        current_timeslice = self.find_index(first_timestep, cumulative_time_partition) # Find which time-slice this rank belongs to
        global_start = self.function_space.dim()*current_timeslice # The offset for the global indices of this time-slice

        self.shifted_patches = list(self.shiftIS(iset, global_start, comm=fd.COMM_SELF) for iset in self.flattened_patches)

        patch_vecs = []

        # with self._x.global_vec_ro() as xvec:
        #     for iset in self.shifted_patches:
        #         # Create sequential Vec of appropriate size
        #         PETSc.Sys.Print(f"Rank: {fd.COMM_WORLD.rank}, indices {iset.indices}", comm=fd.COMM_SELF)
        #         patch_vec = PETSc.Vec().createSeq(iset.getSize(), comm=fd.COMM_SELF)

        #         # Scatter from xvec into local patch_vec
        #         is_to = PETSc.IS().createStride(iset.getSize(), 0, 1, comm=fd.COMM_SELF)
        #         scatter = PETSc.Scatter().create(xvec, iset, patch_vec, is_to)
        #         scatter.begin(xvec, patch_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        #         scatter.end(xvec, patch_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        #         patch_vecs.append(patch_vec)
            
        

        # PETSc.Sys.Print(f"Rank: {rank}, sending to {dst}, receiving from {src}", comm=self.ensemble.comm)

        # 


        # for patch_list in self.patches:
        #     PETSc.Sys.Print(f"Rank: {fd.COMM_WORLD.rank}. Patch list length: {len(patch_list)}", comm=fd.COMM_SELF)
        #     # PETSc.Sys.Print(f"Patch size: {len(patch.indices)}", comm=fd.COMM_SELF)

        # self.submats = self.mat.createSubMatrices(self.patches,self.patches)
        # self.subvecs = [submat.createVecs(side='right') for submat in self.submats] # Create subvectors for each submatrix

        # TODO: Should set up ksp solvers here so they can be reused in apply_impl() maybe?????????

        
    
        # # We need to shift from local to global numbering. 
        # # We also have to create new indices for each time-step on the slice. Best to do all in one go.
        # # Create new IS, cause PETSc IS's are immutable. 
        # ownership_start = pc.getOperators()[0].getOwnershipRange()[0]
        # PETSc.Sys.Print(f"ranges : {pc.getOperators()[0].getOwnershipRanges()}")
        set_size = self.field_function_space.node_set.size 
        # PETSc.Sys.Print(f"field f'n space dim {self.field_function_space.dim()}")
        # for i,patch in enumerate(self.patches):
        #     # Shift from local to global numbering
        #     # indices = patch.indices + ownership_start # Shift indices to global numbering
        #     indices = patch.indices + ownership_start
        #     #Create a list of indices for each time-step
        #     index_list = [indices + set_size*i for i in range(self.nlocal_timesteps)] 
        #     #Flatten the list of indices
        #     indices = [item for sublist in index_list for item in sublist]
        #     #Create a new IS with the global indices

        #     new_patch_IS = PETSc.IS().createGeneral(indices, comm=fd.COMM_SELF)
        #     self.patches[i] = new_patch_IS # Replace the old IS with the new one
        #     patch.destroy() # Destroy the old IS to free memory

        # for patch in self.patches:
        #     PETSc.Sys.Print(f"Patch indices: {patch.indices}", comm=fd.COMM_SELF)
        #     # PETSc.Sys.Print(f"Patch size: {len(patch.indices)}", comm=fd.COMM_SELF)


        # all_indices = [patch.indices for patch in self.patches]
        # all_indices_flat = [item for sublist in all_indices for item in sublist]
        # PETSc.Sys.Print(f"Rank: {fd.COMM_WORLD.rank}, num. patches: {len(self.patches)}, min idx: {min(all_indices_flat)}, max idx: {max(all_indices_flat)}, ownership range: {pc.getOperators()[0].getOwnershipRange()}, set_size: {set_size}",comm=fd.COMM_SELF)
        # fd.COMM_WORLD.Barrier() # Ensure all ranks have the same number of patches

        
        
        # We can either use lists or perhaps a nested matrix to store the LBD systems
        # self.diag_matrices = []
        # self.lower_diag_matrices = []
        # self.rhs = []

        # self.dt1 = fd.Constant(1/self.dt)
        # self.theta = fd.Constant(1)

        # for i in range(self.nlocal_timesteps):
        #     # The reference states
        #     u0 = self.state_func[i]
        #     t = self.time[i]

        #     v = fd.TestFunction(self.field_function_space)
        #     u = fd.TrialFunction(self.field_function_space)

        #     M = self.form_mass(u, v)
        #     K = self.form_function(u, v, t)

        #     # Represents the linear system for timestep/row i. That is L*u[i] + D*u[i+1] = f[i+1]
        #     D = fd.assemble(self.dt1*M + self.theta*K, bcs=self.block_bcs).petscmat # Main diagonal block system
        #     L = fd.assemble(-1*self.dt1*M, bcs=self.block_bcs).petscmat # Lower/off - diagonal block system
        #     f = self._x[i].dat._vec

        #     self.diag_matrices.append(D)
        #     self.lower_diag_matrices.append(L)
        #     self.rhs.append(f)

        # self.block_iterations = SharedArray(self.time_partition,
        #                                     dtype=int,
        #                                     comm=self.ensemble.ensemble_comm) # Not currently used for anything
        # self.initialized = True


    def find_index(self, number, bounds):
        for i, b in enumerate(bounds):
            if number <= b:
                return i
        return len(bounds)
    
    def shiftIS(self, iset, shift, comm=fd.COMM_SELF):
        """
        Shift the indices of a PETSc IS by a given amount.
        """
        new_indices = iset.getIndices() + shift
        iset.destroy()  # Destroy the old IS to free memory
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
        No need to update. The method should only be called once per rank.
        """
        pass

    @profiler()
    def apply_impl(self, pc, x, y):
        
        pass

    @profiler()
    def apply_impl(self, pc, x, y):
        """
        Custom additive Schwarz-style preconditioner:
        - Assumes self.patches: tuple of PETSc IS (global indices for patches)
        - Assumes self.submats: tuple of sequential patch matrices (one per patch)
        """

        # --- Step 1: Build a ghosted local vector covering all patch DOFs ---
        patch_indices = [iset.getIndices() for iset in self.shifted_patches]
        unique_global_ids = sorted(set(i for indices in patch_indices for i in indices))
        global_to_local = {g: i for i, g in enumerate(unique_global_ids)}

        # Unified IS and ghosted local vector (covers all DOFs needed by patches)
        lis = PETSc.IS().createGeneral(unique_global_ids, comm=fd.COMM_SELF)
        local_vec = PETSc.Vec().createSeq(len(unique_global_ids), comm=fd.COMM_SELF)

        # Scatter from parallel input x to ghosted local vector
        with x.global_vec_ro() as xvec:
            is_to = PETSc.IS().createStride(len(unique_global_ids), 0, 1, comm=fd.COMM_SELF)
            scatter = PETSc.Scatter().create(xvec, lis, local_vec, is_to)
            scatter.begin(xvec, local_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            scatter.end(xvec, local_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        patch_subvecs = []

        for patch_gids in patch_indices:  # Each patch's global indices
            local_ids = [global_to_local[gid] for gid in patch_gids]  # Corresponding local indices
            subvec = PETSc.Vec().createSeq(len(local_ids), comm=fd.COMM_SELF)

            is_from = PETSc.IS().createGeneral(local_ids, comm=fd.COMM_SELF)
            is_to = PETSc.IS().createStride(len(local_ids), 0, 1, comm=fd.COMM_SELF)

            s = PETSc.Scatter().create(local_vec, is_from, subvec, is_to)
            s.begin(local_vec, subvec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            s.end(local_vec, subvec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            patch_subvecs.append(subvec)
        
        def reshape_list(data, rows, cols):
            if len(data) != rows * cols:
                raise ValueError("Total elements do not match target shape.")
            return [data[i*cols:(i+1)*cols] for i in range(rows)]
        
        self.rhss = reshape_list(patch_subvecs, len(patch_subvecs)//self.nlocal_timesteps, self.nlocal_timesteps)

        # ---------------------------------------------------------------------------
        # Solve the first block for each patch system on temporal rank 0
        # ---------------------------------------------------------------------------
        
        if self.temporal_rank == 0:
            u1s = []
            for i in range(len(self.diagmats)):
                F = self.get_factored_matrix(self.diagmats[i][0].copy(), comm = fd.COMM_SELF)
                u1 = self.rhss[i][0].duplicate()
                F.solve(self.rhss[i][0], u1)
                u1s.append(u1)

        # ---------------------------------------------------------------------------
        # For each patch system, apply forward reduction to get L, D, f
        # ---------------------------------------------------------------------------

        Ls, Ds, fs = [], [], []
        for offdiags, diags, rhss in zip(self.offdiagmats, self.diagmats, self.rhss):
            # Apply forward reduction for each patch system
            # PETSc.Sys.Print(f"len(offdiags) = {len(offdiags)}, len(diags) = {len(diags)}, len(rhss) = {len(rhss)}", comm=fd.COMM_SELF)
            L, D, f = self.forward_reduction(offdiags, diags, rhss)
            Ls.append(L)
            Ds.append(D)
            fs.append(f)
        
        # ---------------------------------------------------------------------------
        # INTERFACE SOLVE (processor communication)
        # ---------------------------------------------------------------------------

        # Total number of temporal ranks
        n_temporal = self.ensemble.ensemble_comm.size

        # Ring communication
        size = self.ensemble.ensemble_comm.size
        rank = self.ensemble.ensemble_comm.rank

        # ring communication
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
            # PETSc.Sys.Print(f"Rank {self.temporal_rank}, solving for u_next with L: {L.getSize()}, D: {D.getSize()}, f: {f.getSize()}", comm=fd.COMM_SELF)
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

        # All ranks now own a u_prev and u_next. Most importantly, u_prev for each processor can be used 
        # to solve for all its owning rows of the global system.

        # ---------------------------------------------------------------------------
        # Forward substitution to solve for each patch system 
        # ---------------------------------------------------------------------------
        solutions = []
        for offdiags, diags, rhss, u_prev in zip(self.offdiagmats, self.diagmats, self.rhss, u_prevs):
            sols = self.forward_substitution(offdiags, diags, rhss, u_prev, solutions)
            solutions.append(sols)
        
        # we need to append the solution of the first block for temporal rank 0
        if self.temporal_rank == 0:
            for i in range(len(self.diagmats)):
                solutions[i].insert(0, u1s[i])
        
        # Flatten the solutions list to match the patch list
        solutions = [sol for sublist in solutions for sol in sublist]  # Flatten the list of lists

        with y.global_vec_wo() as yvec:
            yvec.set(0)  # Clear the vector before accumulation
            for iset, sol in zip(self.shifted_patches, solutions):
                gids = iset.getIndices()
                yvec.setValues(gids, sol.getArray(), addv=PETSc.InsertMode.ADD_VALUES)
            yvec.assemble()
        
        # --- Step 2: Solve patch systems locally ---
        # with y.global_vec_wo() as yvec:
        #     yvec.set(0)  # Zero output before accumulation
        #     for A_patch, patch, patch_gids in zip(self.submats, self.patches, patch_indices):
        #         # Map global patch DOFs to local vector indices
        #         local_patch_ids = [global_to_local[gid] for gid in patch_gids]

        #         # Use createVecs to get RHS and solution vectors
        #         rhs, sol = A_patch.createVecs()

        #         # Scatter from ghosted vector into RHS
        #         is_from = PETSc.IS().createGeneral(local_patch_ids, comm=fd.COMM_SELF)
        #         is_to = PETSc.IS().createStride(len(local_patch_ids), 0, 1, comm=fd.COMM_SELF)
        #         s = PETSc.Scatter().create(local_vec, is_from, rhs, is_to)
        #         s.begin(local_vec, rhs, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        #         s.end(local_vec, rhs, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                
        #         F = self.get_factored_matrix(A_patch, fd.COMM_SELF)
        #         F.solve(rhs, sol)

        #         # # Accumulate result back into global y
        #         yvec.setValues(patch_gids, sol.getArray(), addv=PETSc.InsertMode.ADD_VALUES)
        #     yvec.assemble()

    
    @profiler()
    def apply_impl1(self, pc, x, y):
        """
        Apply the cyclic reduction preconditioner to the vector x and store the result in y.
        This method is called by the PETSc solver.
        """
        self.patches
        self.submats
        self.subvectors = [] 

        with x.global_vec_ro() as xvec:
            for patch in self.patches:  # each `patch` is a global IS
                subvec = PETSc.Vec().createSeq(patch.getSize(), comm=self.ensemble.comm)
                is_from = patch
                is_to = PETSc.IS().createGeneral(range(patch.getSize()), comm=PETSc.COMM_SELF)
                scatter = PETSc.Scatter().create(xvec, is_from, subvec, is_to)
                scatter.begin(xvec, subvec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                scatter.end(xvec, subvec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                subvec.getArray()
                self.subvectors.append(subvec)

        with y.global_vec_wo() as yvec:
            for lhs, rhs, patch in zip(self.submats, self.subvectors, self.patches):
                ksp = PETSc.KSP().create(comm=self.ensemble.comm)
                ksp.setOperators(lhs)
                ksp.setFromOptions()
                ksp.setUp()
                ksp.solve(rhs, rhs)
                indices = patch.getIndices()  # global indices
                values = rhs.getArray()       # get NumPy array from rhs
                yvec.setValues(indices, values, addv=PETSc.InsertMode.ADD_VALUES) # Any process can write globally. .ADD_VALUES for adding
            yvec.assemble()

    @profiler()
    def apply_impl2(self, pc, x, y):
        # PETSc.Sys.Print(f"Apply called: y.size = {y._vec.getSize()}",comm=fd.COMM_SELF) # Clearly no MG here

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
        L, D, f = self.forward_reduction(self.lower_diag_matrices,
                                         self.diag_matrices,
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
        self.forward_substitution(self.lower_diag_matrices, self.diag_matrices, self.rhs, u_prev, y)
        

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
        
        if len(main_diag) > 1: # This processor owns more than one timestep, so we reduce
            # Reduce onto these variables. L and D are the lower diagonal and main diagonal
            # matrices respectively of type 'mpiaij'.
            L, D, f = lower_diag[offset].copy(), main_diag[offset].copy(), rhs[offset].copy()
            for i in range(offset + 1, self.nlocal_timesteps):
                L_next = lower_diag[i].copy()
                D_next = main_diag[i].copy()
                f_next = rhs[i].copy()

                # Factor using LU decomposition with MUMPS
                D_factored = self.get_factored_matrix(D, comm=fd.COMM_SELF)

                # Compute L <- L_next * D^{-1} * L, D <- -D_next and f <- L_next * D^{-1} * f - f_next
                L_dense = L.copy()
                L_dense.convert('dense')
                D_inv_L_dense = PETSc.Mat().createDense(size=L_dense.getSizes(), comm=fd.COMM_SELF) # Create a dense matrix for D^{-1} * L
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
        else:
            L, D, f = lower_diag[0].copy(), main_diag[0].copy(), rhs[0].copy()
                
        return L, D, f

    
    @profiler()
    def forward_substitution(self, lower_diag, main_diag, rhs, u_prev, y=None):
        """
        Perform the back substitution step of the cyclic reduction algorithm.
        """
        offset = 1 if self.temporal_rank == 0 else 0 # Since temporal rank 0 is offset from other ranks by 1. It has 1 more row than other ranks

        sols = []

        for i in range(offset, self.nlocal_timesteps):

            L = lower_diag[i].copy()
            D = main_diag[i].copy()
            f = rhs[i].copy()

            # Solve: u_next = D^{-1} (f - L * u_prev)
            tmp = f.duplicate()
            f.scale(-1.0) # Set f <- -f
            L.multAdd(u_prev, f, tmp)  # rhs_tmp <- L * u_prev - f
            tmp.scale(-1.0) # rhs_tmp <- f - L * u_prev
            F = self.get_factored_matrix(D, comm=fd.COMM_SELF)
            u_next = tmp.duplicate()
            F.solve(tmp, u_next)

            sols.append(u_next.copy())
            u_prev = u_next.copy()

            # Destruction
            L.destroy()
            D.destroy()
            f.destroy()
            tmp.destroy()
            F.destroy()
            u_next.destroy()
            
        return sols

        
        
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
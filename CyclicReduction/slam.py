import firedrake as fd
from firedrake import PETSc
from firedrake.patches import ASMStarPC # Assuming this is the class you use
from firedrake.assemble import allocate_matrix # For getting DM
from firedrake.parloops import profiler # Your profiler
import numpy as np
from time import time # Your timer

# Assuming AllAtOnceBlockPCBase is a class you have.
# If this PythonSpatialASM_PC is to be a general Firedrake PC,
# it might be better to inherit from firedrake.PCBase.
# For now, sticking to your structure.
class AllAtOnceBlockPCBase: # Dummy base class for the example to run
    def __init__(self):
        self.jacobian = None
        self.aaofunc = None
        self.ensemble = None
        self.aaoform = None
        self.time_partition = None
        self.full_prefix = ""

    def initialize(self, pc, final_initialize=True):
        self.pc = pc # Store the PETSc PC object
        # In a real scenario, self.jacobian, self.aaofunc etc. would be set up
        # by the AllAtOnceSolver or a similar framework.
        # For this example, we'll assume they are populated externally if needed,
        # or we primarily use pc.getOperators()[0] for the matrix.
        pass

    def get_factored_matrix(self, mat, comm):
        # Your implementation for getting a factored matrix (e.g., for direct solve)
        # This is used in your original apply_impl. For the KSP approach,
        # PETSc handles the factorization within the sub-KSP if PC LU/Cholesky is used.
        # For simplicity, we'll use sub-KSPs with LU.
        # If you need this for another part, keep it.
        # For this refactor, sub_ksps will handle local solves.
        pass


class PythonSpatialASM_PC(AllAtOnceBlockPCBase):
    prefix = 'pyasm_' # Changed prefix

    @profiler()
    def initialize(self, pc):
        start_time = time() # Renamed variable for clarity
        super().initialize(pc, final_initialize=False) # Call to super
        # self.patch_parameters = PETSc.Options(self.full_prefix).getAll() # Kept if needed

        # --- Core PCASM-like setup ---
        self.A = pc.getOperators()[0]  # Get the global matrix A from the PETSc PC
                                       # Or use self.jacobian.mat if that's guaranteed to be A
        if hasattr(self, 'jacobian') and self.jacobian and self.jacobian.mat is not None:
            self.A = self.jacobian.mat
            PETSc.Sys.Print(f"[{PETSc.COMM_WORLD.rank}] Using self.jacobian.mat for PythonSpatialASM_PC.")
        else:
            PETSc.Sys.Print(f"[{PETSc.COMM_WORLD.rank}] Using pc.getOperators()[0] for PythonSpatialASM_PC.")


        # Function space for the problem (single time-slice)
        # Assuming self.aaofunc.function_space gives the relevant Firedrake FunctionSpace
        if not hasattr(self, 'aaofunc') or not self.aaofunc.function_space:
            raise ValueError("self.aaofunc.function_space is not available for ASM setup.")
        self.function_space = self.aaofunc.function_space
        self.mesh_comm = self.function_space.mesh().comm

        # Get spatial patches
        # TODO: Make patch type and overlap configurable from options
        patch_pc = ASMStarPC() # Using star patches as in your example
        # The ASMStarPC needs parameters like this, often from a solver_parameters dict
        # For now, assuming defaults are okay, or it pulls from PETSc options if prefix is set.
        # patch_pc.setUp(self.function_space) # ASMStarPC might need setup if it's a full PC
        
        # rank_local_firedrake_is are Firedrake IS objects.
        # These contain indices local to each rank's part of the DM.
        rank_local_firedrake_is = patch_pc.get_patches(self.function_space)

        if not rank_local_firedrake_is:
            PETSc.Sys.Print(f"[{self.mesh_comm.rank}] WARNING: No patches defined on this rank.")
            self.subdomain_is_global = []
            self.sub_matrices = []
            self.sub_ksps = []
            self.master_rhs_vec = None
            self.master_sol_vec = None
            self.scatter_x_to_master = None
            self.scatter_master_to_y = None
            self.scatters_master_to_sub_rhs = []
            self.scatters_sub_sol_to_master = []
            self.sub_rhs_vecs = []
            self.sub_sol_vecs = []
            self.initialized = True
            PETSc.Sys.Print(f"[{self.mesh_comm.rank}] PythonSpatialASM_PC initialized (no patches) in {time() - start_time:.2f} seconds")
            return

        # Convert Firedrake local IS to PETSc IS with global numbering
        lgmap = self.function_space. स्थानीय_to_global_map() # Corrected to local_to_global_map
        self.subdomain_is_global = []
        for rank_local_is in rank_local_firedrake_is:
            global_indices = lgmap.apply(rank_local_is.getIndices())
            self.subdomain_is_global.append(PETSc.IS().createGeneral(global_indices, comm=PETSc.COMM_SELF))

        # Extract submatrices
        self.sub_matrices = self.A.createSubMatrices(self.subdomain_is_global, self.subdomain_is_global)

        # Create local KSP solvers for each subdomain
        self.sub_ksps = []
        ksp_prefix = pc.getOptionsPrefix() + self.prefix # e.g., "ksp_pyasm_"
        for i, sub_mat in enumerate(self.sub_matrices):
            sub_ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            sub_ksp.setOperators(sub_mat)
            # It's important that sub-KSPs have a distinct prefix from the parent KSP
            sub_ksp.setOptionsPrefix(ksp_prefix + f"sub_{i}_")
            
            # Default ASM sub-solver: direct solve
            sub_ksp.setType(PETSc.KSP.Type.PREONLY)
            sub_pc = sub_ksp.getPC()
            sub_pc.setType(PETSc.PC.Type.LU) # or CHOLESKY if symmetric
            sub_pc.setFactorSolverType("mumps") # Example: use MUMPS if available, or PETSc's default

            sub_ksp.setFromOptions() # Allow user overrides
            sub_ksp.setUp() # Important to call after setting options
            self.sub_ksps.append(sub_ksp)

        # --- Setup for scatters in apply_impl ---
        # 1. Master vector setup (covers all DoFs for subdomains on this rank)
        unique_global_ids_on_rank_list = sorted(list(set(idx for iset in self.subdomain_is_global for idx in iset.getIndices())))
        
        if not unique_global_ids_on_rank_list: # Handles case where a rank has no subdomains/DoFs
             PETSc.Sys.Print(f"[{self.mesh_comm.rank}] WARNING: No unique global DoFs for patches on this rank.")
             self.master_rhs_vec = None # Mark that no master vec needed
        else:
            self.master_is_global_numbering = PETSc.IS().createGeneral(unique_global_ids_on_rank_list, comm=PETSc.COMM_SELF)
            self.master_is_local_numbering = PETSc.IS().createStride(len(unique_global_ids_on_rank_list), 0, 1, comm=PETSc.COMM_SELF)
            
            # Create master vectors (like osm->lx and osm->ly)
            self.master_rhs_vec = PETSc.Vec().createSeq(len(unique_global_ids_on_rank_list), comm=PETSc.COMM_SELF)
            self.master_sol_vec = PETSc.Vec().createSeq(len(unique_global_ids_on_rank_list), comm=PETSc.COMM_SELF)

            # 2. Scatter from global x to master_rhs_vec
            # x_global_vec will be pc.getOperators()[0].createVecLeft() or similar layout
            # For now, assume x in apply_impl is a compatible PETSc Vec
            # The Vec x in apply_impl is the global vector.
            self.scatter_x_to_master = PETSc.Scatter().create(self.A.createVecLeft(), self.master_is_global_numbering, 
                                                              self.master_rhs_vec, self.master_is_local_numbering)
            
            # 3. Scatter from master_sol_vec to global y
            self.scatter_master_to_y = PETSc.Scatter().create(self.master_sol_vec, self.master_is_local_numbering, 
                                                              self.A.createVecLeft(), self.master_is_global_numbering)

            # 4. Scatters for each subdomain from/to master vectors
            self.scatters_master_to_sub_rhs = []
            self.scatters_sub_sol_to_master = []
            self.sub_rhs_vecs = [] # To store sub_RHS vectors
            self.sub_sol_vecs = [] # To store sub_solution vectors

            global_to_master_local_map = {gid: lidx for lidx, gid in enumerate(unique_global_ids_on_rank_list)}

            for i, current_subdomain_global_is in enumerate(self.subdomain_is_global):
                sub_mat = self.sub_matrices[i]
                sub_rhs_v = sub_mat.createVecRight()
                sub_sol_v = sub_mat.createVecLeft()
                self.sub_rhs_vecs.append(sub_rhs_v)
                self.sub_sol_vecs.append(sub_sol_v)

                indices_in_master_for_sub_i = [global_to_master_local_map[gid] for gid in current_subdomain_global_is.getIndices()]
                
                is_from_master_for_sub_i = PETSc.IS().createGeneral(indices_in_master_for_sub_i, comm=PETSc.COMM_SELF)
                is_to_sub_i_local_stride = PETSc.IS().createStride(len(indices_in_master_for_sub_i), 0, 1, comm=PETSc.COMM_SELF)

                sc_master_to_sub = PETSc.Scatter().create(self.master_rhs_vec, is_from_master_for_sub_i, 
                                                          sub_rhs_v, is_to_sub_i_local_stride)
                self.scatters_master_to_sub_rhs.append(sc_master_to_sub)

                # For scatter from sub_sol_v to master_sol_vec (used in REVERSE)
                self.scatters_sub_sol_to_master.append(sc_master_to_sub) # We reuse the same scatter object

                # Avoid destroying ISs used by scatter objects until scatters are destroyed
                # PETSc Scatter typically copies IS definitions or increments reference count.
                # For safety, keep them if scatter objects are class members.
                # Or, destroy ISs here if scatters are recreated in apply.
                # Since scatters are class members, ISs should persist or be managed by PETSc.
                # Let's store them too to be explicit.
                if not hasattr(self, 'managed_is'): self.managed_is = []
                self.managed_is.append(is_from_master_for_sub_i)
                self.managed_is.append(is_to_sub_i_local_stride)
        
        self.initialized = True
        PETSc.Sys.Print(f"[{self.mesh_comm.rank}] PythonSpatialASM_PC initialized in {time() - start_time:.2f} seconds. Found {len(self.subdomain_is_global)} subdomains.")

    @profiler()
    def apply(self, pc, x, y): # Standard Firedrake PC apply signature
        """
        Apply the Python Spatial ASM preconditioner.
        x: global input PETSc Vec
        y: global output PETSc Vec (y = Bx where B is the preconditioner)
        """
        if not self.subdomain_is_global or self.master_rhs_vec is None: # No subdomains on this rank
            if self.master_rhs_vec is None and len(self.subdomain_is_global) > 0 : # Should not happen if init is correct
                 PETSc.Sys.Print(f"[{self.mesh_comm.rank}] ERROR: Subdomains exist but master vectors not initialized.")
            # If a rank has no subdomains, it should contribute zero to y.
            # If y must be zeroed globally first, that should be ensured by KSP.
            # Here, this rank does nothing to y if it has no subdomains.
            # However, PCApply should generally compute y = Bx, not y += Bx unless specified.
            # For additive, we often do y.set(0) then sum contributions.
            # If this rank has no part of B, it should ensure its part of y is zero if y isn't zeroed before.
            # Let's assume y is zeroed by KSP before calling PCApply, or we zero it here.
            # y.set(0) # Zero y first. PCASM does this.
            # For now, do nothing if no subdomains. KSP handles y.
            return


        # 1. Scatter global RHS x to master_rhs_vec (local "ghosted" vector)
        #    SCATTER_FORWARD (restriction)
        self.scatter_x_to_master.scatter(x, self.master_rhs_vec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # 2. Solve on subdomains
        self.master_sol_vec.set(0.0) # Initialize local solution accumulator

        for i in range(len(self.subdomain_is_global)):
            sub_ksp = self.sub_ksps[i]
            scatter_master_to_sub = self.scatters_master_to_sub_rhs[i]
            # The same scatter object is used for reverse direction later.
            
            sub_rhs_vec_i = self.sub_rhs_vecs[i]
            sub_sol_vec_i = self.sub_sol_vecs[i]

            # Scatter from master_rhs_vec to sub_rhs_vec_i
            scatter_master_to_sub.scatter(self.master_rhs_vec, sub_rhs_vec_i, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            
            # Solve local problem: KSP_i * u_i = r_i
            sub_ksp.solve(sub_rhs_vec_i, sub_sol_vec_i)
            # Check for solver divergence if needed:
            # if sub_ksp.getConvergedReason() < 0:
            #     PETSc.Sys.Print(f"[{self.mesh_comm.rank}] Subdomain {i} KSP did not converge: {sub_ksp.getConvergedReason()}")

            # Add local solution sub_sol_vec_i to master_sol_vec
            # This is SCATTER_REVERSE with ADD_VALUES using the same scatter object
            scatter_master_to_sub.scatter(sub_sol_vec_i, self.master_sol_vec, addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # 3. Scatter accumulated master_sol_vec to global y
        #    SCATTER_REVERSE (prolongation/summation)
        #    The output y should be Px. Additive Schwarz sums contributions.
        #    PCASM zeroes y then adds. If KSP does not zero y, we should.
        y.set(0.0) # Crucial for additive methods.
        self.scatter_master_to_y.scatter(self.master_sol_vec, y, addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.FORWARD) # This scatter is defined from master to global.
        # The PETSc PCASM code uses its osm->restriction scatter (global to local) in SCATTER_REVERSE mode for this final step.
        # So, if self.scatter_x_to_master is global_to_master, then using it in reverse:
        # self.scatter_x_to_master.scatter(self.master_sol_vec, y, addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)


    @profiler()
    def update(self, pc):
        # For ASM, if the matrix operator changes sparsity, submatrices and KSPs need rebuilding.
        # If only values change, KSPSetOperators might be enough for sub_ksps if their matrices
        # are just views or shallow copies. Here, sub_matrices are new objects.
        # PCASM has logic for MAT_REUSE_MATRIX vs MAT_INITIAL_MATRIX.
        # This simple version reinitializes everything if called.
        # More sophisticated: check pc.getOperatorsChanged() etc.
        PETSc.Sys.Print(f"[{self.mesh_comm.rank}] PythonSpatialASM_PC update called. Reinitializing.")
        self.destroy_petsc_objects() # Clean up old PETSc objects before re-initializing
        self.initialize(pc)

    def destroy_petsc_objects(self):
        # Helper to clean up PETSc objects, important if re-initializing
        if hasattr(self, 'sub_ksps'):
            for ksp in self.sub_ksps: ksp.destroy()
            self.sub_ksps = []
        if hasattr(self, 'sub_matrices'): # These are owned by PETSc Mat.destroySubMatrices if called
                                          # Or if from createSubMatrices, they need individual destroy
            for mat in self.sub_matrices: mat.destroy()
            self.sub_matrices = []
        if hasattr(self, 'subdomain_is_global'):
            for iset in self.subdomain_is_global: iset.destroy()
            self.subdomain_is_global = []
        
        if hasattr(self, 'scatter_x_to_master') and self.scatter_x_to_master: self.scatter_x_to_master.destroy()
        if hasattr(self, 'scatter_master_to_y') and self.scatter_master_to_y: self.scatter_master_to_y.destroy()
        
        if hasattr(self, 'scatters_master_to_sub_rhs'): # These are also in scatters_sub_sol_to_master
            # Only destroy once if they are indeed the same objects
            destroyed_scatters = set()
            for sc in self.scatters_master_to_sub_rhs:
                if sc.handle not in destroyed_scatters: # Check handle to avoid double destroy if objects are reused
                    sc.destroy()
                    destroyed_scatters.add(sc.handle)
            self.scatters_master_to_sub_rhs = []
            self.scatters_sub_sol_to_master = []

        if hasattr(self, 'sub_rhs_vecs'):
            for vec in self.sub_rhs_vecs: vec.destroy()
            self.sub_rhs_vecs = []
        if hasattr(self, 'sub_sol_vecs'):
            for vec in self.sub_sol_vecs: vec.destroy()
            self.sub_sol_vecs = []

        if hasattr(self, 'master_is_global_numbering') and self.master_is_global_numbering: self.master_is_global_numbering.destroy()
        if hasattr(self, 'master_is_local_numbering') and self.master_is_local_numbering: self.master_is_local_numbering.destroy()
        if hasattr(self, 'master_rhs_vec') and self.master_rhs_vec: self.master_rhs_vec.destroy()
        if hasattr(self, 'master_sol_vec') and self.master_sol_vec: self.master_sol_vec.destroy()

        if hasattr(self, 'managed_is'):
            for iset in self.managed_is: iset.destroy()
            self.managed_is = []


    def __del__(self):
        # Basic cleanup when the object is garbage collected
        self.destroy_petsc_objects()

    # Keep your helper functions if they are general and don't involve time-splitting
    # e.g., find_index might not be needed for this spatial PC.
    # shiftIS, splitIS, shift_and_split_IS were for time logic.
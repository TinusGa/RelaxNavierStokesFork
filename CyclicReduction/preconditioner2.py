import numpy as np

import firedrake as fd
from firedrake.petsc import PETSc

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.preconditioners.base import AllAtOncePCBase
from asQ.parallel_arrays import SharedArray
from asQ.allatonce import time_average

from firedrake.preconditioners import ASMPatchPC, PCBase, ASMStarPC
from firedrake.dmhooks import get_function_space
from firedrake.logging import warning


# __all__ = ['CyclicReductionPC2','ASMStarPC','Setup']
__all__ = ['CyclicReductionPC2']

# class Setup(AllAtOncePCBase):
#     """
#     Setup class for the CyclicReduction preconditioner.
#     """
#     prefix = 'setup_'

#     def initialize(self, pc):
#         pc.setOptionsPrefix('setup_')
#         super().initialize(pc, final_initialize=False)

#         prefix = pc.getOptionsPrefix()
#         prefix = 'pc_python_mg_'
#         self.pc = pc
#         self.opts = PETSc.Options(prefix).getAll()
#         PETSc.Sys.Print(f"Setup: {self.opts}")

#         _, A = pc.getOperators()


#         sub_ksp = PETSc.KSP().create(self.ensemble.comm)
#         sub_ksp.setOperators(A)
#         sub_ksp.setOptionsPrefix("pc_python_mg_")
#         sub_ksp.setFromOptions()
#         sub_ksp.setUp()

        
        
#     def update(self, pc):
#         pass

#     def apply(self, pc, x, y):
#         pass

#     def applyTranspose(self, pc, x, y):
#         pass

#     def view(self, pc, viewer=None):
#         pass

#     def destroy(self, pc):
#         pass


# def order_points(mesh_dm, points, ordering_type, prefix):
#     '''Order the points (topological entities) of a patch based
#     on the adjacency graph of the mesh.

#     :arg mesh_dm: the `mesh.topology_dm`
#     :arg points: array with point indices forming the patch
#     :arg ordering_type: a `PETSc.Mat.OrderingType`
#     :arg prefix: the prefix associated with additional ordering options

#     :returns: the permuted array of points
#     '''
#     # Order points by decreasing topological dimension (interiors, faces, edges, vertices)
#     points = points[::-1]
#     if ordering_type == "natural":
#         return points
#     subgraph = [np.intersect1d(points, mesh_dm.getAdjacency(p), return_indices=True)[1] for p in points]
#     ia = np.cumsum([0] + [len(neigh) for neigh in subgraph]).astype(PETSc.IntType)
#     ja = np.concatenate(subgraph).astype(PETSc.IntType)
#     A = PETSc.Mat().createAIJ((len(points), )*2, csr=(ia, ja, np.ones(ja.shape, PETSc.RealType)), comm=PETSc.COMM_SELF)
#     A.setOptionsPrefix(prefix)
#     rperm, cperm = A.getOrdering(ordering_type)
#     indices = points[rperm.getIndices()]
#     A.destroy()
#     rperm.destroy()
#     cperm.destroy()
#     return indices

# class ASMStarPC(ASMPatchPC):
#     '''Patch-based PC using Star of mesh entities implmented as an
#     :class:`ASMPatchPC`.

#     ASMStarPC is an additive Schwarz preconditioner where each patch
#     consists of all DoFs on the topological star of the mesh entity
#     specified by `pc_star_construct_dim`.
#     '''

#     _prefix = "pc_star_"

#     def get_patches(self, V):
#         # PETSc.Sys.Print(f"ASMStarPC: get_patches, rank {fd.COMM_WORLD.rank}",comm = fd.COMM_SELF)
#         mesh = V._mesh
#         mesh_dm = mesh.topology_dm
#         if mesh.cell_set._extruded:
#             warning("applying ASMStarPC on an extruded mesh")

#         # Obtain the topological entities to use to construct the stars
#         opts = PETSc.Options(self.prefix)
#         depth = opts.getInt("construct_dim", default=0)
#         ordering = opts.getString("mat_ordering_type", default="natural")
#         # Accessing .indices causes the allocation of a global array,
#         # so we need to cache these for efficiency
#         V_local_ises_indices = tuple(iset.indices for iset in V.dof_dset.local_ises)

#         # Build index sets for the patches
#         ises = []
#         (start, end) = mesh_dm.getDepthStratum(depth)
#         for seed in range(start, end):
#             # Only build patches over owned DoFs
#             if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
#                 continue

#             # Create point list from mesh DM
#             pt_array, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
#             pt_array = order_points(mesh_dm, pt_array, ordering, self.prefix)

#             # Get DoF indices for patch
#             indices = []
#             for (i, W) in enumerate(V):
#                 section = W.dm.getDefaultSection()
#                 for p in pt_array.tolist():
#                     dof = section.getDof(p)
#                     if dof <= 0:
#                         continue
#                     off = section.getOffset(p)
#                     # Local indices within W
#                     W_indices = slice(off*W.block_size, W.block_size * (off + dof))
#                     indices.extend(V_local_ises_indices[i][W_indices])
#             iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
#             ises.append(iset)
#         return ises



class CyclicReductionPC2(PCBase):

    _prefix = '_CR2'

    def initialize(self,pc):
        PETSc.Sys.Print("CyclicReductionPC2: Initializing preconditioner...")
        _, self.A = pc.getOperators() # <class 'petsc4py.PETSc.Mat'>
        # PETSc.Sys.Print(f"self.A.getOwnershipRanges(): {self.A.getOwnershipRanges()}")
        # PETSc.Sys.Print(f"self.A.getOwnershipRange() for rank {fd.COMM_WORLD.rank}: {self.A.getOwnershipRange()}",comm= fd.COMM_SELF)
        #  PETSc.Sys.Print(f"dir(A) : {dir(self.A)}")
        # PETSc.Sys.Print(f"comm(A) : {self.A.comm}")
        # PETSc.Sys.Print(f"get comm(A) : {self.A.getComm()}")
        # PETSc.Sys.Print(f"rank (A) : {self.A.getComm().rank}")
        # PETSc.Sys.Print(f"ownership(A) : {self.A.getOwnershipRange()}")
        # self.A.view()
        dm = pc.getDM()
        try:
            V = get_function_space(dm)
        except Exception as e:
            PETSc.Sys.Print(f"Error in get_function_space: {e}")

        assert V is not None, "Function space V is None"

        self.star_pc = ASMStarPC()
        self.star_pc.initialize(pc)

        self.patches = self.star_pc.get_patches(V)
        self.submatrices = []


        for i, patch in enumerate(self.patches):
            # if fd.COMM_WORLD.rank == 1:
            #     PETSc.Sys.Print(f"Rank {fd.COMM_WORLD.rank}, patch {i}: {patch.view()}\n A ownership range: {self.A.getOwnershipRange()}, A local size: {self.A.getLocalSize()}",comm=fd.COMM_SELF)
            # self.submatrices.append(self.A.getLocalSubMatrix(patch,patch))
            self.submatrices.append(self.A.createSubMatrix(patch,patch))


        
    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        _, A = pc.getOperators() # <class 'petsc4py.PETSc.Mat'>
        PETSc.Sys.Print(f"self.A.getOwnershipRanges(): {A.getOwnershipRanges()}")
        PETSc.Sys.Print(f"self.A.getOwnershipRange() for rank {fd.COMM_WORLD.rank}: {A.getOwnershipRange()}",comm= fd.COMM_SELF)

        self.subvectors = []
        
        for i, patch in enumerate(self.patches):
            subvec = PETSc.Vec().create()
            subvec.setSizes(len(patch.indices))
            subvec.setFromOptions()

            # Scatter values from x to subvec
            scatter = PETSc.Scatter().create(x, patch, subvec, None)
            scatter.scatter(x, subvec, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            
            self.subvectors.append(subvec)
        

        for lhs, rhs, patch in zip(self.submatrices, self.subvectors, self.patches):
            ksp = PETSc.KSP().create()
            ksp.setOperators(lhs)
            ksp.setFromOptions()
            ksp.setUp()
            ksp.solve(rhs, rhs)

            # Scatter values from rhs to y based on patch
            # scatter = PETSc.Scatter().create(rhs, patch, y, None)
            # with y as yvec:
            #     with rhs.local_vec_ro() as rhsvec:
            #         yvec.setArray(rhsvec.array)
            # scatter.scatter(rhs, y, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD) 
        y.scale(0)
        PETSc.Sys.Print(f"I arrive here")
        
    def applyTranspose(self, pc, x, y):
        pass

    def get_patches(self, V):
        super().get_patches(self, V)

    def view(pc, viewer=None):
        pass

    def destroy(self, pc):
        pass
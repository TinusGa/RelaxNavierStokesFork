"""
Test script solving the heat equation using standard space-time
finite elements
"""
#Global imports
from firedrake import *
from firedrake.petsc import PETSc
import matplotlib.pylab as plt
from time import time

class CyclicReduction(PCBase):
    
    _prefix = "CR_"
    
    def initialize(self, pc):
        # Get context from pc
        _, P = pc.getOperators()
        #print(type(pc))
        #print(dir(pc))
        dm = pc.getDM()
        print(type(dm))
        print(dir(dm))
        self.prefix = pc.getOptionsPrefix() + self._prefix

        # Extract function space and mesh to obtain plex and indexing functions
        V = functionspace(dm)

        # Obtain patches from user defined function
        ises = self.get_patches(V)
        # PCASM expects at least one patch, so we define an empty one on idle processes
        if len(ises) == 0:
            ises = [PETSc.IS().createGeneral(numpy.empty(0, dtype=IntType), comm=PETSc.COMM_SELF)]

        # Create new PC object as ASM type and set index sets for patches
        asmpc = PETSc.PC().create(comm=pc.comm)
        asmpc.incrementTabLevel(1, parent=pc)
        asmpc.setOptionsPrefix(self.prefix + "sub_")
        asmpc.setOperators(*pc.getOperators())

        opts = PETSc.Options(self.prefix)
        backend = opts.getString("backend", default="petscasm").lower()
        # Either use PETSc's ASM PC or use TinyASM (as simple ASM
        # implementation designed to be fast for small block sizes).
        if backend == "petscasm":
            asmpc.setType(asmpc.Type.ASM)
            # Set default solver parameters
            asmpc.setASMType(PETSc.PC.ASMType.BASIC)
            sub_opts = PETSc.Options(asmpc.getOptionsPrefix())
            if "sub_pc_type" not in sub_opts:
                sub_opts["sub_pc_type"] = "lu"
            if "sub_pc_factor_mat_ordering_type" not in sub_opts:
                # Preserve the natural ordering to avoid zero pivots in saddle-point problems
                sub_opts["sub_pc_factor_mat_ordering_type"] = "natural"

            # If an ordering type is provided, PCASM should not sort patch indices, otherwise it can.
            mat_type = P.getType()
            if not mat_type.endswith("sbaij"):
                sentinel = object()
                ordering = opts.getString("mat_ordering_type", default=sentinel)
                asmpc.setASMSortIndices(ordering is sentinel)

            lgmap = V.dof_dset.lgmap
            # Translate to global numbers
            ises = tuple(lgmap.applyIS(iset) for iset in ises)
            asmpc.setASMLocalSubdomains(len(ises), ises)
        elif backend == "tinyasm":
            _, P = asmpc.getOperators()
            lgmap = V.dof_dset.lgmap
            P.setLGMap(rmap=lgmap, cmap=lgmap)

            asmpc.setType("tinyasm")
            # TinyASM wants local numbers, no need to translate
            tinyasm.SetASMLocalSubdomains(
                asmpc, ises,
                [W.dm.getDefaultSF() for W in V],
                [W.block_size for W in V],
                sum(W.block_size * W.dof_dset.total_size for W in V))
            asmpc.setUp()
        else:
            raise ValueError(f"Unknown backend type {backend}")

        asmpc.setFromOptions()
        self.asmpc = asmpc

        self._patch_statistics = []
        if opts.getBool("view_patch_sizes", default=False):
            # Compute and stash patch statistics
            mpi_comm = pc.comm.tompi4py()
            max_local_patch = max(is_.getSize() for is_ in ises)
            min_local_patch = min(is_.getSize() for is_ in ises)
            sum_local_patch = sum(is_.getSize() for is_ in ises)
            max_global_patch = mpi_comm.allreduce(max_local_patch, op=MPI.MAX)
            min_global_patch = mpi_comm.allreduce(min_local_patch, op=MPI.MIN)
            sum_global_patch = mpi_comm.allreduce(sum_local_patch, op=MPI.SUM)
            avg_global_patch = sum_global_patch / mpi_comm.allreduce(len(ises) if sum_local_patch > 0 else 0, op=MPI.SUM)
            msg = f"Minimum / average / maximum patch sizes : {min_global_patch} / {avg_global_patch} / {max_global_patch}\n"
            self._patch_statistics.append(msg)
     
    def get_patches(self, V):
        ''' Get the patches used for PETSc PCASM

        :param  V: the :class:`~.FunctionSpace`.

        :returns: a list of index sets defining the ASM patches in local
            numbering (before lgmap.apply has been called).
        '''
        pass

    def view(self, pc, viewer=None):
        self.asmpc.view(viewer=viewer)
        if viewer is not None:
            for msg in self._patch_statistics:
                viewer.printfASCII(msg)

    def update(self, pc):
        # This is required to update an inplace ILU factorization
        if self.asmpc.getType() == "asm":
            for sub in self.asmpc.getASMSubKSP():
                sub.getOperators()[0].setUnfactored()

    def apply(self, pc, x, y):
        self.asmpc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.asmpc.applyTranspose(x, y)

    def destroy(self, pc):
        if hasattr(self, "asmpc"):
            self.asmpc.destroy()
    


#Problem parameters used if running this script
class parameters:
    def __init__(self):
        self.N = 50
        self.dt = 0.001 #Specified instead of end time
        self.M = 20
        self.Mbase = 5
        self.Mref = 2
        self.degree = {'space': 1,
                       'time': 1}
        self.plot = True
        self.solver = None


#Solve the heat equation with timings
def heat(para=parameters):

    start = time()

    def plus(v):
        return -0.5*jump(v,n[2]) + avg(v)
    
    #Define mesh
    distribution_parameters={"partition": True,
                             "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    base_ = UnitSquareMesh(para.Mbase,para.Mbase,
                           distribution_parameters=distribution_parameters)
    spatial_mh = MeshHierarchy(base_,para.Mref)
    mh = ExtrudedMeshHierarchy(spatial_mh, para.N*para.dt,
                        base_layer = para.N,
                        refinement_ratio=1,
                        extrusion_type='uniform')
    mesh = mh[-1]
    n = FacetNormal(mesh)
    
    #Define function space
    space_element = FiniteElement("CG", triangle, para.degree['space'])
    time_element = FiniteElement("DG", interval, para.degree['time'])
    spacetime_element = TensorProductElement(space_element,time_element)
    U = FunctionSpace(mesh,spacetime_element)

    #Define initial condition
    x, y, t = SpatialCoordinate(U.mesh())
    u0 = interpolate(sin(pi*x)+cos(2*pi*y), U)

    #Set up residual
    u = Function(U)
    phi = TestFunction(U)

    gradu = as_vector([u.dx(0),
                       u.dx(1)])
    gradphi = as_vector([phi.dx(0),
                         phi.dx(1)])
    
    F_space = inner(gradu,gradphi) * dx(degree=16)
    F_time = u.dx(2) * phi * dx(degree=16) - jump(u,n[2]) * plus(phi) * dS_h(degree=16)
    F_ic = 0.5*(u-u0)*phi*ds_b

    F = F_space + F_time + F_ic

    #Set up solver
    if para.solver=='lu':
        solver_parameters = {'mat_type': 'aij',
                             'ksp_type': 'preonly',
                             "pc_factor_mat_solver_type":"mumps",
                             'pc_type': 'lu'}
    else:
        solver_parameters = {'snes_type': 'ksponly',
                             'mat_type': 'aij',
                             'ksp_type': 'fgmres',
                             "ksp_monitor_true_residual": None,
                             "ksp_max_it": 100,
                             "ksp_gmres_restart": 100,
                             "ksp_atol": 1e-6,
                             "ksp_rtol": 1e-6,
                             'pc_type': 'mg',
                             "pc_mg_type": "multiplicative",
                             "pc_mg_cycles": "v",
                             "mg_levels_ksp_type": "chebyshev",
                             "mg_levels_ksp_chebyshev_esteig": "0,0.25,0,1.05",
                             "mg_levels_ksp_max_it": 2,
                             "mg_levels_ksp_convergence_test": "skip",
                             "mg_levels_pc_type": "python",
                             "mg_levels_pc_python_type": "firedrake.ASMStarPC",
                             "mg_levels_pc_star_construct_dim": 0,
                             #"mg_levels_pc_star_sub_sub_pc_type": "lu",
                             #"mg_levels_pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
                             "mg_levels_pc_star_sub_sub_pc_type":"python",
                             "mg_levels_pc_star_sub_sub_pc_python_type": __name__ + ".CyclicReduction",
                             "mg_coarse_pc_type": "python",
                             "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                             "mg_coarse_assembled_pc_type": "lu",
                             "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                             }
        
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    start_solve = time()

    #Solve
    solver.solve()

    end = time()

#     iterations = solver.snes.getLinearSolveIterations()

#     print('iterations', iterations)

#     #Get number of nonzero entries
#     A, P = solver.snes.ksp.getOperators()
#     nnz = int(A.getInfo()['nz_allocated'])
    
#     #Plot
#     if para.plot:
#         ufile = File('plots/heat.pvd')
#         u.rename("u","u")
#         ufile.write(u)

#     #Output relevant info
#     out = {'dof': U.dim(),
#            'nnz': nnz,
#            'iterations': iterations,
#            'time_total': end-start,
#            'time_solve': end-start_solve}

#     return out



if __name__=="__main__":
    print(heat(parameters()))

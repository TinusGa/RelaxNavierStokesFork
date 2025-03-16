"""
Lid driven cavity
"""
#Global imports
from firedrake import *
from firedrake.petsc import PETSc
import matplotlib.pylab as plt
import numpy as np
from time import time
from pyop2.datatypes import IntType

#Parallel safe printing
Print = PETSc.Sys.Print

class parameters:
    def __init__(self):
        self.N = 5
        self.dt = 0.001 #Specified instead of end time
        self.Mbase = 10
        self.Mref = 3
        self.degree = {'space': 2}
        self.R = Constant(100) #Reynolds number
        self.alpha = Constant(1) #Diffusion constant
        self.plot = False
        self.solver = 'one_step'
    

def lid(para=parameters):
    
    start = time()
    
    def plus(v):
        return -0.5*jump(v,n[2]) + avg(v)
    
    number_of_ranks : int = COMM_WORLD.size
    number_of_temporal_ranks : int = 2

    if number_of_ranks % number_of_temporal_ranks != 0:
            raise ValueError("Number of time slices must be exact factor of number of MPI ranks")

    number_of_spatial_ranks : int = number_of_ranks // number_of_temporal_ranks

    PETSc.Sys.Print('Setting up mesh across %d processes. There are %d spatial processes, each with %d temporal processes' % (number_of_ranks, number_of_spatial_ranks, number_of_temporal_ranks))
    my_ensemble = Ensemble(COMM_WORLD, number_of_spatial_ranks)
    
    #Define mesh
    distribution_parameters={"partition": True,
                             "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    
    base_ = UnitSquareMesh(para.Mbase,para.Mbase,
                           distribution_parameters=distribution_parameters, comm=my_ensemble.comm)

    # base_ = UnitSquareMesh(para.Mbase,para.Mbase,
    #                         distribution_parameters=distribution_parameters)

    PETSc.Sys.Print(f"Ensemble comm rank: {my_ensemble.ensemble_comm.rank}")
    PETSc.Sys.Print(f"Comm name: {my_ensemble.comm.name}") # Spatial comm
    PETSc.Sys.Print(f"Comm size: {my_ensemble.comm.size}") # Spatial comm
    PETSc.Sys.Print(f"Ensemble comm ranme: {my_ensemble.ensemble_comm.name}") # Should be temporal comm

    spatial_mh = MeshHierarchy(base_,para.Mref)

    mesh = spatial_mh[-1]

    PETSc.Sys.Print('  rank %d owns %d elements and can access %d vertices' \
                % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices()),
                comm=my_ensemble.global_comm)

    n = FacetNormal(mesh)
    
    #Define function space
    CG2 = VectorFunctionSpace(mesh,"CG",para.degree['space']+1,dim=2)
    CG1 = FunctionSpace(mesh,"CG",para.degree['space'])
    Z = MixedFunctionSpace((CG2,CG1))
    
    #Define initial condition
    x, y = SpatialCoordinate(Z.mesh())

    z0 = Function(Z)
    u0, p0 = z0.subfunctions
    
    #Set up residual
    z = Function(Z)
    u, p = split(z)
    phi, psi = TestFunctions(Z)

    gradu = as_vector([u.dx(0),
                       u.dx(1)])

    gradphi = as_vector([phi.dx(0),
                         phi.dx(1)])
    
    F_1 = para.R * inner(dot(u, gradu),phi) * dx \
        + p * (phi[0].dx(0) + phi[1].dx(1)) * dx \
        + para.alpha * inner(gradu,gradphi) * dx

    F_2 = (u[0].dx(0) + u[1].dx(1)) * psi * dx

    F_time = para.dt**(-1)*inner(u - u0, phi) * dx

    F = F_1 + F_2 + F_time
    F = para.dt * F #rescale

    class PressureFixBC(DirichletBC):
        def __init__(self, V, val, subdomain):
            super().__init__(V, val, subdomain)
            sec = V.dm.getDefaultSection()
            dm = V.mesh().topology_dm
            coordsSection = dm.getCoordinateSection()
            coordsDM = dm.getCoordinateDM()
            dim = dm.getCoordinateDim()
            coordsVec = dm.getCoordinatesLocal()
            (vStart, vEnd) = dm.getDepthStratum(0)
            indices = []
            for pt in range(vStart, vEnd):
                x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
                if (x[1]==0.) and (x[0]==0.):
                    if dm.getLabelValue("pyop2_ghost", pt) == -1:
                        indices.append(pt)
            nodes = []
            for i in indices:
                num_of_layers = sec.getDof(i)
                if num_of_layers > 0:
                    off = sec.getOffset(i)
                    nodes.append(np.arange(off,off+num_of_layers))

            self.nodes = np.asarray(nodes, dtype=IntType)

            Print("Fixing nodes %s" % self.nodes)
    
    bc1 = DirichletBC(Z.sub(0),as_vector((0,0)),(1,2,3))
    bc2 = DirichletBC(Z.sub(0),as_vector((1,0)),4)
    bc3 = PressureFixBC(Z.sub(1),Constant(0),1)
    
    
    #Set up solver
    if para.solver=='lu':
        solver_parameters = {'mat_type': 'aij',
                             'ksp_type': 'preonly',
                             "pc_factor_mat_solver_type":"mumps",
                             "pc_factor_shift_type":"nonzero",
                             'pc_type': 'lu',
                             'snes_max_it': 1000,
                             'snes_rtol': 1e-6,
                             'snes_stol': 1e-6,
                             'snes_monitor': None}
    elif para.solver=='one_step':    
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
                             "mg_levels_pc_python_type": __name__ + ".ASMVankaStarPC",
                             "mg_levels_pc_vankastar_construct_dim": 0,
                             "mg_levels_pc_vankastar_exclude_subfunctions": "1,2",
                             "mg_levels_pc_vankastar_sub_sub_pc_type": "lu",
                             "mg_levels_pc_vankastar_sub_sub_pc_factor_mat_solver_type": "umfpack",
                             "mg_coarse_pc_type": "python",
                             "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                             "mg_coarse_assembled_pc_type": "lu",
                             "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                             }
    else:
        solver_parameters = {'snes_type': 'newtonls',
                             'snes_ksp_ew': None,
                             'snes_monitor': None,
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
                             "mg_levels_pc_python_type": __name__ + ".ASMVankaStarPC",
                             "mg_levels_pc_vankastar_construct_dim": 0,
                             "mg_levels_pc_vankastar_exclude_subfunctions": "1",
                             "mg_levels_pc_vankastar_sub_sub_pc_type": "lu",
                             "mg_levels_pc_vankastar_sub_sub_pc_factor_mat_solver_type": "umfpack",
                             "mg_coarse_pc_type": "python",
                             "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                             "mg_coarse_assembled_pc_type": "lu",
                             "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                             }
        
    problem = NonlinearVariationalProblem(F, z, bcs=[bc1,bc2,bc3])
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    t = 0 #initalise time
    #set up iteration counters
    iterations = []
    nl_iterations = []
    #Initialise plots
    if para.plot:
        zfile = File('plots/lid_stepper.pvd')
        u0, p0 = z0.subfunctions
        u0.rename('u','u')
        p0.rename('p','p')
        zfile.write(u0,p0, time=0)
        
    
    start_solve = time()
    while t<para.N * para.dt:
        #Solve
        #A,_ = solver.snes.ksp.getOperators()
        # print(A.getInfo())
        # print(A.getSize())
        # print(A.getSizes())
        # print(type(A))
        solver.solve()
        z0.assign(z)
        t += para.dt
        

        #Save iteration counts
        iterations.append(solver.snes.getLinearSolveIterations())
        nl_iterations.append(solver.snes.getIterationNumber())

        #Save solution (if plotting)
        if para.plot:
            zfile.write(u0,p0, time=t)
            
        

    end = time()

    print('iterations', iterations)
    print(np.mean(iterations))
    #print('nl iterations', nl_iterations)
    #print(np.mean(nl_iterations))
    
        
    #Output relevant info
    out = {'dof': Z.dim(),
           'iterations': np.mean(iterations),
           'newton iterations': np.mean(nl_iterations),
           'time_total': end-start,
           'time_solve': end-start_solve}

    

    return out



def order_points(mesh_dm, points, ordering_type, prefix):
    '''Order a the points (topological entities) of a patch based                                                        
    on the adjacency graph of the mesh.                                                                               
    :arg mesh_dm: the `mesh.topology_dm`                                                                                 
    :arg points: array with point indices forming the patch                                                              
    :arg ordering_type: a `PETSc.Mat.OrderingType`                                                                       
    :arg prefix: the prefix associated with additional ordering options                                              

    :returns: the permuted array of points                                                                            
    '''
    if ordering_type == "natural":
        return points
    subgraph = [np.numpy.intersect1d(points, mesh_dm.getAdjacency(p), return_indices=True)[1] for p in points]
    ia = np.numpy.cumsum([0] + [len(neigh) for neigh in subgraph]).astype(PETSc.IntType)
    ja = np.numpy.concatenate(subgraph).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ((len(points), )*2, csr=(ia, ja, np.numpy.ones(ja.shape, PETSc.RealType)), comm=PETSc.COMM_SELF)
    #A = PETSc.Mat().createAIJ((len(points), )*2, csr=(ia, ja, np.numpy.ones(ja.shape, PETSc.RealType)), comm = Ensemble(COMM_WORLD, 4).comm)
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
            pt_array_star = order_points(mesh_dm, star, ordering, self.prefix)
            
            pt_array_vanka = set()
            for pt in star.tolist():
                closure, _ = mesh_dm.getTransitiveClosure(pt, useCone=True)
                pt_array_vanka.update(closure.tolist())

            pt_array_vanka = order_points(mesh_dm, pt_array_vanka, ordering, self.prefix)
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


if __name__=="__main__":
    Print(lid(parameters()))

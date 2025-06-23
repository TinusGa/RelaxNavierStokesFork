"""
Test script solving the heat equation using standard space-time
finite elements
"""
#Global imports
from firedrake import *
import matplotlib.pylab as plt
from time import time
import warnings
warnings.simplefilter("ignore", FutureWarning)

#Problem parameters used if running this script
class parameters:
    def __init__(self):
        self.N = 8
        self.dt = 0.001 #Specified instead of end time
        self.M = 0
        self.Mbase = 4
        self.Mref = 1
        self.degree = {'space': 2,
                       'time': 0} # DG degree 0 gives backward Euler
        self.plot = True
        self.solver = None

# Solve the heat equation with timings
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
    u0 = Function(U)
    # u0 = interpolate(sin(pi*x)+cos(2*pi*y), U)
    u0.interpolate(cos(pi*x)*cos(2*pi*y))

    bcs = []

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
                             "ksp_atol": 1e-12,
                             "ksp_rtol": 1e-12,
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
                             "mg_levels_pc_star_sub_sub_pc_type": "lu",
                             "mg_levels_pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
                             "mg_coarse_pc_type": "python",
                             "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                             "mg_coarse_assembled_pc_type": "lu",
                             "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                             }
    
    problem = NonlinearVariationalProblem(F, u, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    start_solve = time()

    #Solve
    solver.solve()

    end = time()

    iterations = solver.snes.getLinearSolveIterations()

    PETSc.Sys.Print('iterations', iterations)
    PETSc.Sys.Print('time taken: ', end-start_solve)
    #Get number of nonzero entries
    A, P = solver.snes.ksp.getOperators()
    nnz = int(A.getInfo()['nz_allocated'])
    
    #Plot
    # if para.plot:
    #     ufile = File('plots/heat.pvd')
    #     u.rename("u","u")
    #     ufile.write(u)

    #Output relevant info
    out = {'dof': U.dim(),
           'nnz': nnz,
           'iterations': iterations,
           'time_total': end-start,
           'time_solve': end-start_solve}

    return out

if __name__=="__main__":
    PETSc.Sys.Print(heat(parameters()))

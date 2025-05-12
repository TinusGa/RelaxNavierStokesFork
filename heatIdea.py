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
        self.N = 257
        self.dt = 0.001 #Specified instead of end time
        self.M = 25
        self.Mbase = 5
        self.Mref = 2
        self.degree = {'space': 1,
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
    mesh = UnitSquareMesh(para.M,para.M,
                           distribution_parameters=distribution_parameters)
    
    mesh = ExtrudedMesh(mesh = mesh, layers = para.N, layer_height = para.dt, extrusion_type='uniform')

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

    solver_parameters = {'snes_type': 'ksponly',
                         'mat_type': 'aij',
                         'ksp_type': 'preonly',
                         'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps'
                        }
    
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    A,_ = solver.snes.ksp.getOperators()
    PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")

    start_solve = time()

    #Solve
    solver.solve()

    end = time()

    iterations = solver.snes.getLinearSolveIterations()

    # print('iterations', iterations)
    # print('time taken: ', end-start_solve)
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

from firedrake import *
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", degree = 2)
u = TrialFunction(V)
v = TestFunction(V)
x, y = SpatialCoordinate(mesh)
f = Function(V)
f.interpolate(sin(pi*x)*sin(2*pi*y))
a = inner(grad(u), grad(v)) * dx
L = f*v*dx

bcs = [DirichletBC(V, 0, "on_boundary")]
solution = Function(V, name = "solution")
# solver_parameters = {'ksp_monitor':None,'ksp_rtol': 1e-12, 'ksp_atol': 1e-12}
problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
solver = LinearVariationalSolver(problem)
solver.solve()

import shutil
import os
from pyop2.mpi import MPI
from firedrake import VTKFile

file_dir = "Poisson_2D"

# Clear directory if it exists
if COMM_WORLD.rank == 0:
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
    os.makedirs(f"{file_dir}", exist_ok=True)


vtkfile = VTKFile(f"{file_dir}/u_t.pvd")
vtkfile.write(solution)

u_exact = Function(V)
factor = Constant(1/(5*pi**2))
u_exact.interpolate(factor*sin(pi*x)*sin(2*pi*y))

PETSc.Sys.Print(f"error: {errornorm(u_exact, solution, norm_type="L2")}")

A = solver.snes.ksp.getOperators()[0]
PETSc.Sys.Print(f"Size A: {A.getSize()}, with ownership ranges: {A.getOwnershipRanges()}")
node_set = V.node_set.size
PETSc.Sys.Print(f"RANK {COMM_WORLD.rank}. Number of nodes in the set: {node_set}",comm=COMM_SELF)
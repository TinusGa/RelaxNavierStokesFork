from firedrake import *
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

mesh = UnitSquareMesh(5, 5)
extruded_mesh = ExtrudedMesh(mesh, layers=1, layer_height=0.2)

space_element = FiniteElement("CG", triangle, 1)
time_element = FiniteElement("DG", interval, 1)
spacetime_element = TensorProductElement(space_element,time_element)
U = FunctionSpace(extruded_mesh,spacetime_element)

x, y, t = SpatialCoordinate(U.mesh())
u0 = interpolate(sin(pi*x)+cos(2*pi*y), U)

u = Function(U)
u.assign(u0)

vtkfile = VTKFile(f"{'MeshView'}/u_t.pvd")
vtkfile.write(u, time=0.0)


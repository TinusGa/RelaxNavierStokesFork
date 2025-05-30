from firedrake import *

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "CG", 2)

u = Function(V)
u.dat.data[:] = range(V.dim())  # Label each DoF with its global number

u.dat._vec.view()

VTKFile("cg2_dof_numbers.pvd").write(u)

############################################################
# from firedrake import *
# import matplotlib.pyplot as plt
# from firedrake.pyplot import triplot
# import numpy as np

# # Create a mesh and function space
# mesh = UnitSquareMesh(4, 4)
# V = FunctionSpace(mesh, "CG", 2)
# rank = mesh.comm.rank

# # Extract owned vertex coordinates
# coords = mesh.coordinates.dat.data_ro.copy()

# # Plot mesh
# fig, ax = plt.subplots()
# triplot(mesh, axes=ax)
# ax.scatter(coords[:, 0], coords[:, 1], c=[rank]*len(coords), cmap="tab10", s=60, edgecolors="k", label=f"Rank {rank}")
# ax.set_title(f"DoF Ownership (Rank {rank})")
# ax.set_aspect("equal")
# ax.legend()
# plt.savefig(f"dof_ownership_rank{rank}.png", dpi=150)
# plt.close(fig)




######################## LATEX EXPORT #######################

# from firedrake import *
# from firedrake.pyplot.pgf import pgfplot

# # Create a 2D mesh and function space
# mesh = UnitSquareMesh(4, 4)
# V = FunctionSpace(mesh, "CG", 1)

# # Define a scalar function
# f = Function(V)
# x, y = SpatialCoordinate(mesh)
# f.interpolate(sin(pi * x) * sin(pi * y))

# # Export to PGFPlots-compatible .tex file
# pgfplot(f, "output_plot.tex", degree=1)

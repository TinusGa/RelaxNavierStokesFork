# import matplotlib.pyplot as plt
# from firedrake import *
# from firedrake.pyplot import triplot

# # Create a unit square mesh
# mesh = UnitSquareMesh(4, 4)
# V = FunctionSpace(mesh, "CG", 1)

# coordinates = mesh.coordinates.dat.data_ro
# print(coordinates)
# print(len(coordinates))

# # V_local_ises_indices = tuple(iset.indices for iset in V.dof_dset.local_ises)
# for iset in V.dof_dset:
#     local_ises = iset.local_ises
#     global_ises = iset.lgmap.apply(local_ises)  # This will convert local indices to global indices
#     print(global_ises,len(global_ises))

# # print("V_local_ises_indices:", V_local_ises_indices)

# # Plot the mesh
# fig, ax = plt.subplots()
# triplot(mesh, axes=ax)
# ax.set_aspect('equal')
# plt.title("Mesh Visualization")
# plt.savefig("mesh_plot.png", dpi=150)
# plt.close(fig)

############################################################
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import triplot
import numpy as np

# Create a mesh and function space
mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 2)
rank = mesh.comm.rank

# Extract owned vertex coordinates
coords = mesh.coordinates.dat.data_ro.copy()

# Plot mesh
fig, ax = plt.subplots()
triplot(mesh, axes=ax)
ax.scatter(coords[:, 0], coords[:, 1], c=[rank]*len(coords), cmap="tab10", s=60, edgecolors="k", label=f"Rank {rank}")
ax.set_title(f"DoF Ownership (Rank {rank})")
ax.set_aspect("equal")
ax.legend()
plt.savefig(f"dof_ownership_rank{rank}.png", dpi=150)
plt.close(fig)




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

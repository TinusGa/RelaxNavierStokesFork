from CyclicReduction import preconditioner
from petsc4py import PETSc
pc = PETSc.PC().create()
pc.setType("python")
pc.setPythonContext(preconditioner())
pc.setFromOptions()
print(pc.view())
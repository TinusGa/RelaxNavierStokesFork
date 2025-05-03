import numpy as np

import firedrake as fd
from firedrake.petsc import PETSc

from asQ.pencil import Pencil, Subcomm
from asQ.profiling import profiler
from asQ.preconditioners.base import AllAtOnceBlockPCBase
from asQ.parallel_arrays import SharedArray
from asQ.allatonce import time_average

from firedrake.preconditioners import ASMPatchPC, PCBase


__all__ = ['CyclicReductionPC2']


class CyclicReductionPC2(ASMPatchPC):

    _prefix = '_CR2'

    def initialize(self,pc):
        _, self.A = pc.getOperators() # <class 'petsc4py.PETSc.Mat'>
        PETSc.Sys.Print(f"Ownership range of A: {self.A.getOwnershipRange()}")
        PETSc.Sys.Print(f"Ownership ranges of A: {self.A.getOwnershipRanges()}")
        PETSc.Sys.Print(f"local size of A: {self.A.getLocalSize()}")
        PETSc.Sys.Print(f"size of A: {self.A.getSize()} \n \n")
        dm = self.A.getDM()

        PETSc.Sys.Print(f"dm dir: {dir(dm)}")

        # V = self.mesh._V
        # ises = self.get_patches(V)

        # if True:
        #     raise ValueError("This is a test error")
        
    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        
        PETSc.Sys.Print("CyclicReductionPC2: Applying preconditioner...")
        _, A = pc.getOperators() 
        self.ksp = PETSc.KSP().create(comm = fd.COMM_SELF)
        self.ksp.setOperators(self.A)
        self.ksp.setFromOptions()
        self.ksp.solve(x,y)
        # y = x.copy()
        # y.scale(0.5)

        # if True:
        #      raise ValueError("This is a test error")
        
    def applyTranspose(self, pc, x, y):
        pass

    def get_patches(self, V):
        super().get_patches(self, V)

    def view(pc, viewer=None):
        pass

    def destroy(self, pc):
        pass
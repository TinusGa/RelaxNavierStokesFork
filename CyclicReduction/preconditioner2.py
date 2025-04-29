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
        PETSc.Sys.Print(f"dir for self {dir(self)} \n \n")



        if True:
            raise ValueError("Stopping for simplicity")

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        
        PETSc.Sys.Print("CyclicReductionPC2: Applying preconditioner...")
        _, A = pc.getOperators() 
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(A)
        self.ksp.setFromOptions()
        self.ksp.solve(x,y)
        

    def applyTranspose(self, pc, x, y):
        pass

    def get_patches(self, V):
        pass

    def view(pc, viewer=None):
        pass

    def destroy(self, pc):
        pass
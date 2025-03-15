from asQ.preconditioners.base import AllAtOnceBlockPCBase

class CyclicReductionPC(AllAtOnceBlockPCBase):
     
     def initialize(self, pc):
        super().initialize(pc, final_initialize=False)
        #A,_ = pc.
from firedrake import *
import argparse
import pickle
import os

from heat import heat


if __name__=="__main__":
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10,
                        help = 'Number of time steps')
    parser.add_argument('--dt', type=float, default=0.01,
                        help = 'Time step')
    parser.add_argument('--tdegree', type=int, default=2,
                       help = 'Temporal degree')
    parser.add_argument('--sdegree', type=int, default=1,
                        help = 'Spatial degree')
    parser.add_argument('--alpha', type=float, default=1,
                        help = 'Diffusion parameter')
    parser.add_argument('--Mbase', type=int, default=10,
                        help = 'Number of points in the base mesh')
    parser.add_argument('--Mref', type=int, default = 2,
                        help = 'Number of refinement levels')
    parser.add_argument('--tmpname', type=str, default='_',
                        help = 'Add a string here to make temporary file name unique')
    parser.add_argument('--plot', action='store_true',
                        help = 'Set this flag to generate plots')
    parser.add_argument('--lu', action='store_true',
                        help = 'Solve linear system with LU')
    args, _ = parser.parse_known_args()

    class input_args:
        def __init__(self):
            self.N = args.N
            self.dt = args.dt
            self.Mbase = args.Mbase
            self.Mref = args.Mref
            self.degree = {'space': args.sdegree,
                           'time': args.tdegree}
            self.alpha = Constant(args.alpha)
            self.plot = args.plot
            if args.lu==True:
                self.solver = 'lu'
                if args.onestep==True:
                    raise ValueError('Cannot set lu and onestep at the same time')
            elif args.onestep==True:
                self.solver = 'one_step'
            else:
                self.solver = None

    out = heat(input_args())

    #Save output
    filename = 'tmp/heatdat%s%s%s%s%s%s%s%s' % (args.tmpname, args.N, args.dt,
                                                args.tdegree, args.sdegree,
                                                args.alpha, args.Mbase, args.Mref)
    file = open(filename,'wb')
    pickle.dump(out,file)
    file.close()

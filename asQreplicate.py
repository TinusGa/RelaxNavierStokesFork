import firedrake as fd
from pyop2.mpi import MPI
from numpy import zeros as zero_array
from numpy import asarray

time_partition = [2,2,2,2] # Requests 4 processors, each holding two timesteps. Spatial separation is determined by total MPI ranks

def create_ensemble(time_partition, comm = fd.COMM_WORLD):
    '''
    asQ-function:

    Create an Ensemble for the given slice partition.
    Checks that the number of slices and the size of the communicator are compatible.

    :arg time_partition: a list of integers, the number of timesteps on each time-rank
    :arg comm: the global communicator for the ensemble
    '''
    nslices = 1 if type(time_partition) is int else len(time_partition)
    nranks = comm.size

    if nranks % nslices != 0:
        raise ValueError("Number of time slices must be exact factor of number of MPI ranks")

    nspatial_domains = nranks//nslices

    return fd.Ensemble(comm, nspatial_domains)

ensemble = create_ensemble(time_partition)

mesh = fd.SquareMesh(nx=8, ny=8, L=1,
                  comm=ensemble.comm)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)
uinitial = fd.Function(V)
uinitial.project(fd.sin(x) + fd.cos(y))

def form_mass(u, v):
    return u*v*fd.dx

def form_function(u, v, t):
    return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

solver_parameters = {
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'snes_type': 'ksponly',
    'mat_type': 'matfree',
    'ksp_type': 'richardson',
    'ksp_rtol': 1e-10,
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': 1e-4,
    'circulant_block': {
        'ksp_rtol': 1e-6,
        'ksp_type': 'gmres',
        'pc_type': 'ilu',
    },
}

### Let's replicate this ###
# paradiag = asQ.Paradiag(
#     ensemble=ensemble,
#     form_mass=form_mass,
#     form_function=form_function,
#     ics=uinitial, dt=0.1, theta=0.5,
#     time_partition=time_partition,
#     solver_parameters=solver_parameters)
def in_range(i, length, allow_negative=True, throws=False):
    '''
    Is the index i within the range of length?
    :arg i: index to check
    :arg length: the number of elements in the range
    :arg allow_negative: is negative indexing allowed?
    :arg throws: if True, an IndexError is raised if the index is out of range
    '''
    if allow_negative:
        result = (-length <= i < length)
    else:
        result = (0 <= i < length)
    if throws and result is False:
        raise IndexError(f"Index {i} is outside the range {length}")
    return result

class DistributedDataLayout1D(object):
    def __init__(self, partition, comm=MPI.COMM_WORLD):
        '''
        A representation of a 1D set of data distributed over several MPI ranks.

        :arg partition: The number of data elements on each rank. Can be a list of integers, in which case len(partition) must be comm.size. Can be a single integer, in which case all ranks have the same number of elements.
        :arg comm: MPI communicator the data is distributed over.
        '''
        if isinstance(partition, int):
            partition = tuple(partition for _ in range(comm.size))
        else:
            if len(partition) != comm.size:
                raise ValueError(f"Partition size {len(partition)} not equal to comm size {comm.size}")
            partition = tuple(partition)
        self.partition = partition
        self.comm = comm
        self.rank = comm.rank
        self.nranks = comm.size
        self.local_size = partition[self.rank]
        self.global_size = sum(partition)
        self.offset = sum(partition[:self.rank])

    def transform_index(self, i, itype='l', rtype='l'):
        '''
        Shift index between local and global addressing, and transform negative indices to their positive equivalent.

        For example if there are 3 ranks each owning two elements then:
            global indices 0,1 are local indices 0,1 on rank 0.
            global indices 2,3 are local indices 0,1 on rank 1.
            global indices 4,5 are local indices 0,1 on rank 2.
        Negative indices are shifted to their positive equivalent:
            local index -1 becomes local index 1.
            global index -2 becomes local index 0 on rank 2.
        Throws IndexError if original or shifted index is out of bounds.

        :arg i: index to shift.
        :arg itype: type of index i. 'l' for local, 'g' for global.
        :arg rtype: type of returned shifted index. 'l' for local, 'g' for global.
        '''
        if itype not in ['l', 'g']:
            raise ValueError(f"itype {itype} must be either 'l' or 'g'")
        if rtype not in ['l', 'g']:
            raise ValueError(f"rtype {rtype} must be either 'l' or 'g'")

        # validate
        sizes = {'l': self.local_size, 'g': self.global_size}
        in_range(i, sizes[itype], throws=True)

        # deal with -ve index
        i = i % sizes[itype]

        # no shift needed
        if itype == rtype:
            return i
        else:
            if itype == 'l':  # rtype == 'g'
                i += self.offset
            elif itype == 'g':  # rtype == 'l'
                i -= self.offset
            in_range(i, sizes[rtype], allow_negative=False, throws=True)
            return i

    def is_local(self, i, throws=False):
        '''
        Is the globally addressed index i owned by this time rank?

        :arg i: globally addressed index.
        :arg throws: if True, raises IndexError if i is outside the global address range
        '''
        try:
            self.transform_index(i, itype='g', rtype='l')
            return True
        except IndexError:
            if throws:
                raise
            else:
                return False

    def rank_of(self, i):
        '''
        Return which rank element i lives on.

        :arg i: globally addressed index.
        '''
        i = self.transform_index(i, itype='g', rtype='g')
        for rank in range(self.nranks):
            begin = sum(self.partition[:rank])
            end = sum(self.partition[:rank+1])
            if begin <= i < end:
                return rank
        return -1
    
class TimePartitionMixin(object):
    """
    Mixin class for all-at-once types related to a timeseries
    distributed over the ranks of an Ensemble communicator.

    Provides the following member variables
    layout: a DistributedDataLayout describing the partition over the ensemble.
    ensemble: the time-parallel ensemble.
    time_partition: a list of integers for the number of timesteps stored on each ensemble rank.
    time_rank: the ensemble rank of the current process.
    nlocal_timesteps: the number of timesteps on the current ensemble member.
    ntimesteps: the number of timesteps in the entire time-series.
    """
    def __init__(self):
        pass

    def _time_partition_setup(self, ensemble, time_partition):
        """
        Sets the provided member variables. This is not implemented in the
        the __init__ method to accomodate child classes which will be
        instantiated by PETSc (and hence we have no control over the values
        passed to __init__).

        :arg ensemble: the time-parallel ensemble communicator.
        :arg time_partition: a list of integers for the number of timesteps stored on each ensemble rank.
        """
        self.layout = DistributedDataLayout1D(time_partition, ensemble.ensemble_comm)
        self.ensemble = ensemble
        self.time_partition = self.layout.partition
        self.time_rank = ensemble.ensemble_comm.rank
        self.nlocal_timesteps = self.layout.local_size
        self.ntimesteps = self.layout.global_size

class Paradiag(TimePartitionMixin):
    @profiler()
    def __init__(self, ensemble,
                 time_partition,
                 form_mass, form_function,
                 ics, dt, theta,
                 solver_parameters={},
                 appctx={}, bcs=[],
                 options_prefix="",
                 reference_state=None,
                 function_alpha=None, jacobian_alpha=None,
                 jacobian_mass=None, jacobian_function=None,
                 pc_mass=None, pc_function=None,
                 pre_function_callback=None, post_function_callback=None,
                 pre_jacobian_callback=None, post_jacobian_callback=None):
        """A class to implement paradiag timestepping.

        :arg ensemble: time-parallel ensemble communicator. The timesteps are partitioned
            over the ensemble members according to time_partition so
            ensemble.ensemble_comm.size == len(time_partition) must be True.
        :arg time_partition: a list of integers for the number of timesteps stored on each
            ensemble rank.
        :arg form_mass: a function that returns a linear form on ics.function_space()
            providing the time derivative mass operator for  the PDE w_t + f(w) = 0.
            Must have signature `def form_mass(*u, *v):` where *u and *v are a split(TrialFunction)
            and a split(TestFunction) from ics.function_space().
        :arg form_function: a function that returns a form on ics.function_space()
            providing f(w) for the PDE w_t + f(w) = 0.
            Must have signature `def form_function(*u, *v):` where *u and *v are a split(Function)
            and a split(TestFunction) from ics.function_space().
        :arg ics: a Function containing the initial conditions.
        :arg dt: float, the timestep size.
        :arg theta: float, implicit timestepping parameter.
        :arg solver_parameters: options dictionary for nonlinear solver.
        :arg appctx: A dictionary containing application context that is
            passed to the preconditioner if matrix-free.
        :arg bcs: a list of DirichletBC boundary conditions on ics.function_space.
        :arg options_prefix: an optional prefix used to distinguish PETSc options.
            Use this option if you want to pass options to the solver from the
            command line in addition to through the solver_parameters dict.
        :arg reference_state: A reference firedrake.Function in ics.function_space().
            Only needed if 'aaos_jacobian_state' or 'diagfft_state' is 'reference'.
        :arg function_alpha: float, circulant matrix parameter used in the nonlinear residual.
            This is used for the waveform relaxation method. If None then no circulant
            approximation used.
        :arg jacobian_alpha: float, circulant matrix parameter used in the Jacobian.
            This introduces the circulant approximation in the AllAtOnceJacobian but not in the
            nonlinear residual. If None then no circulant approximation used in the Jacobian.
        :arg jacobian_mass: equivalent to form_mass, but used to construct the AllAtOnceJacobian
            not the nonlinear residual.
        :arg jacobian_function: equivalent to form_function, but used to construct the
            AllAtOnceJacobian not the nonlinear residual.
        :arg pc_mass: equivalent to form_mass, but used to construct the preconditioner.
        :arg pc_function: equivalent to form_function, but used to construct the preconditioner.
        :arg pre_function_callback: A user-defined function that will be called immediately
            before residual assembly. This can be used, for example, to update a coefficient
            function that has a complicated dependence on the unknown solution.
        :arg post_function_callback: As above, but called immediately after residual assembly.
        :arg pre_jacobian_callback: As above, but called immediately before Jacobian assembly.
        :arg post_jacobian_callback: As above, but called immediately after Jacobian assembly.
        """
        self._time_partition_setup(ensemble, time_partition)

        # all-at-once function and form

        function_space = ics.function_space()
        self.aaofunc = AllAtOnceFunction(ensemble, time_partition,
                                         function_space)
        self.aaofunc.assign(ics)

        self.aaoform = AllAtOnceForm(self.aaofunc, dt, theta,
                                     form_mass, form_function,
                                     bcs=bcs, alpha=function_alpha)

        # all-at-once jacobian
        if jacobian_mass is None:
            jacobian_mass = form_mass
        if jacobian_function is None:
            jacobian_function = form_function

        self.jacobian_aaofunc = self.aaofunc.copy()

        self.jacobian_form = AllAtOnceForm(self.jacobian_aaofunc, dt, theta,
                                           jacobian_mass, jacobian_function,
                                           bcs=bcs, alpha=jacobian_alpha)

        # pass seperate forms to the preconditioner
        if pc_mass is not None:
            appctx['pc_form_mass'] = pc_mass
        if pc_function is not None:
            appctx['pc_form_function'] = pc_function

        self.solver = AllAtOnceSolver(self.aaoform, self.aaofunc,
                                      solver_parameters=solver_parameters,
                                      options_prefix=options_prefix, appctx=appctx,
                                      jacobian_form=self.jacobian_form,
                                      jacobian_reference_state=reference_state,
                                      pre_function_callback=pre_function_callback,
                                      post_function_callback=post_function_callback,
                                      pre_jacobian_callback=pre_jacobian_callback,
                                      post_jacobian_callback=post_jacobian_callback)

        # iteration counts
        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm)
        self.reset_diagnostics()

paradiag.solve(nwindows=1)



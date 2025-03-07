from firedrake import *
from firedrake.petsc import PETSc, OptionsManager, flatten_parameters
from pyop2.mpi import MPI
from numpy import zeros as zero_array
from numpy import asarray
from pyop2 import MixedDat
from functools import reduce
from operator import mul
import contextlib
from ufl.duals import is_primal, is_dual

time_partition = [2,2,2,2] # Requests 4 processors, each holding two timesteps. Spatial separation is determined by total MPI ranks

def create_ensemble(time_partition, comm = COMM_WORLD):
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

    return Ensemble(comm, nspatial_domains)

ensemble = create_ensemble(time_partition)

mesh = SquareMesh(nx=8, ny=8, L=1,
                  comm=ensemble.comm)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
uinitial = Function(V)
uinitial.project(sin(x) + cos(y))

def form_mass(u, v):
    return u*v*dx

def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx

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
#     ensemble=ensemble, check
#     form_mass=form_mass, check
#     form_function=form_function, check
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


def time_average(aaofunc, uout, uwrk, average='window'):
    """
    Compute the time average of an all-at-once function
    over either the entire window or the current slice.

    :arg aaofunc: AllAtOnceFunction to average.
    :arg uout: Function to save average into.
    :arg uwrk: Function to use as working buffer.
        TODO: make this optional once Ensemble.allreduce accepts MPI.IN_PLACE.
    :arg average: range of time-average.
        'window': compute over all timesteps in all-at-once function.
        'slice': compute only over timesteps on local ensemble member.
    """
    # accumulate over local slice
    uout.zero()
    for i in range(aaofunc.nlocal_timesteps):
        uout += aaofunc[i]

    if average == 'slice':
        nsamples = aaofunc.nlocal_timesteps
        uout /= Constant(nsamples)
    elif average == 'window':
        aaofunc.ensemble.allreduce(uout, uwrk)
        nsamples = aaofunc.ntimesteps
        uout.assign(uwrk/Constant(nsamples))
    else:
        raise ValueError(f"average type must be 'window' or 'slice', not {average}")

    return

class AllAtOnceForm(TimePartitionMixin):
    def __init__(self,
                 aaofunc, dt, theta,
                 form_mass, form_function,
                 bcs=[], alpha=None):
        """
        The all-at-once form representing the implicit theta-method (trapezium rule version)
        over multiple timesteps of a time-dependent finite-element problem.

        :arg aaofunction: AllAtOnceFunction to create the form over.
        :arg dt: the timestep size.
        :arg theta: implicit timestepping parameter.
        :arg form_mass: a function that returns a linear form on aaofunction.field_function_space
            providing the mass operator for the time derivative.
            Must have signature `def form_mass(*u, *v):` where *u and *v are a split(TrialFunction)
            and a split(TestFunction) from aaofunction.field_function_space.
        :arg form_function: a function that returns a form on aaofunction.field_function_space
            providing f(w) for the ODE w_t + f(w) = 0.
            Must have signature `def form_function(*u, *v):` where *u and *v are a split(Function)
            and a split(TestFunction) from aaofunction.field_function_space.
        :arg bcs: a list of DirichletBC boundary conditions on aaofunc.field_function_space.
        :arg alpha: float, circulant matrix parameter. if None then no circulant approximation used.
        """
        self._time_partition_setup(aaofunc.ensemble, aaofunc.time_partition)

        self.aaofunc = aaofunc
        self.field_function_space = aaofunc.field_function_space
        self.function_space = aaofunc.function_space

        self.dt = Constant(dt)
        self.t0 = Constant(0)
        self.time = tuple(Constant(0) for _ in range(self.aaofunc.nlocal_timesteps))
        for n in range((self.aaofunc.nlocal_timesteps)):
            self.time[n].assign(self.t0 + self.dt*(self.aaofunc.transform_index(n, from_range='slice', to_range='window') + 1))
        self.theta = Constant(theta)

        self.form_mass = form_mass
        self.form_function = form_function

        self.alpha = None if alpha is None else Constant(alpha)

        # should this make a copy of bcs instead of taking a reference?
        self.field_bcs = bcs
        self.bcs = self._set_bcs(self.field_bcs)

        for bc in self.bcs:
            bc.apply(aaofunc.function)

        # cofunction to assemble the nonlinear residual into
        self.F = AllAtOnceCofunction(self.ensemble, self.time_partition,
                                     aaofunc.field_function_space.dual())

        self.form = self._construct_form()

    def time_update(self, t=None):
        """
        Update the time points that the form is defined over.

        Default behaviour is to update the initial time t0 to be the
        time of the final timestep. The last timestep of the
        AllAtOnceFunction can then be used as the new initial condition.
        The time at each timestep is updated according to the initial time.

        :arg t: New initial time t0. If None then the current final
            time is used as the new initial time.
        """
        if t is not None:
            self.t0.assign(t)
        else:
            self.t0.assign(self.t0 + self.dt*self.ntimesteps)

        for n in range((self.nlocal_timesteps)):
            time_idx = self.aaofunc.transform_index(n, from_range='slice', to_range='window')
            self.time[n].assign(self.t0 + self.dt*(time_idx + 1))
        return

    def _set_bcs(self, field_bcs):
        """
        Create a list of  boundary conditions on the all-at-once function space corresponding
        to the boundary conditions `field_bcs` on a single timestep applied to every timestep.

        :arg field_bcs: a list of the boundary conditions to apply.
        """
        aaofunc = self.aaofunc
        is_mixed_element = isinstance(aaofunc.field_function_space.ufl_element(), MixedElement)

        bcs_all = []
        for bc in field_bcs:
            for step in range(aaofunc.nlocal_timesteps):
                if is_mixed_element:
                    cpt = bc.function_space().index
                else:
                    cpt = 0
                index = aaofunc._component_indices(step)[cpt]
                bc_all = DirichletBC(aaofunc.function_space.sub(index),
                                        bc.function_arg,
                                        bc.sub_domain)
                bcs_all.append(bc_all)

        return bcs_all

    
    def copy(self, aaofunc=None):
        """
        Return a copy of the AllAtOnceForm.

        :arg aaofunc: An optional AllAtOnceFunction. If present, the new AllAtOnceForm
            will be defined over aaofunc. If None, the new AllAtOnceForm will be defined
            over a copy of self.aaofunc.
        """
        if aaofunc is None:
            aaofunc = self.aaofunc.copy()

        return AllAtOnceForm(aaofunc, self.dt, self.theta,
                             self.form_mass, self.form_function,
                             bcs=self.field_bcs, alpha=self.alpha)

    
    def assemble(self, func=None, tensor=None):
        """
        Evaluates the form.

        By default the form will be evaluated at the state in self.aaofunc,
        and the result will be placed into self.F.

        :arg func: may optionally be an AllAtOnceFunction or a global PETSc Vec.
            if not None, the form will be evaluated at the state in `func`.
        :arg tensor: may optionally be an AllAtOnceCofunction, in which case
            the result will be placed into `tensor`.
        """
        if tensor is not None and not isinstance(tensor, AllAtOnceCofunction):
            raise TypeError(f"tensor must be an AllAtOnceCofunction, not {type(tensor)}")

        # Set the current state
        if func is not None:
            self.aaofunc.assign(func, update_halos=False)

        # Assembly stage
        # The residual on the DirichletBC nodes is set to zero,
        # so we need to make sure that the function conforms
        # with the boundary conditions.
        for bc in self.bcs:
            bc.apply(self.aaofunc.function)

        # Update the halos after enforcing the bcs so we
        # know they are correct. This doesn't make a
        # difference now because we only support constant
        # bcs, but it will be important once we support
        # time-dependent bcs.
        self.aaofunc.update_time_halos()

        assemble(self.form, bcs=self.bcs,
                    tensor=self.F.cofunction)

        if tensor:
            tensor.assign(self.F)
            result = tensor
        else:
            result = self.F
        return result

    def _construct_form(self):
        """
        Constructs the (possibly nonlinear) form for the all at once system.
        Specific to the implicit theta-method (trapezium rule version).
        """
        aaofunc = self.aaofunc

        funcs = split(aaofunc.function)

        ics = split(aaofunc.initial_condition)
        uprevs = split(aaofunc.uprev)

        form_mass = self.form_mass
        form_function = self.form_function

        test_funcs = TestFunctions(aaofunc.function_space)

        dt = self.dt
        theta = self.theta

        def get_components(i, funcs=None):
            return tuple(funcs[j] for j in aaofunc._component_indices(i))

        get_step = partial(get_components, funcs=funcs)
        get_test = partial(get_components, funcs=test_funcs)

        for n in range(self.nlocal_timesteps):

            if n == 0:  # previous timestep is ic or is on previous slice
                if self.time_rank == 0:
                    uns = ics
                    if self.alpha is not None:
                        uns = tuple(un + self.alpha*up for un, up in zip(uns, uprevs))
                else:
                    uns = uprevs
            else:
                uns = get_step(n-1)

            # current time level
            un1s = get_step(n)
            vs = get_test(n)

            # time derivative
            if n == 0:
                form = (1.0/dt)*form_mass(*un1s, *vs)
            else:
                form += (1.0/dt)*form_mass(*un1s, *vs)
            form -= (1.0/dt)*form_mass(*uns, *vs)

            # vector field
            form += theta*form_function(*un1s, *vs, self.time[n])
            form += (1.0 - theta)*form_function(*uns, *vs, self.time[n]-dt)

        return form


class AllAtOnceFunctionBase(TimePartitionMixin):
    def __init__(self, ensemble, time_partition, function_space):
        """
        A (co)function representing multiple timesteps of a time-dependent finite-element problem,
        i.e. the solution to an all-at-once system.

        :arg ensemble: time-parallel ensemble communicator. The timesteps are partitioned
            over the ensemble members according to time_partition so
            ensemble.ensemble_comm.size == len(time_partition) must be True.
        :arg time_partition: a list of integers for the number of timesteps stored on each
            ensemble rank.
        :arg function_space: a Space for the a single timestep.
            Either `FunctionSpace` or `DualSpace` depending if the child is AAO(Co)Function.
        """
        self._time_partition_setup(ensemble, time_partition)

        # function space for single timestep
        self.field_function_space = function_space

        # function space for the slice of the all-at-once system on this process
        self.function_space = reduce(mul, (self.field_function_space
                                           for _ in range(self.nlocal_timesteps)))

        self.ncomponents = len(self.field_function_space.subfunctions)

        # this will be renamed either self.function or self.cofunction
        self._fbuf = Function(self.function_space)

        # Functions to view each timestep
        def field_function(i):
            if self.ncomponents == 1:
                j = self._component_indices(i)[0]
                dat = self._fbuf.subfunctions[j].dat
            else:
                dat = MixedDat((self._fbuf.subfunctions[j].dat
                                for j in self._component_indices(i)))

            return Function(self.field_function_space,
                               val=dat)

        self._fields = tuple(field_function(i)
                             for i in range(self.nlocal_timesteps))

        # (co)functions containing the last step of the previous
        # and current slice for parallel communication
        self.uprev = Function(self.field_function_space)
        self.unext = Function(self.field_function_space)

        self.nlocal_dofs = self.function_space.node_set.size
        self.nglobal_dofs = self.ntimesteps*self.field_function_space.dim()

        with self._fbuf.dat.vec as fvec:
            sizes = (self.nlocal_dofs, self.nglobal_dofs)
            self._vec = PETSc.Vec().createWithArray(fvec.array,
                                                    size=sizes,
                                                    comm=ensemble.global_comm)
            self._vec.setFromOptions()

    def transform_index(self, i, from_range='slice', to_range='slice'):
        '''
        Shift timestep index from one range to another, and account for pythonic -ve indices.

        For example, if there are 3 ensemble ranks with time_partition=(2, 3, 2), then:
            window index 0 is slice index 0 on ensemble rank 0
            window index 1 is slice index 1 on ensemble rank 0
            window index 2 is slice index 0 on ensemble rank 1
            window index 3 is slice index 1 on ensemble rank 1
            window index 4 is slice index 2 on ensemble rank 1
            window index 5 is slice index 0 on ensemble rank 2
            window index 6 is slice index 1 on ensemble rank 2

        Raises IndexError if original or shifted index is out of bounds.

        :arg i: timestep index to shift.
        :arg from_range: range of i. Either slice or window.
        :arg to_range: range to shift i to. Either 'slice' or 'window'.
        '''
        idxtypes = {'slice': 'l', 'window': 'g'}

        if from_range not in idxtypes:
            raise ValueError("from_range must be "+" or ".join(idxtypes.keys()))
        if to_range not in idxtypes:
            raise ValueError("to_range must be "+" or ".join(idxtypes.keys()))

        i = self.layout.transform_index(i, itype=idxtypes[from_range], rtype=idxtypes[to_range])

        return i

    def _component_indices(self, step, from_range='slice', to_range='slice'):
        '''
        Return indices of the components of a timestep in the all-at-once MixedFunction.

        :arg step: timestep index to get component indices for.
        :arg from_range: range of step. Either slice or window.
        :arg to_range: range to shift the indices to. Either 'slice' or 'window'.
        '''
        step = self.transform_index(step, from_range=from_range, to_range=to_range)
        return tuple(self.ncomponents*step + c
                     for c in range(self.ncomponents))

    def __getitem__(self, i):
        '''
        Get a Function that is a view over a timestep.

        :arg i: index of timestep to view.
        :arg idx: is index in window or slice?
        '''
        index = i[0] if type(i) is tuple else i
        itype = i[1] if type(i) is tuple else 'slice'
        j = self.transform_index(index, from_range=itype, to_range='slice')
        return self._fields[j]

    def riesz_representation(self, riesz_map='L2', **kwargs):
        '''
        Return the Riesz representation with respect to the given Riesz map.

        :arg riesz_map: The Riesz map to use (l2, L2, or H1). This can also be a callable.
        :arg kwargs: other arguments to be passed to the firedrake.riesz_map.
        '''
        DualType = AllAtOnceCofunction if type(self) is AllAtOnceFunction else AllAtOnceFunction
        riesz = DualType(self.ensemble, self.time_partition, self.field_function_space.dual())
        riesz._fbuf.assign(self._fbuf.riesz_representation(riesz_map=riesz_map, **kwargs))
        return riesz

    def bcast_field(self, step, u):
        """
        Broadcast solution at given timestep `step` to Function `u` on all time-ranks.

        :arg step: window index of field to broadcast.
        :arg u: Function to place field into.
        """
        # find which rank step is on.
        root = self.layout.rank_of(step)

        # get u if step on this rank
        if self.time_rank == root:
            u.assign(self[step, 'window'])

        # bcast u
        self.ensemble.bcast(u, root=root)

        return u

    def update_time_halos(self, blocking=True):
        '''
        Update uprev with the last step from the previous time slice (periodic).

        :arg blocking: Whether to use blocking MPI communications.
            If False then a list of MPI requests is returned.
        '''
        # sending last timestep on current slice to next slice
        self.unext.assign(self[-1])

        size = self.ensemble.ensemble_comm.size
        rank = self.ensemble.ensemble_comm.rank

        # ring communication
        dst = (rank+1) % size
        src = (rank-1) % size

        if blocking:
            sendrecv = self.ensemble.sendrecv
        else:
            sendrecv = self.ensemble.isendrecv

        return sendrecv(fsend=self.unext, dest=dst, sendtag=rank,
                        frecv=self.uprev, source=src, recvtag=src)

    def copy(self, copy_values=True):
        """
        Return a deep copy of the AllAtOnceFunction.

        :arg copy_values: If true, the values of the current AllAtOnceFunction
            will be copied into the new AllAtOnceFunction.
        """
        new = type(self)(self.ensemble, self.time_partition,
                         self.field_function_space)
        if copy_values:
            new.assign(self)
        return new

    def assign(self, src, update_halos=True, blocking=True):
        """
        Set value of AllAtOnceFunction from another object.

        :arg src: object to set value from. Can be one of:
            - AllAtOnceFunction: assign all values from src.
            - PETSc Vec: assign self.function from src via self.global_vec.
            - firedrake.Function in self.function_space:
                assign timesteps from src.
            - firedrake.Function in self.field_function_space:
                assign initial condition and all timesteps from src.
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        def func_assign(x, y):
            return y.assign(x)

        def vec_assign(x, y):
            x.copy(y)

        if isinstance(src, type(self)):
            return self._vs_op(src, func_assign, vec_assign,
                               update_ics=True,
                               update_halos=update_halos,
                               blocking=blocking)

        # TODO: We should be able to use _vs_op here too but
        #       test_allatoncesolver:::test_solve_heat_equation
        #       fails if we do. The only difference is that
        #       _vs_op accesses the global vec with read/write
        #       access instead of write only.
        #       It isn't clear why this makes a difference (it
        #       shouldn't).
        elif isinstance(src, PETSc.Vec):
            with self.global_vec_wo() as gvec:
                src.copy(gvec)

        elif isinstance(src, type(self._fbuf)):
            return self._vs_op(src, func_assign, vec_assign,
                               update_ics=True,
                               update_halos=update_halos,
                               blocking=blocking)

        else:
            raise TypeError(f"src value must be AllAtOnceFunction or PETSc.Vec or field Function, not {type(src)}")

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    def zero(self, subset=None, zero_ics=True):
        """
        Set all values to zero.

        :arg subset: pyop2.types.set.Subset indicating the nodes to zero.
            If None then the whole function is zeroed.
        """
        funcs = [self[i] for i in range(self.nlocal_timesteps)]
        funcs.extend([self.uprev, self.unext])
        if hasattr(self, 'initial_condition') and zero_ics:
            funcs.append(self.initial_condition)
        for f in funcs:
            f.zero(subset=subset)
        return self

    def scale(self, a, update_ics=False,
              update_halos=False, blocking=True):
        """
        Scale the AllAtOnceFunction by a scalar.

        :arg a: scalar to multiply the function by.
        :arg update_ics: if True then the initial conditions will be scaled
            as well as the timestep values (if possible).
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        self._fbuf.assign(a*self._fbuf)

        if update_ics and hasattr(self, 'initial_condition'):
            self.initial_condition.assign(a*self.initial_condition)

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    def axpy(self, a, x, update_ics=False,
             update_halos=False, blocking=True):
        """
        Compute y = a*x + y where y is this AllAtOnceFunction.

        :arg a: scalar to multiply x.
        :arg x: other object for calculation. Can be one of:
            - AllAtOnceFunction: all timesteps are updated, and optionally the ics.
            - PETSc Vec: all timesteps are updated.
            - firedrake.Function in self.function_space:
                all timesteps are updated.
            - firedrake.Function in self.field_function_space:
                all timesteps are updated, and optionally the ics.
        :arg update_ics: if True then the initial conditions will be updated
            from x as well as the timestep values (if possible).
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        def func_axpy(x, y):
            return y.assign(a*x + y)

        def vec_axpy(x, y):
            y.axpy(a, x)

        return self._vs_op(x, func_axpy, vec_axpy,
                           update_ics=update_ics,
                           update_halos=update_halos,
                           blocking=blocking)

    def aypx(self, a, x, update_ics=False,
             update_halos=False, blocking=True):
        """
        Compute y = x + a*y where y is this AllAtOnceFunction.

        :arg a: scalar to multiply y.
        :arg x: other object for calculation. Can be one of:
            - AllAtOnceFunction: all timesteps are updated, and optionally the ics.
            - PETSc Vec: all timesteps are updated.
            - firedrake.Function in self.function_space:
                all timesteps are updated.
            - firedrake.Function in self.field_function_space:
                all timesteps are updated, and optionally the ics.
        :arg update_ics: if True then the initial conditions will be updated
            from x as well as the timestep values (if possible).
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        def func_aypx(x, y):
            return y.assign(x + a*y)

        def vec_aypx(x, y):
            y.aypx(a, x)

        return self._vs_op(x, func_aypx, vec_aypx,
                           update_ics=update_ics,
                           update_halos=update_halos,
                           blocking=blocking)

    def axpby(self, a, b, x, update_ics=False,
              update_halos=False, blocking=True):
        """
        Compute y = a*x + b*y where y is this AllAtOnceFunction.

        :arg a: scalar to multiply x.
        :arg b: scalar to multiply y.
        :arg x: other object for calculation. Can be one of:
            - AllAtOnceFunction: all timesteps are updated, and optionally the ics.
            - PETSc Vec: all timesteps are updated.
            - firedrake.Function in self.function_space:
                all timesteps are updated.
            - firedrake.Function in self.field_function_space:
                all timesteps are updated, and optionally the ics.
        :arg update_ics: if True then the initial conditions will be updated
            from x as well as the timestep values (if possible).
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        def func_axpby(x, y):
            return y.assign(a*x + b*y)

        def vec_axpby(x, y):
            y.axpby(a, b, x)

        return self._vs_op(x, func_axpby, vec_axpby,
                           update_ics=update_ics,
                           update_halos=update_halos,
                           blocking=blocking)

    def _vs_op(self, x, func_op, vec_op, update_ics=False,
               update_halos=False, blocking=True):
        """
        Vector space operations (axpy, xpby, axpby)

        :arg func_op: apply operation to a firedrake.Function.
        :arg vec_op: apply operation to a PETSc.Vec.
        :arg x: other object for calculation. Can be one of:
            - AllAtOnceFunction: all timesteps are updated, and optionally the ics.
            - PETSc Vec: all timesteps are updated.
            - firedrake.Function in self.function_space:
                all timesteps are updated.
            - firedrake.Function in self.field_function_space:
                all timesteps are updated, and optionally the ics.
        :arg update_ics: if True then the initial conditions will be updated
            from x as well as the timestep values (if possible).
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        if isinstance(x, type(self)):
            func_op(x._fbuf, self._fbuf)
            if update_ics and hasattr(self, 'initial_condition'):
                func_op(x.initial_condition, self.initial_condition)

        elif isinstance(x, PETSc.Vec):
            with self.global_vec() as gvec:
                vec_op(x, gvec)

        elif isinstance(x, type(self._fbuf)):
            if x.function_space() == self.field_function_space:
                for i in range(self.nlocal_timesteps):
                    func_op(x, self[i])
                if update_ics and hasattr(self, 'initial_condition'):
                    func_op(x, self.initial_condition)

            elif x.function_space() == self.function_space:
                func_op(x, self._fbuf)

            else:
                raise ValueError(f"x must be be in the `function_space` {self.function_space}"
                                 + f" or `field_function_space` {self.field_function_space} of the"
                                 + f" the AllAtOnceFunction, not in {x.function_space}")

        else:
            raise TypeError(f"x value must be AllAtOnce(Co)Function or PETSc.Vec or field (Co)Function, not {type(x)}")

        if update_halos:
            return self.update_time_halos(blocking=blocking)

    @contextlib.contextmanager
    def global_vec(self):
        """
        Context manager for the global PETSc Vec with read/write access.

        It is invalid to access the Vec outside of a context manager.
        """
        # fvec shares the same storage as _vec, so we need this context
        # manager to make sure that the data gets copied to/from the
        # Function.dat storage and _vec.
        with self._fbuf.dat.vec:
            self._vec.stateIncrease()
            yield self._vec

    @contextlib.contextmanager
    def global_vec_ro(self):
        """
        Context manager for the global PETSc Vec with read only access.

        It is invalid to access the Vec outside of a context manager.
        """
        # fvec shares the same storage as _vec, so we need this context
        # manager to make sure that the data gets copied into _vec from
        # the Function.dat storage.
        with self._fbuf.dat.vec_ro:
            self._vec.stateIncrease()
            yield self._vec

    @contextlib.contextmanager
    def global_vec_wo(self):
        """
        Context manager for the global PETSc Vec with write only access.

        It is invalid to access the Vec outside of a context manager.
        """
        # fvec shares the same storage as _vec, so we need this context
        # manager to make sure that the data gets copied back into the
        # Function.dat storage from _vec.
        with self._fbuf.dat.vec_wo:
            yield self._vec


class AllAtOnceFunction(AllAtOnceFunctionBase):
    def __init__(self, ensemble, time_partition, function_space):
        """
        A function representing multiple timesteps of a time-dependent finite-element problem,
        i.e. the solution to an all-at-once system.

        :arg ensemble: time-parallel ensemble communicator. The timesteps are partitioned
            over the ensemble members according to time_partition so
            ensemble.ensemble_comm.size == len(time_partition) must be True.
        :arg time_partition: a list of integers for the number of timesteps stored on each
            ensemble rank.
        :arg function_space: a FunctionSpace for the solution at a single timestep.
        """
        if not is_primal(function_space):
            raise TypeError("Cannot only make AllAtOnceFunction from a FunctionSpace")
        super().__init__(ensemble, time_partition, function_space)
        self.function = self._fbuf
        self.initial_condition = Function(self.field_function_space)


class AllAtOnceCofunction(AllAtOnceFunctionBase):
    def __init__(self, ensemble, time_partition, function_space):
        """
        A Cofunction representing multiple timesteps of a time-dependent finite-element problem,
        i.e. the solution to an all-at-once system.

        :arg ensemble: time-parallel ensemble communicator. The timesteps are partitioned
            over the ensemble members according to time_partition so
            ensemble.ensemble_comm.size == len(time_partition) must be True.
        :arg time_partition: a list of integers for the number of timesteps stored on each
            ensemble rank.
        :arg function_space: a FunctionSpace for the solution at a single timestep.
        """
        if not is_dual(function_space):
            raise TypeError("Can only make an AllAtOnceCofunction from a DualSpace")
        super().__init__(ensemble, time_partition, function_space)
        self.cofunction = self._fbuf

    def scale(self, a, update_halos=False, blocking=True):
        """
        Scale the AllAtOnceCofunction by a scalar.

        :arg a: scalar to multiply the function by.
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        return super().scale(a, update_halos=update_halos, blocking=blocking,
                             update_ics=False)

    def axpy(self, a, x, update_halos=False, blocking=True):
        """
        Compute y = a*x + y where y is this AllAtOnceCofunction.

        :arg a: scalar to multiply x.
        :arg x: other object for calculation. Can be one of:
            - AllAtOnceCofunction: all timesteps are updated, and optionally the ics.
            - PETSc Vec: all timesteps are updated.
            - firedrake.Cofunction in self.function_space:
                all timesteps are updated.
            - firedrake.Cofunction in self.field_function_space:
                all timesteps are updated, and optionally the ics.
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        return super().axpy(a, x, update_halos=update_halos, blocking=blocking,
                            update_ics=False)

    def aypx(self, a, x, update_halos=False, blocking=True):
        """
        Compute y = x + a*y where y is this AllAtOnceCofunction.

        :arg a: scalar to multiply y.
        :arg x: other object for calculation. Can be one of:
            - AllAtOnceCofunction: all timesteps are updated, and optionally the ics.
            - PETSc Vec: all timesteps are updated.
            - firedrake.Cofunction in self.function_space:
                all timesteps are updated.
            - firedrake.Cofunction in self.field_function_space:
                all timesteps are updated, and optionally the ics.
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        return super().aypx(a, x, update_halos=update_halos, blocking=blocking,
                            update_ics=False)

    def axpby(self, a, b, x, update_halos=False, blocking=True):
        """
        Compute y = a*x + b*y where y is this AllAtOnceCofunction.

        :arg a: scalar to multiply x.
        :arg b: scalar to multiply y.
        :arg x: other object for calculation. Can be one of:
            - AllAtOnceFunction: all timesteps are updated, and optionally the ics.
            - PETSc Vec: all timesteps are updated.
            - firedrake.Cofunction in self.function_space:
                all timesteps are updated.
            - firedrake.Cofunction in self.field_function_space:
                all timesteps are updated, and optionally the ics.
        :arg update_halos: if True then the time-halos will be updated.
        :arg blocking: if update_halos is True, then this argument determines
            whether blocking communication is used. A list of MPI Requests is returned
            if non-blocking communication is used.
        """
        return super().axpby(a, b, x, update_halos=update_halos, blocking=blocking,
                             update_ics=False)

class Paradiag(TimePartitionMixin):
    #()
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



class AllAtOnceSolver(TimePartitionMixin):
    
    def __init__(self, aaoform, aaofunc,
                 solver_parameters={},
                 appctx={},
                 options_prefix="",
                 jacobian_form=None,
                 jacobian_reference_state=None,
                 pre_function_callback=None,
                 post_function_callback=None,
                 pre_jacobian_callback=None,
                 post_jacobian_callback=None):
        """
        Solve an all-at-once form over an all-at-once function.

        This is used to solve for a timeseries defined by the all-at-once form
        and the initial condition in the all-at-once function.

        :arg aaoform: the AllAtOnceForm to solve.
        :arg aaofunc: the AllAtOnceFunction solution.
        :arg solver_parameters: solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        :arg appctx: A dictionary containing application context that is
            passed to the preconditioner if matrix-free.
        :arg options_prefix: an optional prefix used to distinguish PETSc options.
            Use this option if you want to pass options to the solver from the
            command line in addition to through the solver_parameters dict.
        :arg jacobian_form: an AllAtOnceForm to create the AllAtOnceJacobian from.
            Allows the Jacobian to be defined around a form different from the form
            used to assemble the residual.
        :arg jacobian_reference_state: a firedrake.Function to pass to the
            AllAtOnceJacobian as a reference state.
        :arg pre_function_callback: A user-defined function that will be called immediately
            before residual assembly. This can be used, for example, to update a coefficient
            function that has a complicated dependence on the unknown solution.
        :arg post_function_callback: As above, but called immediately after residual assembly.
        :arg pre_jacobian_callback: As above, but called immediately before Jacobian assembly.
        :arg post_jacobian_callback: As above, but called immediately after Jacobian assembly.
        """
        self._time_partition_setup(aaofunc.ensemble, aaofunc.time_partition)
        self.aaofunc = aaofunc
        self.aaoform = aaoform

        self.appctx = appctx

        self.jacobian_form = aaoform.copy() if jacobian_form is None else jacobian_form

        def passthrough(*args, **kwargs):
            pass

        # callbacks
        if pre_function_callback is None:
            self.pre_function_callback = passthrough
        else:
            self.pre_function_callback = pre_function_callback

        if post_function_callback is None:
            self.post_function_callback = passthrough
        else:
            self.post_function_callback = post_function_callback

        if pre_jacobian_callback is None:
            self.pre_jacobian_callback = passthrough
        else:
            self.pre_jacobian_callback = pre_jacobian_callback

        if post_jacobian_callback is None:
            self.post_jacobian_callback = passthrough
        else:
            self.post_jacobian_callback = post_jacobian_callback

        # solver options
        self.solver_parameters = solver_parameters
        self.flat_solver_parameters = flatten_parameters(solver_parameters)
        self.options = OptionsManager(self.flat_solver_parameters, options_prefix)
        options_prefix = self.options.options_prefix

        # snes
        self.snes = PETSc.SNES().create(comm=self.ensemble.global_comm)

        self.snes.setOptionsPrefix(options_prefix)

        def assemble_function(snes, X, F):
            self.pre_function_callback(self, X)
            self.aaoform.assemble(X)
            with aaoform.F.global_vec_ro() as fvec:
                fvec.copy(F)
            self.post_function_callback(self, X, F)

        self._F = aaoform.F._vec.duplicate()
        self.snes.setFunction(assemble_function, self._F)

        # Jacobian
        with self.options.inserted_options():
            self.jacobian = AllAtOnceJacobian(self.jacobian_form,
                                              reference_state=jacobian_reference_state,
                                              options_prefix=options_prefix,
                                              appctx=appctx)

        self.jacobian_mat = self.jacobian.petsc_mat()

        def form_jacobian(snes, X, J, P):
            self.pre_jacobian_callback(self, X, J)
            self.jacobian.update(X)
            self.post_jacobian_callback(self, X, J)
            J.assemble()
            P.assemble()

        self.snes.setJacobian(form_jacobian,
                              J=self.jacobian_mat,
                              P=self.jacobian_mat)

        # complete the snes setup
        self.options.set_from_options(self.snes)

    
    def solve(self, rhs=None):
        """
        Solve the all-at-once system.

        :arg rhs: optional constant part of the system.
        """
        with self.aaofunc.global_vec() as gvec, self.options.inserted_options():
            if rhs is None:
                self.snes.solve(None, gvec)
            else:
                if not isinstance(rhs, AllAtOnceCofunction):
                    msg = f"Right hand side of all-at-once problem must be AllAtOnceCofunction not {type(rhs)}."
                    raise TypeError(msg)
                with rhs.global_vec_ro() as rvec:
                    self.snes.solve(rvec, gvec)


class LinearSolver(TimePartitionMixin):
    
    def __init__(self, aaoform,
                 solver_parameters={},
                 appctx={},
                 options_prefix=""):
        """
        Solve a linear system where the matrix is an all-at-once Jacobian.

        This does not solve for a timeseries (use an AllAtOnceSolver if this
        is what you need), but simply uses the AllAtOnceJacobian as the Mat
        for a KSP.

        :arg aaoform: the AllAtOnceForm to form the Jacobian from.
        :arg solver_parameters: solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        :arg appctx: A dictionary containing application context that is
            passed to the preconditioner if matrix-free.
        :arg options_prefix: an optional prefix used to distinguish PETSc options.
            Use this option if you want to pass options to the solver from the
            command line in addition to through the solver_parameters dict.
        """
        self._time_partition_setup(aaoform.ensemble, aaoform.time_partition)

        self.aaoform = aaoform
        self.appctx = appctx

        # manage options from both dict and command line
        self.solver_parameters = solver_parameters
        self.flat_solver_parameters = flatten_parameters(solver_parameters)
        self.options = OptionsManager(self.flat_solver_parameters, options_prefix)
        options_prefix = self.options.options_prefix

        # the solver
        self.ksp = PETSc.KSP().create(comm=self.ensemble.global_comm)
        self.ksp.setOptionsPrefix(options_prefix)

        # create the all-at-once jacobian
        with self.options.inserted_options():
            self.jacobian = AllAtOnceJacobian(aaoform, appctx=appctx,
                                              options_prefix=options_prefix)

        # create petsc matrix
        self.jacobian_mat = self.jacobian.petsc_mat()

        # finish setting up the ksp
        self.ksp.setOperators(self.jacobian_mat)
        self.options.set_from_options(self.ksp)

    
    def solve(self, b, x):
        """
        Solve the all-at-once matrix Ax=b.

        :arg b: AllAtOnceCofunction right hand side vector.
        :arg x: AllAtOnceFunction solution vector.
        """
        if not isinstance(x, AllAtOnceFunction):
            msg = f"Solution of all-at-once problem must be AllAtOnceFunction not {type(x)}."
            raise TypeError(msg)

        if not isinstance(b, AllAtOnceCofunction):
            msg = f"Right hand side of all-at-once problem must be AllAtOnceCofunction not {type(b)}."
            raise TypeError(msg)

        with x.global_vec() as xvec, b.global_vec_ro() as bvec:
            with self.options.inserted_options():
                self.ksp.solve(bvec, xvec)

paradiag = Paradiag(
    ensemble=ensemble, 
    form_mass=form_mass, 
    form_function=form_function, 
    ics=uinitial, dt=0.1, theta=0.5,
    time_partition=time_partition,
    solver_parameters=solver_parameters)

paradiag.solve(nwindows=1)



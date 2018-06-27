import copy
import datetime
import uuid
import weakref
from collections import abc
from numbers import Integral
from typing import MutableSequence, Any, Optional, Sequence, MutableMapping
import typing

import numpy as np
from scipy.stats import multivariate_normal

from ..base import Property
from .array import StateVector, CovarianceMatrix
from .base import Type
from .particle import Particle


class State(Type):
    """State type.

    Most simple state type, which only has time and a state vector."""
    timestamp: datetime.datetime = Property(
        default=None, doc="Timestamp of the state. Default None.")
    state_vector: StateVector = Property(doc='State vector.')

    def __init__(self, state_vector, *args, **kwargs):
        state_vector = state_vector.view(StateVector)
        super().__init__(state_vector, *args, **kwargs)

    @property
    def ndim(self):
        """The number of dimensions represented by the state."""
        return self.state_vector.shape[0]

    @staticmethod
    def from_state(state: 'State', *args: Any, target_type: Optional[typing.Type] = None,
                   **kwargs: Any) -> 'State':
        """Class utility function to create a new state (or compatible type) from an existing
        state. The type and properties of this new state are defined by `state` except for any
        explicitly overwritten via `args` and `kwargs`.

        It acts similarly in feel to a copy constructor, with the optional over-writing of
        properties.

        Parameters
        ----------
        state: State
            :class:`~.State` to use existing properties from, and identify new state-type from.
        \\*args: Sequence
            Arguments to pass to newly created state, replacing those with same name in `state`.
        target_type: Type,  optional
            Optional argument specifying the type of of object to be created. This need not
            necessarily be :class:`~.State` subclass. Any arguments that match between the input
            `state` and the target type will be copied from the old to the new object (except those
            explicitly specified in `args` and `kwargs`.
        \\*\\*kwargs: Mapping
            New property names and associate value for use in newly created state, replacing those
            on the `state` parameter.
        """
        # Handle being initialised with state sequence
        if isinstance(state, StateMutableSequence):
            state = state.state

        if target_type is None:
            target_type = type(state)

        args_property_names = {
            name for n, name in enumerate(target_type.properties) if n < len(args)}

        new_kwargs = {
            name: getattr(state, name)
            for name in type(state).properties.keys() & target_type.properties.keys()
            if name not in args_property_names and name not in kwargs}

        new_kwargs.update(kwargs)

        return target_type(*args, **new_kwargs)


class CreatableFromState:
    class_mapping = {}

    def __init_subclass__(cls, **kwargs):
        bases = cls.__bases__
        if CreatableFromState in bases:
            # Direct subclasses should not be added to the class mapping, only subclasses of
            # subclasses
            return
        if len(bases) != 2:
            raise TypeError('A CreatableFromState subclass must have exactly two superclasses')
        base_class, state_type = cls.__bases__
        if not issubclass(base_class, CreatableFromState):
            raise TypeError('The first superclass of a CreatableFromState subclass must be a '
                            'CreatableFromState (or a subclass)')
        if not issubclass(state_type, State):
            # Non-state subclasses do not need adding to the class mapping, as they should not
            # be created from States
            return
        if base_class not in CreatableFromState.class_mapping:
            CreatableFromState.class_mapping[base_class] = {}
        CreatableFromState.class_mapping[base_class][state_type] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_state(
            cls,
            state: State,
            *args: Any,
            target_type: Optional[type] = None,
            **kwargs: Any) -> 'State':
        """
        Return new object instance of suitable type from an existing `state`.
        The type returned can be explicitly specified using the `target_type` argument, otherwise
        it is chosen by introspection of the created subclasses of this type: see below for an
        example. Any compatible properties are copied from the input `state` to the returned
        object, except for those specified by `args` and `kwargs`, which take precedence over those
        from the input `state`.

        This method is primarily concerned with type selection, with actual copying performed by
        the static :meth:`~.State.from_state` method. As an example of the type selection
        algorithm, consider the case of the class
        `GaussianStatePrediction(Prediction, GaussianState)`. This is subclass of `Prediction`,
        and `GaussianState` and so the `class_mapping` property will have an entry added (when
        `GaussianStatePrediction` is defined) such that
        `class_mapping[Prediction][GaussianState] = GaussianStatePrediction`. If this method is
        then called like below

        >>>> gaussian_state = GaussianState(some_arguments)
        >>>> new_prediction = Prediction.from_state(gaussian_state, *args, **kwargs)

        then the `from_state` method will look up the class mapping and see that
        `Prediction.from_state()` called with a `GaussianState` input should return a
        `GaussianStatePrediction` object, and therefore the type of `new_prediction` will be
        `GaussianStatePrediction`

        The functionality is currently used by :class:`~.Prediction` and :class:`~.Updater`
        objects.

        Parameters
        ----------
        state: State
            :class:`~.State` to use existing properties from, and identify prediction type from
        \\*args: Sequence
            Arguments to pass to newly created prediction, replacing those with same name on
            ``state`` parameter.
        target_type: Type, optional
            Type to use for prediction, overriding one from :attr:`class_mapping`.
        \\*\\*kwargs: Mapping
            New property names and associate value for use in newly created prediction, replacing
            those on the ``state`` parameter.
        """
        # Handle being initialised with state sequence
        if isinstance(state, StateMutableSequence):
            state = state.state
        try:
            state_type = next(type_ for type_ in type(state).mro()
                              if type_ in CreatableFromState.class_mapping[cls])
        except StopIteration:
            raise TypeError(f'{cls.__name__} type not defined for {type(state).__name__}')
        if target_type is None:
            target_type = CreatableFromState.class_mapping[cls][state_type]

        return target_type.from_state(state, *args, **kwargs, target_type=target_type)


class ASDState(Type):
    """ASD State type

    For the use of Accumulated State Densities.
    """

    multi_state_vector: StateVector = Property(
        doc="State vector of all timestamps")
    timestamps: Sequence[datetime.datetime] = Property(
        doc="List of all timestamps which have a state in the ASDState")
    max_nstep: int = Property(
        doc="Decides when the state is pruned in a prediction step. If 0 then there is no pruning")

    def __init__(self, multi_state_vector, timestamps,
                 max_nstep=0, *args, **kwargs):
        if multi_state_vector is not None and timestamps is not None:
            multi_state_vector = StateVector(multi_state_vector)
            if not isinstance(timestamps, Sequence):
                timestamps = list([timestamps])
            self.max_nstep = max_nstep
        super().__init__(multi_state_vector, timestamps, max_nstep, *args, **kwargs)

    def __getitem__(self, item):
        if isinstance(item, Integral):
            ndim = self.ndim
            start = item * ndim
            end = None if item == -1 else (item+1) * ndim
            state_slice = slice(start, end)
            state_vector = StateVector(self.multi_state_vector[state_slice])
            timestamp = self.timestamps[item]
            return State(state_vector=state_vector, timestamp=timestamp)
        else:
            raise TypeError(f'{type(self).__name__!r} only subscriptable by int')

    @property
    def state_vector(self):
        """The State vector of the newest timestamp"""
        return self.multi_state_vector[0:self.ndim]

    @property
    def timestamp(self):
        """The newest timestamp"""
        return self.timestamps[0]

    @property
    def ndim(self):
        """Dimension of one State"""
        return int(self.multi_state_vector.shape[0] / len(self.timestamps))

    @property
    def nstep(self):
        """Number of timesteps which are in the ASDState"""
        return len(self.timestamps)

    @clearable_cached_property('multi_state_vector', 'timestamps')
    def state(self):
        """A :class:`~.State` object representing latest timestamp"""
        return self[0]

    @clearable_cached_property('multi_state_vector', 'timestamps')
    def states(self):
        return [self[i] for i in range(self.nstep)]


State.register(ASDState)


class StateMutableSequence(Type, abc.MutableSequence):
    """A mutable sequence for :class:`~.State` instances

    This sequence acts like a regular list object for States, as well as
    proxying state attributes to the last state in the sequence. This sequence
    can also be indexed/sliced by :class:`datetime.datetime` instances.

    Notes
    -----
    If shallow copying, similar to a list, it is safe to add/remove states
    without affecting the original sequence.

    Example
    -------
    >>> t0 = datetime.datetime(2018, 1, 1, 14, 00)
    >>> t1 = t0 + datetime.timedelta(minutes=1)
    >>> state0 = State([[0]], t0)
    >>> sequence = StateMutableSequence([state0])
    >>> print(sequence.state_vector, sequence.timestamp)
    [[0]] 2018-01-01 14:00:00
    >>> sequence.append(State([[1]], t1))
    >>> for state in sequence[t1:]:
    ...     print(state.state_vector, state.timestamp)
    [[1]] 2018-01-01 14:01:00
    """

    states: MutableSequence[State] = Property(
        default=None,
        doc="The initial list of states. Default `None` which initialises with empty list.")

    def __init__(self, states=None, *args, **kwargs):
        if states is None:
            states = []
        elif not isinstance(states, abc.Sequence):
            # Ensure states is a list
            states = [states]
        super().__init__(states, *args, **kwargs)

    def __len__(self):
        return self.states.__len__()

    def __setitem__(self, index, value):
        return self.states.__setitem__(index, value)

    def __delitem__(self, index):
        return self.states.__delitem__(index)

    def __getitem__(self, index):
        if isinstance(index, slice) and (
                isinstance(index.start, datetime.datetime)
                or isinstance(index.stop, datetime.datetime)):
            items = []
            for state in self.states:
                try:
                    if index.start and state.timestamp < index.start:
                        continue
                    if index.stop and state.timestamp >= index.stop:
                        continue
                except TypeError as exc:
                    raise TypeError(
                        'both indices must be `datetime.datetime` objects for'
                        'time slice') from exc
                items.append(state)
            return StateMutableSequence(items[::index.step])
        elif isinstance(index, datetime.datetime):
            for state in reversed(self.states):
                if state.timestamp == index:
                    return state
            else:
                raise IndexError('timestamp not found in states')
        elif isinstance(index, slice):
            return StateMutableSequence(self.states.__getitem__(index))
        else:
            return self.states.__getitem__(index)

    def __getattribute__(self, name):
        # This method is called if we try to access an attribute of self. First we try to get the
        # attribute directly, but if that fails, we want to try getting the same attribute from
        # self.state instead. If that, in turn,  fails we want to return the error message that
        # would have originally been raised, rather than an error message that the State has no
        # such attribute.
        #
        # An alternative mechanism using __getattr__ seems simpler (as it skips the first few lines
        # of code, but __getattr__ has no mechanism to capture the originally raised error.
        try:
            # This tries first to get the attribute from self.
            return Type.__getattribute__(self, name)
        except AttributeError as original_error:
            if name.startswith("_"):
                # Don't proxy special/private attributes to `state`, just raise the original error
                raise original_error
            else:
                # For non _ attributes, try to get the attribute from self.state instead of self.
                try:
                    my_state = Type.__getattribute__(self, 'state')
                    return getattr(my_state, name)
                except AttributeError:
                    # If we get the error about 'State' not having the attribute, then we want to
                    # raise the original error instead
                    raise original_error

    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        property_name = self.__class__.states._property_name
        inst.__dict__[property_name] = copy.copy(self.__dict__[property_name])
        return inst

    def insert(self, index, value):
        return self.states.insert(index, value)

    @property
    def state(self):
        return self.states[-1]

    def last_timestamp_generator(self):
        """Generator yielding the last state for each timestamp

        This provides a method of iterating over a sequence of states,
        such that when multiple states for the same timestamp exist,
        only the last state is yielded. This is particularly useful in
        cases where you may have multiple :class:`~.Update` states for
        a single timestamp e.g. multi-sensor tracking example.

        Yields
        ------
        State
            A state for each timestamp present in the sequence.
        """
        state_iter = iter(self)
        current_state = next(state_iter)
        for next_state in state_iter:
            if next_state.timestamp > current_state.timestamp:
                yield current_state
            current_state = next_state
        yield current_state


class GaussianState(State):
    """Gaussian State type

    This is a simple Gaussian state object, which, as the name suggests,
    is described by a Gaussian state distribution.
    """
    covar: CovarianceMatrix = Property(doc='Covariance matrix of state.')

    def __init__(self, state_vector, covar, *args, **kwargs):
        covar = covar.view(CovarianceMatrix)
        super().__init__(state_vector, covar, *args, **kwargs)
        if self.state_vector.shape[0] != self.covar.shape[0]:
            raise ValueError(
                "state vector and covariance should have same dimensions")

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return self.state_vector


class ParticleState(Type):
    """Particle State type

    This is a particle state object which describes the state as a
    distribution of particles"""

    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the state. Default None.")
    particles = Property([Particle],
                         doc='List of particles representing state')

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return np.average([p.state_vector for p in self.particles], axis=0,
                          weights=[p.weight for p in self.particles])

    @property
    def state_vector(self):
        """The mean value of the particle states"""
        return self.mean

    @property
    def covar(self):
        cov = np.cov(np.hstack([p.state_vector for p in self.particles]),
                     ddof=0, aweights=[p.weight for p in self.particles])
        # Fix one dimensional covariances being returned with zero dimension
        if not cov.shape:
            cov = cov.reshape(1, 1)
        return cov

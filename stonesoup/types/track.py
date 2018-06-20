import copy
import uuid
from typing import MutableSequence, MutableMapping

from .multihypothesis import MultipleHypothesis
from .state import State, StateMutableSequence
from .update import Update
from ..base import Property


class Track(StateMutableSequence):
    """Track type

    Parameters
    ----------

    Attributes
    ----------
    state : State
        Most recent state
    state_vector : StateVector
        Most recent state vector
    covar : CovarianceMatrix
        Most recent state covariance
    timestamp : :class:`datetime.datetime`
        Most recent state timestamp
    """

    states = Property(
        [State],
        default=None,
        doc="The initial states of the track. Default `None` which initialises"
            "with empty list.")

    def __init__(self, states=None, *args, **kwargs):
        if states is None:
            states = []
        super().__init__(states, *args, **kwargs)

    def _update_metadatas(self, index):
        """Update track :attr:`metadatas` property, starting at specified index.

        Parameters
        ----------
        index: Int
            Index of :attr:`metadatas` to update from.
        """
        # Plus one for 0th initial track meta data
        self.metadatas = self.metadatas[:index]

        for future_state in self.states[index:]:
            self._update_metadata_from_state(future_state)

    def _update_metadata_from_state(self, state):
        """Update :attr:`metadatas` with an updated metadata entry, accounting for extracted
        metadata from state.

        Parameters
        ----------
        state: State
            A state object from which to extract metadata. Metadata can only be extracted from
            Update (or subclassed) objects. Calling this method with a non-Update (subclass) object
            will NOT raise an error, but will have no effect on the metadata.
        """
        self.metadatas.append(self.metadata.copy())

        if isinstance(state, Update):
            if isinstance(state.hypothesis, MultipleHypothesis):
                # Sort and iterate through multiple hypotheses such that most
                # likely hypothesis comes last. This ensures that metadata
                # from all hypotheses are retained, but more likely
                # hypotheses will over-write the metadata set by less likely
                # ones.
                try:
                    for hypothesis in sorted(state.hypothesis, reverse=True):
                        if hypothesis \
                                and hypothesis.measurement.metadata is not None:
                            self.metadata.update(hypothesis.measurement.metadata)
                except TypeError:
                    pass
            else:
                hypothesis = state.hypothesis
                if hypothesis and hypothesis.measurement.metadata is not None:
                    self.metadata.update(hypothesis.measurement.metadata)

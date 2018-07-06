import uuid
from typing import MutableSequence, MutableMapping, Sequence

from ..base import Property
from .base import Type
from .state import State


class GroundTruthState(State):
    """Ground Truth State type"""
    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class CategoricalGroundTruthState(GroundTruthState, CategoricalState):
    """Categorical Ground Truth State type"""


class GroundTruthPath(StateMutableSequence):
    """Ground Truth Path type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states: MutableSequence[GroundTruthState] = Property(
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")
    id: str = Property(
        default=None,
        doc="The unique path ID. Default `None` where random UUID is "
            "generated.")

    states = Property(
        [GroundTruthState],
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")

    def __init__(self, states=None, *args, **kwargs):
        if states is None:
            states = []
        super().__init__(states, *args, **kwargs)

    def __len__(self):
        return len(self.states)

    def __setitem__(self, index, value):
        return self.states.__setitem__(index, value)

    def insert(self, index, value):
        return self.states.insert(index, value)

    def __delitem__(self, index):
        return self.states.__delitem__(index)

    def __getitem__(self, index):
        return self.states.__getitem__(index)

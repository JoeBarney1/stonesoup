# -*- coding: utf-8 -*-
from ..base import Property
from .state import State, StateMutableSequence


class GroundTruthState(State):
    """Ground Truth State type"""
    metadata = Property(dict, default=None,
                        doc='Dictionary of metadata items for Detections.')

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class GroundTruthPath(StateMutableSequence):
    """Ground Truth Path type

    A :class:`~.StateMutableSequence` representing a track.
    """

    states = Property(
        [GroundTruthState],
        default=None,
        doc="List of groundtruth states to initialise path with. Default "
            "`None` which initialises with an empty list.")

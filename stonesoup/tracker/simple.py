import datetime
from typing import Set, Tuple

import numpy as np

from .base import Tracker, _TrackerMixInNext
from ..base import Property
from ..dataassociator import DataAssociator
from ..deleter import Deleter
from ..functions import gm_reduce_single
from ..initiator import Initiator
from ..reader import DetectionReader
from ..types.array import StateVectors
from ..types.prediction import GaussianStatePrediction
from ..types.track import Track
from ..types.update import GaussianStateUpdate
from ..updater import Updater


class SingleTargetTracker(_TrackerMixInNext, Tracker):
    """A simple single target tracker.

    Track a single object using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active track, and then
    either updating the track state with the result of the :attr:`updater` if
    a detection is associated, or with the prediction if no detection is
    associated to the track. The track is then checked for deletion by the
    :attr:`deleter`, and if deleted the :attr:`initiator` is called to generate
    a new track. Similarly if no track is present (i.e. tracker is initialised
    or deleted in previous iteration), only the :attr:`initiator` is called.

    Parameters
    ----------

    Attributes
    ----------
    track : :class:`~.Track`
        Current track being maintained. Also accessible as the sole item in
        :attr:`tracks`
    """
    initiator: Initiator = Property(doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(doc="Deleter used to delete tracks.")
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections")
    updater: Updater = Property(doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track = None

    @property
    def tracks(self):
        return {self.track}

    def tracks_gen(self):
        self.track = None

    def __next__(self) -> Tuple[datetime.datetime, Set[Track]]:
        time, detections = next(self.detector_iter)
        if self._track is not None:
            associations = self.data_associator.associate(
                self.tracks, detections, time)
            if associations[self._track]:
                state_post = self.updater.update(associations[self._track])
                self._track.append(state_post)
            else:
                self._track.append(
                    associations[self._track].prediction)

            if self.track is not None:
                associations = self.data_associator.associate(
                        self.tracks, detections, time)
                if associations[self.track].detection is not None:
                    state_post = self.updater.update(
                        associations[self.track].prediction,
                        associations[self.track].detection,
                        associations[self.track].innovation)
                    self.track.states.append(state_post)
                else:
                    self.track.states.append(
                        associations[self.track].prediction)

            if self.track is None or self.deletor.delete_tracks(self.tracks):
                new_tracks = self.initiator.initiate(detections)
                if new_tracks:
                    track = next(iter(new_tracks))
                    self.track = track
                else:
                    self.track = None

            yield time, self.tracks

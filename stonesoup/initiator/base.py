import datetime
from abc import abstractmethod
from typing import Set

from ..base import Base
from ..types.detection import Detection
from ..types.track import Track


class Initiator(Base):
    """Initiator base class

    Creates zero or more tracks based on provided detections.
    """

    @abstractmethod
    def initiate(self, detections: Set[Detection], timestamp: datetime.datetime,
                 **kwargs) -> Set[Track]:
        """Generate tracks from detections.

        Parameters
        ----------
        detections : set of :class:`~.Detection`
            Detections used to generate set of tracks
        timestamp: datetime.datetime
            Current timestamp

        Returns
        -------
        : set of :class:`~.Track`
            Tracks generated from detections
        """
        raise NotImplementedError

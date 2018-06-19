import datetime
from abc import abstractmethod
from typing import Iterator, Set, Tuple

from ..base import Base
from ..types.detection import Detection
from ..types.track import Track


class Tracker(Base):
    """Tracker base class"""

    @property
    @abstractmethod
    def tracks(self):
        """The tracks at the current time step.

        This is the set of tracks last returned by the
        :meth:`tracks_gen` generator, to allow other components, like
        metrics, to access the data.
        """
        raise NotImplementedError

    @abstractmethod
    def tracks(self) -> Set[Track]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[datetime.datetime, Set[Track]]]:
        return self

    @abstractmethod
    def __next__(self) -> Tuple[datetime.datetime, Set[Track]]:
        """
        Returns
        -------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Track`
            Tracks existing in the time step
        """
        raise NotImplementedError

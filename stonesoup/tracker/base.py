import datetime
from abc import abstractmethod
from typing import Iterator, Set, Tuple

from ..base import Base
from stonesoup.buffered_generator import BufferedGenerator


class Tracker(Base, BufferedGenerator):
    """Tracker base class"""

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

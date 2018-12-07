import datetime
from abc import abstractmethod
from typing import Set, Mapping, Tuple

from ..base import Base, Property
from ..types import SingleMeasurementJointHypothesis
from ..hypothesiser import Hypothesiser
from ..types.detection import Detection
from ..types.hypothesis import Hypothesis
from ..types.track import Track
from ..types.association import AssociationSet


class DataAssociator(Base):
    """Data Associator base class

    A data associator is used to associate tracks and detections, and may also
    include an association of a missed detection. The associations generate are
    in the form a mapping each track to a hypothesis, based on "best" choice
    from hypotheses generate from a :class:`~.Hypothesiser`.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Generate a set of hypotheses for each track-detection pair")

    def generate_hypotheses(self, tracks, detections, timestamp, **kwargs):
        return {track: self.hypothesiser.hypothesise(
                    track, detections, timestamp, **kwargs)
                for track in tracks}

    @abstractmethod
    def associate(self, tracks: Set[Track], detections: Set[Detection],
                  timestamp: datetime.datetime, **kwargs) -> Mapping[Track, Hypothesis]:
        """Associate tracks and detections

        Parameters
        ----------
        tracks : set of :class:`~.Track`
            Tracks which detections will be associated to.
        detections : set of :class:`~.Detection`
            Detections to be associated to tracks.
        timestamp : datetime.datetime
            Timestamp to be used for missed detections and to predict to.

        Returns
        -------
        : mapping of :class:`~.Track` : :class:`~.Hypothesis`
            Mapping of track to Hypothesis
        """
        raise NotImplementedError


class Associator(Base):
    """Associator base class

    An associator is used to associate objects for the generation of
    metrics. It returns an :class:`~.AssociationSet` containing
    a set of :class:`~.Association`
    objects.
    """


class TrackToTrackAssociator(Associator):
    """Associates *n* sets of :class:`~.Track` objects together"""

    @abstractmethod
    def associate_tracks(self, *tracks_sets: Set[Track]) \
            -> AssociationSet:
        """Associate *n* sets of tracks together.

        Parameters
        ----------
        joint_hypothesis : :class:`SingleMeasurementJointHypothesis`
            A set of hypotheses linking each prediction to a single detection

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """

        number_hypotheses = len(joint_hypothesis)
        unique_hypotheses = len(
            {hyp.measurement for hyp in joint_hypothesis} - {None})
        number_null_hypotheses = sum(
            hyp.measurement is None for hyp in joint_hypothesis)

        # joint_hypothesis is invalid if one detection is assigned to more than
        # one prediction. Multiple missed detections are valid.
        if unique_hypotheses + number_null_hypotheses == number_hypotheses:
            return True
        else:
            return False

    @classmethod
    def enumerate_joint_hypotheses(cls, hypotheses):
        """Enumerate the possible joint hypotheses.

        Create a list of all possible joint hypotheses from the individual
        hypotheses and determine whether each is valid.

        Parameters
        ----------
        tracks_sets : *n* sets of :class:`~.Track` objects
            Tracks to associate to other track sets

        Returns
        -------
        joint_hypotheses : list of :class:`SingleMeasurementJointHypothesis`
            A list of all valid joint hypotheses with a score on each
        """

        # Create a list of dictionaries of valid track-hypothesis pairs
        joint_hypotheses = [
            SingleMeasurementJointHypothesis({
                track: hypothesis
                for track, hypothesis in zip(hypotheses, joint_hypothesis)})
            for joint_hypothesis in itertools.product(*hypotheses.values())
            if cls.isvalid(joint_hypothesis)]

        unassociated_tracks = tuple(tracks_set - associated_tracks for tracks_set in tracks_sets)
        return associations, unassociated_tracks


class TwoTrackToTrackAssociator(TrackToTrackAssociator):
    """Associates two sets of :class:`~.Track` objects together"""

    @abstractmethod
    def associate_tracks(self, tracks_set_1: Set[Track], tracks_set_2: Set[Track]) \
            -> AssociationSet:
        """Associate two sets of tracks together.

        Parameters
        ----------
        tracks_set_1 : set of :class:`~.Track` objects
            Tracks to associate to track set 2
        tracks_set_2 : set of :class:`~.Track` objects
            Tracks to associate to track set 1

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """

# -*- coding: utf-8 -*-
import numpy as np

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..assignment import assignAlgs
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleDistanceHypothesis

class NearestNeighbour(DataAssociator):
    """Nearest Neighbour Associator

    Scores and associates detections to a predicted state using the Nearest
    Neighbour method.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, timestamp, **kwargs):

        # Generate a set of hypotheses for each track on each detection
        hypotheses = self.generate_hypotheses(tracks, detections, timestamp, **kwargs)

        # Only associate tracks with one or more hypotheses
        associate_tracks = {track
                            for track, track_hypotheses in hypotheses.items()
                            if track_hypotheses}

        # Only associate tracks with one or more hypotheses
        associate_tracks = {track
                            for track, track_hypotheses in hypotheses.items()
                            if track_hypotheses}

        associations = {}
        associated_measurements = set()
        while associate_tracks > associations.keys():
            # Define a 'greedy' association
            best_hypothesis = None
            for track in associate_tracks - associations.keys():
                for hypothesis in hypotheses[track]:
                    # A measurement may only be associated with a single track
                    if hypothesis.measurement in associated_measurements:
                        continue
                    # best_hypothesis is 'greater than' other
                    if (best_hypothesis is None
                            or hypothesis > best_hypothesis):
                        best_hypothesis = hypothesis
                        best_hypothesis_track = track

            associations[best_hypothesis_track] = best_hypothesis
            if best_hypothesis:
                associated_measurements.add(best_hypothesis.measurement)

        return associations


class GlobalNearestNeighbour(DataAssociator):
    """Global Nearest Neighbour Associator

    Scores and associates detections to a predicted state using the Global
    Nearest Neighbour method, assuming a distance-based hypothesis score.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, timestamp, **kwargs):

        # Generate a set of hypotheses for each track on each detection
        hypotheses = self.generate_hypotheses(tracks, detections, timestamp, **kwargs)

        # Link hypotheses into a set of joint_hypotheses and evaluate
        joint_hypotheses = self.enumerate_joint_hypotheses(hypotheses)
        associations = max(joint_hypotheses)

        return associations

    @staticmethod
    def isvalid(joint_hypothesis):
        """Determine whether a joint_hypothesis is valid.

        Check the set of hypotheses that define a joint hypothesis to ensure a
        single detection is not associated to more than one track.

        Parameters
        ----------
        joint_hypothesis : :class:`JointHypothesis`
            A set of hypotheses linking each prediction to a single detection

        Returns
        -------
        bool
            Whether joint_hypothesis is a valid set of hypotheses
        """

        number_hypotheses = len(joint_hypothesis)
        unique_hypotheses = len(
            {hyp.measurement for hyp in joint_hypothesis if hyp})
        number_null_hypotheses = sum(not hyp for hyp in joint_hypothesis)

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
        hypotheses : dict of :class:`~.Track`: :class:`~.Hypothesis`
            A list of all hypotheses linking predictions to detections,
            including missed detections

        Returns
        -------
        joint_hypotheses : list of :class:`JointHypothesis`
            A list of all valid joint hypotheses with a score on each
        """

        # Create a list of dictionaries of valid track-hypothesis pairs
        joint_hypotheses = [
            JointHypothesis({
                track: hypothesis
                for track, hypothesis in zip(hypotheses, joint_hypothesis)})
            for joint_hypothesis in itertools.product(*hypotheses.values())
            if cls.isvalid(joint_hypothesis)]

        return joint_hypotheses


class GNNWith2DAssignment(DataAssociator):
    """Global Nearest Neighbour Associator

    Associates detections to a predicted state using the
    Global Nearest Neighbour method, utilising a 2D matrix of
    distances and a "shortest path" assignment algorithm.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, timestamp, **kwargs):
        """Associate a set of detections with predicted states.

        Parameters
        ----------
        tracks : set of :class:`Track`
            Current tracked objects
        detections : set of :class:`Detection`
            Retrieved measurements
        timestamp : datetime.datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Generate a set of hypotheses for each track on each detection
        hypotheses = self.generate_hypotheses(tracks, detections, timestamp, **kwargs)

        # Link hypotheses into a set of joint_hypotheses and evaluate
        joint_hypotheses = self.enumerate_joint_hypotheses(hypotheses)
        associations = max(joint_hypotheses)
        
        return associations

class GNNwith2DAssignment(DataAssociator):
    """Global Nearest Neighbour Associator

    Associates detections to a predicted state using the Global Nearest Neighbour method, 
    utilising a 2D matrix of distances and a "shortest path" assignment algorithm.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate a set of detections with predicted states.

        Parameters
        ----------
        tracks : set of :class:`Track`
            Current tracked objects
        detections : set of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """
        
        # Convert sets to indexable lists
        tracks = list(tracks)
        detections = list(detections)
        
        # Generate 2d array "matrix" of hypotheses mapping track to detection
        hypothesis_matrix = np.empty([len(tracks), len(detections) + len(tracks)], SingleDistanceHypothesis)
        for i in range(len(tracks)):
            row = np.empty([len(detections) + len(tracks)], SingleDistanceHypothesis)
            track_hypotheses = self.hypothesiser.hypothesise(tracks[i], detections, time)
            for hypothesis in track_hypotheses:
                if isinstance(hypothesis.measurement, MissedDetection):
                    row[len(detections) + i] = hypothesis
                else:
                    row[int(detections.index(hypothesis.measurement))] = hypothesis
            hypothesis_matrix[i] = row
        
        # Generate 2d array "matrix" of distances
        distance_matrix = np.empty([len(tracks), len(detections) + len(tracks)])
        for x in range(hypothesis_matrix.shape[0]):
            for y in range(hypothesis_matrix.shape[1]):
                if hypothesis_matrix[x][y] is None:
                    distance_matrix[x][y] = np.inf
                else:
                    distance_matrix[x][y] = hypothesis_matrix[x][y].distance
        
        # Use "shortest path" assignment algorithm on distance matrix to assign tracks to nearest detection
        gain, col4row, row4col = assignAlgs.assign2D(distance_matrix)
        
        # Ensure the problem was feasible
        if gain == np.empty(0):
            print("IMPOSSIBLE ASSIGNMENT")
            return
        
        # Generate dict of key/value pairs
        associations = {}
        for j in range(len(col4row)):
            associations[tracks[j]] = hypothesis_matrix[j][int(col4row[j])]
        
        return associations

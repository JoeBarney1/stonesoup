from typing import Optional
import datetime
from typing import Sequence, Collection

import numpy as np
from ordered_set import OrderedSet

from ..base import Property
from ..models.measurement import MeasurementModel
from ..models.transition import TransitionModel
from ..reader import GroundTruthReader
from ..types import (Detection, Clutter, GaussianState, GroundTruthState,
                     GroundTruthPath, Probability, State)
from .base import DetectionSimulator, GroundTruthSimulator
from stonesoup.buffered_generator import BufferedGenerator


class SingleTargetGroundTruthSimulator(GroundTruthSimulator):
    """Target simulator that produces a single target"""
    transition_model: TransitionModel = Property(
        doc="Transition Model used as propagator for track.")
    initial_state: State = Property(doc="Initial state to use to generate ground truth")
    timestep: datetime.timedelta = Property(
        default=datetime.timedelta(seconds=1),
        doc="Time step between each state. Default one second.")
    number_steps = Property(
        int, default=100, doc="Number of time steps to run for")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._groundtruth_paths = set()

    @property
    def groundtruth_paths(self):
        return self._groundtruth_paths.copy()

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        self._groundtruth_paths = set()
        time = self.initial_state.timestamp or datetime.datetime.now()

        gttrack = GroundTruthPath([
            GroundTruthState(self.initial_state.state_vector, timestamp=time)])
        self._groundtruth_paths.add(gttrack)
        yield time, self.groundtruth_paths

        for _ in range(self.number_steps - 1):
            time += self.timestep
            # Move track forward
            trans_state_vector = self.transition_model.function(
                gttrack[-1], noise=True, time_interval=self.timestep)
            gttrack.append(GroundTruthState(
                trans_state_vector, timestamp=time))

            yield time, self.groundtruth_paths


class SwitchOneTargetGroundTruthSimulator(SingleTargetGroundTruthSimulator):
    """Target simulator that produces a single target. This target switches
    between multiple transition models based on a markov matrix
    (:attr:`model_probs`)"""
    transition_models: Sequence[TransitionModel] = Property(
        doc="List of transition models to be used, ensure that they all have the same dimensions.")
    model_probs: np.ndarray = Property(doc="A matrix of probabilities.\
    The element in the ith row and the jth column is the probability of\
     switching from the ith transition model in :attr:`transition_models`\
     to the jth")
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation."
                                                     " Default None")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random.mtrand._rand

    @property
    def transition_model(self):
        self.index = self.random_state.choice(range(0, len(self.transition_models)),
                                              p=self.model_probs[self.index])
        return self.transition_models[self.index]


class MultiTargetGroundTruthSimulator(SingleTargetGroundTruthSimulator):
    """Target simulator that produces multiple targets.

    Targets are created and destroyed randomly, as defined by the birth rate
    and death probability."""
    transition_model: TransitionModel = Property(
        doc="Transition Model used as propagator for track.")
    initial_state: GaussianState = Property(doc="Initial state to use to generate states")
    birth_rate: float = Property(
        default=1.0, doc="Rate at which tracks are born. Expected number of occurrences (λ) in "
                         "Poisson distribution. Default 1.0.")
    death_probability: Probability = Property(
        default=0.1, doc="Probability of track dying in each time step. Default 0.1.")
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation."
                                                     " Default None")
    preexisting_states: Collection[StateVector] = Property(
        default=list(), doc="State vectors at time 0 for "
                            "groundtruths which should exist at the start of simulation.")
    initial_number_targets: int = Property(
        default=0, doc="Initial number of targets to be "
                       "simulated. These simulated targets will be made in addition to those "
                       "defined by :attr:`preexisting_states`.")

    def groundtruth_paths_gen(self):
        self._groundtruth_paths = set()
        time = self.initial_state.timestamp or datetime.datetime.now()

        for _ in range(self.number_steps):
            # Random drop tracks
            self._groundtruth_paths.difference_update(
                gttrack
                for gttrack in self.groundtruth_paths
                if np.random.rand() <= self.death_probability)

            # Move tracks forward
            for gttrack in self.groundtruth_paths:
                trans_state_vector = self.transition_model.function(
                    gttrack[-1], noise=True, time_interval=self.timestep)
                gttrack.append(GroundTruthState(
                    trans_state_vector, timestamp=time,
                    metadata={"index": self.index}))

            # Random create
            for _ in range(np.random.poisson(self.birth_rate)):
                gttrack = GroundTruthPath()
                gttrack.append(GroundTruthState(
                    self.initial_state.state_vector +
                    np.sqrt(self.initial_state.covar) @
                    np.random.randn(self.initial_state.ndim, 1),
                    timestamp=time))
                self._groundtruth_paths.add(gttrack)

            yield time, self.groundtruth_paths
            time += self.timestep


class SimpleDetectionSimulator(DetectionSimulator):
    """A simple detection simulator.

    Parameters
    ----------
    groundtruth : GroundTruthReader
        Source of ground truth tracks used to generate detections for.
    measurement_model : MeasurementModel
        Measurement model used in generating detections.
    """
    groundtruth = Property(GroundTruthReader)
    measurement_model = Property(MeasurementModel)
    meas_range = Property(np.ndarray)
    detection_probability = Property(Probability, default=0.9)
    clutter_rate = Property(float, default=2.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_detections = set()
        self.clutter_detections = set()
        self.index = 0
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random.mtrand._rand

    @property
    def clutter_spatial_density(self):
        """returns the clutter spatial density of the measurement space - num
        clutter detections per unit volume per timestep"""
        return self.clutter_rate/np.prod(np.diff(self.meas_range))

    def __in_state_space(self, detection):
        """
        Checks if a measurement is in the state space
        """
        for dim in range(self.meas_range.ndim):
            if not self.meas_range[dim][0] <= detection.state_vector[dim] \
                                            <= self.meas_range[dim][-1]:
                return False
        return True

        for time, tracks in self.groundtruth.groundtruth_paths_gen():
            self.real_detections.clear()
            self.clutter_detections.clear()

            for track in tracks:
                if np.random.rand() < self.detection_probability:
                    detection = Detection(
                        H @ track[-1].state_vector +
                        self.measurement_model.rvs(),
                        timestamp=track[-1].timestamp)
                    detection.clutter = False
                    self.real_detections.add(detection)

            # generate clutter
            for _ in range(np.random.poisson(self.clutter_rate)):
                detection = Clutter(
                    np.random.rand(H.shape[0], 1) *
                    np.diff(self.meas_range) + self.meas_range[:, :1],
                    timestamp=time)
                self.clutter_detections.add(detection)

            yield time, self.detections

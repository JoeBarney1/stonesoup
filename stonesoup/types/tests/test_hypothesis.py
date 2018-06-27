from copy import deepcopy

import numpy as np
import pytest

from ..prediction import Prediction, MeasurementPrediction
from ..detection import Detection
from ..track import Track
from ..hypothesis import (
    Hypothesis, DistanceHypothesis, JointHypothesis, DistanceJointHypothesis)

prediction = Prediction(np.array([[1], [0]]))
measurement_prediction = MeasurementPrediction(np.array([[1], [0]]))
detection = Detection(np.array([[1], [0]]))
distance = float(1)


def test_single_hypothesis():
    """Single Measurement Hypothesis type test"""

    with pytest.raises(TypeError):
        Hypothesis(prediction, measurement_prediction, detection)


def test_single_distance_hypothesis():
    """Single Measurement Distance Hypothesis type test"""

    hypothesis = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)

    assert hypothesis.prediction is prediction
    assert hypothesis.measurement_prediction is measurement_prediction
    assert hypothesis.detection is detection
    assert hypothesis.distance is distance
    assert hypothesis.measurement_prediction is measurement_prediction
    assert hypothesis.weight == 1/distance

    hypothesis.distance = 0
    assert hypothesis.weight == float('inf')


def test_single_distance_hypothesis_comparison():
    """Single Measurement Distance Hypothesis comparison test"""

    h1 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)
    h2 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance + 1)

    assert h1 > h2
    assert h2 < h1
    assert h1 <= h1
    assert h1 >= h1
    assert h1 == h1


def test_single_probability_hypothesis_comparison():
    """Single Measurement Probability Hypothesis comparison test"""

    h1 = SingleProbabilityHypothesis(
        prediction, detection, 0.9, measurement_prediction)
    h2 = SingleProbabilityHypothesis(
        prediction, detection, 0.1, measurement_prediction)
    h3 = SingleHypothesis(prediction, detection, measurement_prediction)

    assert h1 > h2
    assert h2 < h1
    assert h1 <= h1
    assert h1 >= h1
    assert h1 == h1
    assert h1 != h3


def test_probability_joint_hypothesis():
    """Probability Joint Hypothesis type test"""

    t1 = Track()
    t2 = Track()
    h1 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)
    h2 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)

    hypotheses = {t1: h1, t2: h2}
    joint_hypothesis = JointHypothesis(hypotheses)

    assert isinstance(joint_hypothesis,
                      ProbabilityJointHypothesis)
    assert joint_hypothesis[t1] is h1
    assert joint_hypothesis[t2] is h2
    assert joint_hypothesis.distance == distance * 2


def test_distance_joint_hypothesis_comparison():
    """Distance Joint Hypothesis comparison test"""

    t1 = Track()
    t2 = Track()
    h1 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)
    h2 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance)
    h3 = DistanceHypothesis(
        prediction, measurement_prediction, detection, distance + 1)

    hypotheses1 = {t1: h1, t2: h2}
    hypotheses2 = {t1: h1, t2: h3}
    j1 = JointHypothesis(hypotheses1)
    j1.normalise()
    j2 = JointHypothesis(hypotheses2)

    assert j1 > j2
    assert j2 < j1
    assert j1 <= j1
    assert j1 >= j1
    assert j1 == j1


def test_invalid_joint_hypothesis():
    """Invalid Joint Hypothesis test"""

    t1 = Track()
    t2 = Track()

    h1 = object()
    h2 = object()

    hypotheses = {t1: h1, t2: h2}

    with pytest.raises(NotImplementedError):
        JointHypothesis(hypotheses)

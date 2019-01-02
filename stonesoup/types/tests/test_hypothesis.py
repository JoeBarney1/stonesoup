# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ..prediction import StatePrediction, StateMeasurementPrediction
from ..detection import Detection
from ..track import Track
from ..hypothesis import (
    SingleMeasurementHypothesis,
    SingleMeasurementDistanceHypothesis,
    SingleMeasurementProbabilityHypothesis,
    JointHypothesis,
    ProbabilityJointHypothesis,
    DistanceJointHypothesis)

prediction = StatePrediction(np.array([[1], [0]]))
measurement_prediction = StateMeasurementPrediction(np.array([[1], [0]]))
detection = Detection(np.array([[1], [0]]))
distance = float(1)


def test_single_measurement_hypothesis():
    """Single Measurement Hypothesis type test"""

    hypothesis = SingleMeasurementHypothesis(prediction, detection)
    assert hypothesis.prediction is prediction
    assert hypothesis.measurement is detection
    assert hypothesis.measurement_prediction is None
    assert hypothesis

    hypothesis = SingleMeasurementHypothesis(prediction, detection,
                                             measurement_prediction)
    assert hypothesis.prediction is prediction
    assert hypothesis.measurement is detection
    assert hypothesis.measurement_prediction is measurement_prediction
    assert hypothesis

    hypothesis = SingleMeasurementHypothesis(prediction, None)
    assert hypothesis.prediction is prediction
    assert hypothesis.measurement is None
    assert hypothesis.measurement_prediction is None
    assert not hypothesis


def test_single_measurement_distance_hypothesis():
    """Single Measurement Distance Hypothesis type test"""

    hypothesis = SingleMeasurementDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)

    assert hypothesis.prediction is prediction
    assert hypothesis.measurement is detection
    assert hypothesis.distance is distance
    assert hypothesis.measurement_prediction is measurement_prediction


def test_single_measurement_distance_hypothesis_comparison():
    """Single Measurement Distance Hypothesis comparison test"""

    h1 = SingleMeasurementDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)
    h2 = SingleMeasurementDistanceHypothesis(
        prediction, detection, distance + 1, measurement_prediction)

    assert h1 > h2
    assert h2 < h1
    assert h1 <= h1
    assert h1 >= h1
    assert h1 == h1


def test_single_measurement_probability_hypothesis_comparison():
    """Single Measurement Probability Hypothesis comparison test"""

    h1 = SingleMeasurementProbabilityHypothesis(
        prediction, detection, 0.9, measurement_prediction)
    h2 = SingleMeasurementProbabilityHypothesis(
        prediction, detection, 0.1, measurement_prediction)

    assert h1 > h2
    assert h2 < h1
    assert h1 <= h1
    assert h1 >= h1
    assert h1 == h1


def test_probability_joint_hypothesis():
    """Probability Joint Hypothesis type test"""

    t1 = Track()
    t2 = Track()
    h1 = SingleMeasurementProbabilityHypothesis(
        prediction, detection, 0.9, measurement_prediction)
    h2 = SingleMeasurementProbabilityHypothesis(
        prediction, detection, 0.1, measurement_prediction)

    hypotheses = {t1: h1, t2: h2}
    joint_hypothesis = JointHypothesis(hypotheses)

    assert isinstance(joint_hypothesis,
                      ProbabilityJointHypothesis)
    assert joint_hypothesis[t1] is h1
    assert joint_hypothesis[t2] is h2
    assert joint_hypothesis.probability == h1.probability * h2.probability


def test_probability_joint_hypothesis_comparison():
    """Probability Joint Hypothesis comparison test"""

    t1 = Track()
    t2 = Track()
    h1 = SingleMeasurementProbabilityHypothesis(
        prediction, detection, 0.75, measurement_prediction)
    h2 = SingleMeasurementProbabilityHypothesis(
        prediction, detection, 0.75, measurement_prediction)
    h3 = SingleMeasurementProbabilityHypothesis(
        prediction, detection, 0.25, measurement_prediction)

    hypotheses1 = {t1: h1, t2: h2}
    hypotheses2 = {t1: h1, t2: h3}
    j1 = JointHypothesis(hypotheses1)
    j1.normalize()
    j2 = JointHypothesis(hypotheses2)
    j2.normalize()

    assert j1 > j2
    assert j2 < j1
    assert j1 <= j1
    assert j1 >= j1
    assert j1 == j1


def test_distance_joint_hypothesis():
    """Distance Joint Hypothesis type test"""

    t1 = Track()
    t2 = Track()
    h1 = SingleMeasurementDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)
    h2 = SingleMeasurementDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)

    hypotheses = {t1: h1, t2: h2}
    joint_hypothesis = JointHypothesis(hypotheses)

    assert isinstance(joint_hypothesis,
                      DistanceJointHypothesis)
    assert joint_hypothesis[t1] is h1
    assert joint_hypothesis[t2] is h2
    assert joint_hypothesis.distance == distance * 2


def test_distance_joint_hypothesis_comparison():
    """Distance Joint Hypothesis comparison test"""

    t1 = Track()
    t2 = Track()
    h1 = SingleMeasurementDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)
    h2 = SingleMeasurementDistanceHypothesis(
        prediction, detection, distance, measurement_prediction)
    h3 = SingleMeasurementDistanceHypothesis(
        prediction, detection, distance + 1, measurement_prediction)

    hypotheses1 = {t1: h1, t2: h2}
    hypotheses2 = {t1: h1, t2: h3}
    j1 = JointHypothesis(hypotheses1)
    j2 = JointHypothesis(hypotheses2)

    assert j1 > j2
    assert j2 < j1
    assert j1 <= j1
    assert j1 >= j1
    assert j1 == j1


def test_invalid_single_measurement_joint_hypothesis():
    """Invalid Single Measurement Joint Hypothesis test"""

    t1 = Track()
    t2 = Track()

    h1 = object()
    h2 = object()

    hypotheses = {t1: h1, t2: h2}

    with pytest.raises(NotImplementedError):
        JointHypothesis(hypotheses)

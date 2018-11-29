"""Test for updater.kalman module"""

import pytest
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types import GaussianState, GaussianStatePrediction,\
    GaussianMeasurementPrediction, Hypothesis


@pytest.fixture(params=[KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater,
                        IteratedKalmanUpdater, SchmidtKalmanUpdater, CubatureKalmanUpdater])
def updater_class(request):
    return request.param


    # Define predicted state
    prediction = GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                         np.array([[4.1123, 0.0013],
                                                   [0.0013, 0.0365]]))
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        lg.matrix()@prediction.mean,
        lg.matrix()@prediction.covar@lg.matrix().T+lg.covar(),
        cross_covar=prediction.covar@lg.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar@np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar @ kalman_gain.T)

    # Initialise a kalman updater
    updater = updater_class(measurement_model=measurement_model, use_joseph_cov=use_joseph_cov)

    # Get and assert measurement prediction without measurement noise
    measurement_prediction = updater.predict_measurement(prediction, measurement_noise=False)
    assert np.allclose(measurement_prediction.mean,
                       eval_measurement_prediction.mean,
                       0, atol=1.e-14)
    assert np.allclose(measurement_prediction.covar,
                       eval_measurement_prediction.covar - measurement_model.covar(),
                       0, atol=1.e-14)
    assert np.allclose(measurement_prediction.cross_covar,
                       eval_measurement_prediction.cross_covar,
                       0, atol=1.e-13)

    # Get and assert measurement prediction
    measurement_prediction = updater.get_measurement_prediction(prediction)
    assert(np.array_equal(measurement_prediction.mean,
                          eval_measurement_prediction.mean))
    assert(np.array_equal(measurement_prediction.covar,
                          eval_measurement_prediction.covar))
    assert(np.array_equal(measurement_prediction.cross_covar,
                          eval_measurement_prediction.cross_covar))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(Hypothesis(
        prediction=prediction,
        measurement=measurement))
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.allclose(posterior.covar, eval_posterior.covar))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(posterior.hypothesis.measurement_prediction.covar,
                           measurement_prediction.covar))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(Hypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.allclose(posterior.covar, eval_posterior.covar))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(posterior.hypothesis.measurement_prediction.covar,
                           measurement_prediction.covar))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)


def test_extendedkalman():

    # Initialise a measurement model
    lg = LinearGaussian(ndim_state=2, mapping=[0],
                        noise_covar=np.array([[0.04]]))

    # Define predicted state
    prediction = GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                         np.array([[4.1123, 0.0013],
                                                   [0.0013, 0.0365]]))
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        lg.matrix()@prediction.mean,
        lg.matrix()@prediction.covar@lg.matrix().T+lg.covar(),
        cross_covar=prediction.covar@lg.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar@np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T)

    # Test Square root form returns the same as standard form
    updater = KalmanUpdater(measurement_model=measurement_model)
    sqrt_updater = SqrtKalmanUpdater(measurement_model=measurement_model, qr_method=False)
    qr_updater = SqrtKalmanUpdater(measurement_model=measurement_model, qr_method=True)

    # Get and assert measurement prediction
    measurement_prediction = updater.get_measurement_prediction(prediction)
    assert(np.array_equal(measurement_prediction.mean,
                          eval_measurement_prediction.mean))
    assert(np.array_equal(measurement_prediction.covar,
                          eval_measurement_prediction.covar))
    assert(np.array_equal(measurement_prediction.cross_covar,
                          eval_measurement_prediction.cross_covar))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(Hypothesis(
        prediction=prediction,
        measurement=measurement))
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.allclose(posterior.covar, eval_posterior.covar))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(posterior.hypothesis.measurement_prediction.covar,
                           measurement_prediction.covar))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(Hypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.allclose(posterior.covar, eval_posterior.covar))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(posterior.hypothesis.measurement_prediction.covar,
                           measurement_prediction.covar))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

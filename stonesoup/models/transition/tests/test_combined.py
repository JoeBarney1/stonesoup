import pytest
import datetime
from numbers import Real

import numpy as np

from ..linear import (
    LinearGaussianTimeInvariantTransitionModel, ConstantVelocity,
    CombinedLinearGaussianTransitionModel)


@pytest.mark.parametrize("comb_model", [CombinedGaussianTransitionModel,
                                        CombinedLinearGaussianTransitionModel])
def test__linear_combined(comb_model):
    F = 3*np.eye(3)
    Q = 3*np.eye(3)
    model_1 = LinearGaussianTimeInvariantTransitionModel(
        transition_matrix=F, covariance_matrix=Q)
    model_2 = ConstantVelocity(noise_diff_coeff=3)
    model_3 = ConstantVelocity(noise_diff_coeff=1)
    model_4 = ConstantVelocity(noise_diff_coeff=1)

    DIM = 9

    combined_model = CombinedGaussianTransitionModel(
        [model_1, model_2, model_3, model_4])
    t_delta = datetime.timedelta(0, 3)

    x_prior = np.ones([DIM, 1])
    x_post = np.ones([DIM, 1])

    state = State(x_prior)
    sv = state.state_vector
    state.state_vector = StateVectors([sv, sv, sv])

    assert DIM == combined_model.ndim_state
    assert (DIM, DIM) == combined_model.jacobian(State(x_prior),
                                                 time_interval=t_delta).shape
    assert (DIM, DIM) == combined_model.covar(time_interval=t_delta).shape
    assert (DIM, 1) == combined_model.function(
        State(x_prior), noise=np.random.randn(DIM, 1),
        time_interval=t_delta).shape
    # Test vectorized handling i.e. multiple state vector inputs
    assert state.state_vector.shape == combined_model.function(
        state,
        time_interval=t_delta).shape
    assert (DIM, 1) == combined_model.rvs(time_interval=t_delta).shape
    assert isinstance(
        combined_model.pdf(x_post, x_prior, time_interval=t_delta), Real)

import warnings

import numpy as np
import scipy.linalg as la
from functools import lru_cache

from ..base import Property
from .base import Updater
from ..base import Property
from ..types import (GaussianMeasurementPrediction,
                     GaussianStateUpdate)


class KalmanUpdater(Updater):
    r"""A class which embodies Kalman-type updaters; also a class which
    performs measurement update step as in the standard Kalman filter.

    The Kalman updaters assume :math:`h(\mathbf{x}) = H \mathbf{x}` with
    additive noise :math:`\sigma = \mathcal{N}(0,R)`. Daughter classes can
    overwrite to specify a more general measurement model
    :math:`h(\mathbf{x})`.

    :meth:`update` first calls :meth:`predict_measurement` function which
    proceeds by calculating the predicted measurement, innovation covariance
    and measurement cross-covariance,

    .. math::

        \mathbf{z}_{k|k-1} &= H_k \mathbf{x}_{k|k-1}

        S_k &= H_k P_{k|k-1} H_k^T + R_k

        \Upsilon_k &= P_{k|k-1} H_k^T

    where :math:`P_{k|k-1}` is the predicted state covariance.
    :meth:`predict_measurement` returns a
    :class:`~.GaussianMeasurementPrediction`. The Kalman gain is then
    calculated as,

    .. math::

        K_k = \Upsilon_k S_k^{-1}

    and the posterior state mean and covariance are,

    .. math::

        \mathbf{x}_{k|k} &= \mathbf{x}_{k|k-1} + K_k (\mathbf{z}_k - H_k
        \mathbf{x}_{k|k-1})

        P_{k|k} &= P_{k|k-1} - K_k S_k K_k^T

    These are returned as a :class:`~.GaussianStateUpdate` object.
    """

    def get_measurement_prediction(self, state_prediction,
                                   measurement_model=None, **kwargs):
        """Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.\
            Should be used in cases where the measurement model is dependent\
            on the received measurement.\
            (the default is ``None``, in which case the updater will use the\
            measurement model specified on initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """
        return self._check_measurement_model(
            measurement_model).matrix(**kwargs)

        # Measurement model parameters
        if measurement_model is None:
            measurement_model = self.measurement_model
        measurement_matrix, measurement_noise_covar = \
            self._extract_model_parameters(measurement_model)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_matrix,
                                                     measurement_noise_covar)

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

    def update(self, hypothesis, **kwargs):
        """Kalman Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.GaussianStateUpaate`
            The computed state posterior
        """
        post_mean = predicted_state.state_vector + \
            kalman_gain @ (measurement.state_vector - measurement_prediction.state_vector)
        return post_mean.view(StateVector)

        # Extract model parameters
        measurement_matrix, measurement_noise_covar = \
            self._extract_model_parameters(self.measurement_model,
                                           hypothesis.measurement)

        # If no measurement prediction is provided with hypothesis
        if hypothesis.measurement_prediction is None:
            # Perform full update step
            posterior_mean, posterior_covar, meas_pred_mean,\
                meas_pred_covar, cross_covar, _ = \
                self.update_lowlevel(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    measurement_matrix,
                    measurement_noise_covar,
                    hypothesis.measurement.state_vector
                )
            # Augment hypothesis with measurement prediction
            hypothesis = Hypothesis(hypothesis.prediction,
                                    hypothesis.measurement,
                                    GaussianMeasurementPrediction(
                                        meas_pred_mean, meas_pred_covar,
                                        hypothesis.prediction.timestamp,
                                        cross_covar)
                                    )
        else:
            # Otherwise, utilise the provided measurement prediction
            posterior_mean, posterior_covar, _ = \
                self.update_on_measurement_prediction(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    hypothesis.measurement.state_vector,
                    hypothesis.measurement_prediction.mean,
                    hypothesis.measurement_prediction.covar,
                    hypothesis.measurement_prediction.cross_covar,
                    measurement_matrix,
                    measurement_noise_covar
                )

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None, measurement_noise=True,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.GaussianState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`, :math:`P_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        measurement_noise : bool
            Whether to include measurement noise :math:`R` with innovation covariance.
            Default `True`
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`GaussianMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        x_post, P_post, K = \
            KalmanUpdater.update_on_measurement_prediction(x_pred, P_pred,
                                                           y, y_pred, S,
                                                           Pxy, H, R)

        return x_post, P_post, y_pred, S, Pxy, K

        # The measurement cross covariance and innovation covariance
        meas_cross_cov = self._measurement_cross_covariance(predicted_state, hh)
        innov_cov = self._innovation_covariance(
            meas_cross_cov, hh, measurement_model, measurement_noise, **kwargs)

        return MeasurementPrediction.from_state(
            predicted_state, pred_meas, innov_cov, cross_covar=meas_cross_cov)

    def update(self, hypothesis, **kwargs):
        r"""The Kalman update method. Given a hypothesised association between
        a predicted state or predicted measurement and an actual measurement,
        calculate the posterior state.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.GaussianStateUpdate`
            The posterior state Gaussian with mean :math:`\mathbf{x}_{k|k}` and
            covariance :math:`P_{x|x}`

        """
        # Get the predicted state out of the hypothesis
        predicted_state = hypothesis.prediction

        y_pred = H@x_pred
        S = H@P_pred@H.T + R
        Pxy = P_pred@H.T

            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state, measurement_model=measurement_model, **kwargs)

    @staticmethod
    def update_on_measurement_prediction(x_pred, P_pred, y,
                                         y_pred, S, Pxy, H=None, R=None):
        """Low level Kalman Filter update, based on a measurement prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector
        y_pred: :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        S: :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance
        H: :class:`numpy.ndarray` of shape (Nm,Nm), optional
            The measurement model matrix. If both `H` and `R` are provided
            then the update will be performed based on the slower, but more
            numerically stable, "Joseph form" update equation:
            :math:`P_{k|k} = (I-K_kH_k)P_{k|k-1}(I-K_kH_k)^T + K_kR_kK_k^T`
            (default is `None`)
        R: :class:`numpy.ndarray` of shape (Nm,Nm), optional
            The measurement model matrix. See information for `H` above.

        # Posterior mean
        posterior_mean = self._posterior_mean(predicted_state, kalman_gain,
                                              hypothesis.measurement,
                                              hypothesis.measurement_prediction)

        K = Pxy@np.linalg.pinv(S)

        x_post = x_pred + K@(y-y_pred)

        if(H is not None and R is not None):
            # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
            # and works for non-optimal K vs the equation
            # P = (I-KH)P usually seen in the literature.
            ndim_state = x_pred.shape[0]
            I_KH = np.eye(ndim_state) - K@H
            P_post = I_KH@P_pred@I_KH.T + K@R@K.T
        elif(H is not None):
            ndim_state = x_pred.shape[0]
            P_post = (np.eye(ndim_state) - K@H)@P_pred
        else:
            P_post = P_pred - K@Pxy.T
            P_post = (P_post+P_post.T)/2

        return x_post, P_post, K

    @staticmethod
    def _extract_model_parameters(measurement_model, measurement=None,
                                  **kwargs):
        """Extract measurement model parameters

        Parameters
        ----------
        measurement_model: :class:`~.MeasurementModel`
            A measurement model whose parameters are to be extracted
        measurement : :class:`~.Detection`, optional
            If provided and `measurement.measurement_model` is not `None`,\
            then its parameters will be returned instead\
            (the default is `None`, in which case `self.measurement_model`'s\
            parameters will be returned)

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model transformation matrix
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model covariance matrix
        """

        if(measurement is not None
           and measurement.measurement_model is not None):
            measurement_matrix = measurement.measurement_model.matrix(**kwargs)
            measurement_noise_covar = measurement.measurement_model.covar(
                **kwargs)
        else:
            measurement_matrix = measurement_model.matrix(**kwargs)
            measurement_noise_covar = measurement_model.covar(**kwargs)

        return measurement_matrix, measurement_noise_covar


class ExtendedKalmanUpdater(KalmanUpdater):
    r"""The Extended Kalman Filter version of the Kalman Updater. Inherits most
    of the functionality from :class:`~.KalmanUpdater`.

    The difference is that the measurement model may now be non-linear, though
    must be differentiable to return the linearisation of :math:`h(\mathbf{x})`
    via the matrix :math:`H` accessible via :meth:`~.NonLinearModel.jacobian`.

    """
    # TODO: Enforce the fact that this version of MeasurementModel must be
    # TODO: capable of executing :attr:`jacobian()`
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="A measurement model. This need not be defined if a measurement "
            "model is provided in the measurement. If no model specified on "
            "construction, or in the measurement, then error will be thrown. "
            "Must be linear or capable or implement the "
            ":meth:`~.NonLinearModel.jacobian`.")

    def _measurement_matrix(self, predicted_state, measurement_model=None,
                            linearisation_point=None, **kwargs):
        r"""Return the (via :meth:`NonLinearModel.jacobian`) measurement matrix

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            Passed to :meth:`~.MeasurementModel.matrix` if linear
            or :meth:`~.MeasurementModel.jacobian` if not

        Returns
        -------
        : :class:`numpy.ndarray`
            The measurement matrix, :math:`H_k`

        """

        measurement_model = self._check_measurement_model(measurement_model)

        if isinstance(measurement_model, LinearModel):
            return measurement_model.matrix(**kwargs)
        else:
            if linearisation_point is None:
                linearisation_point = predicted_state
            return measurement_model.jacobian(linearisation_point, **kwargs)


class UnscentedKalmanUpdater(KalmanUpdater):
    """The Unscented Kalman Filter version of the Kalman Updater. Inherits most
    of the functionality from :class:`~.KalmanUpdater`.

    In this case the :meth:`predict_measurement` function uses the
    :func:`unscented_transform` function to estimate a (Gaussian) predicted
    measurement. This is then updated via the standard Kalman update equations.

    """
    # Can be non-linear and non-differentiable
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="The measurement model to be used. This need not be defined if a "
            "measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")
    alpha: float = Property(
        default=0.5,
        doc="Primary sigma point spread scaling parameter. Default is 0.5.")
    beta: float = Property(
        default=2,
        doc="Used to incorporate prior knowledge of the distribution. If the "
            "true distribution is Gaussian, the value of 2 is optimal. "
            "Default is 2")
    kappa: float = Property(
        default=None,
        doc="Secondary spread scaling parameter. Default is calculated as "
            "3-Ns")

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None, measurement_noise=True,
                            **kwargs):
        """Unscented Kalman Filter measurement prediction step. Uses the
        unscented transform to estimate a Gauss-distributed predicted
        measurement.

        Parameters
        ----------
        predicted_state : :class:`~.GaussianStatePrediction`
            A predicted state
        measurement_model : :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.
            This should be used in cases where the measurement model is
            dependent on the received measurement (the default is `None`, in
            which case the updater will use the measurement model specified on
            initialisation)
        measurement_noise : bool
            Whether to include measurement noise :math:`R` with innovation covariance

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction

        """

        measurement_model = self._check_measurement_model(measurement_model)

        sigma_points, mean_weights, covar_weights = \
            gauss2sigma(predicted_state,
                        self.alpha, self.beta, self.kappa)

        covar_noise = measurement_model.covar(**kwargs) if measurement_noise else None
        meas_pred_mean, meas_pred_covar, cross_covar, *_ = \
            unscented_transform(sigma_points, mean_weights, covar_weights,
                                measurement_model.function, covar_noise=covar_noise)

        return MeasurementPrediction.from_state(
            predicted_state, meas_pred_mean, meas_pred_covar, cross_covar=cross_covar)


class SqrtKalmanUpdater(ExtendedKalmanUpdater):
    r"""The Square root version of the Kalman Updater.

    The input :class:`~.State` is a :class:`~.SqrtGaussianState` which means
    that the covariance of the predicted state is stored in square root form.
    This can be achieved by keeping :attr:`covar` attribute as :math:`L` where
    the 'full' covariance matrix :math:`P_{k|k-1} = L_{k|k-1} L^T_{k|k-1}`
    [Eq1].

    In its basic form :math:`L` is the lower triangular matrix returned via
    Cholesky factorisation. There's no reason why other forms that satisfy Eq 1
    above can't be used.

    References
    ----------
    1. Schmidt, S.F. 1970, Computational techniques in Kalman filtering, NATO advisory group for
       aerospace research and development, London 1970
    2. Andrews, A. 1968, A square root formulation of the Kalman covariance equations, AIAA
       Journal, 6:6, 1165-1166

    """
    qr_method: bool = Property(
        default=False,
        doc="A switch to do the update via a QR decomposition, rather than using the (vector form "
            "of) the Potter method.")

    def _measurement_cross_covariance(self, predicted_state, measurement_matrix):
        """
        Return the measurement cross covariance matrix, :math:`P_{k|k-1} H_k^T`. This differs
        slightly from its parent in that it the predicted state covariance (now a square root
        matrix) is transposed.

        Parameters
        ----------
        predicted_state : :class:`SqrtGaussianState`
            The predicted state which contains the square root form of the covariance matrix
            :math:`W` as :attr:`.covar` attribute
        measurement_matrix : numpy.array
            The measurement matrix, :math:`H`

        Returns
        -------
        :  numpy.ndarray
            The measurement cross-covariance matrix

        """
        return predicted_state.sqrt_covar.T @ measurement_matrix.T

    def _innovation_covariance(self, m_cross_cov, meas_mat, meas_mod, measurement_noise, **kwargs):
        """Compute the innovation covariance

        Parameters
        ----------
        m_cross_cov : numpy.ndarray
            The measurement cross covariance matrix
        meas_mat : numpy.ndarray
            The measurement matrix. Not required in this instance. Ignored.
        meas_mod : :class:`~.MeasurementModel`
            Measurement model. The class attribute :attr:`sqrt_covar` indicates whether this is
            passed in square root form. If it doesn't exist then :attr:`covar` is assumed to exist
            and is used instead.
        measurement_noise : bool
            Include measurement noise or not

        Returns
        -------
        : numpy.ndarray
            The innovation covariance

        """
        innov_covar = m_cross_cov.T @ m_cross_cov
        if measurement_noise:
            # If the measurement covariance matrix is square root then square it
            try:
                meas_cov = meas_mod.sqrt_covar @ meas_mod.sqrt_covar.T
            except AttributeError:
                meas_cov = meas_mod.covar(**kwargs)
            innov_covar += meas_cov

        return innov_covar

    def _posterior_covariance(self, hypothesis):
        """
        Return the posterior covariance for a given hypothesis. Hypothesis contains the predicted
        state covariance in square root form, the measurement prediction (which in turn contains
        the measurement cross covariance, :math:`P_{k|k-1} H_k^T and the innovation covariance,
        :math:`S = H_k P_{k|k-1} H_k^T + R`, not in square root form). The hypothesis or the
        updater contain the measurement noise matrix. The :attr:`sqrt_measurement_noise` flag
        indicates whether we should use the square root form of this matrix (True) or its full
        form (False).

        Parameters
        ----------
        hypothesis: :class:`~.Hypothesis`
            A hypothesised association between state prediction and measurement

        Method
        ------
        If the :attr:`qr_method` flag is set to True then the update proceeds via a QR
        decomposition which requires only one further matrix inversion (see [1]), rather than
        three plus a Cholesky factorisation, for the method set out in [2].

        Returns
        -------
        : numpy.array
            The posterior covariance matrix rendered via the Kalman update process in
            lower-triangular form.
        : numpy.array
            The Kalman gain, :math:`K = P_{k|k-1} H_k^T S^{-1}`

        """
        # Do we already have a measurement model?
        measurement_model = \
            self._check_measurement_model(hypothesis.measurement.measurement_model)
        # Square root of the noise covariance, account for the fact that it may be supplied in one
        # of two ways
        try:
            sqrt_noise_cov = measurement_model.sqrt_covar
        except AttributeError:
            sqrt_noise_cov = la.sqrtm(measurement_model.covar())

        if self.qr_method:
            # The prior and noise covariances and the measurement matrix
            sqrt_prior_cov = hypothesis.prediction.sqrt_covar
            bigh = measurement_model.matrix()

            # Set up and execute the QR decomposition
            measdim = measurement_model.ndim_meas
            zeros = np.zeros((measurement_model.ndim_state, measdim))
            biga = np.block([[sqrt_noise_cov, bigh@sqrt_prior_cov], [zeros, sqrt_prior_cov]])
            _, upper = np.linalg.qr(biga.T)

            # Extract meaningful quantities
            atheta = upper.T
            sqrt_innov_cov = atheta[:measdim, :measdim]
            kalman_gain = atheta[measdim:, :measdim]@(np.linalg.inv(sqrt_innov_cov))
            post_cov = atheta[measdim:, measdim:]
        else:
            # Kalman gain
            kalman_gain = \
                hypothesis.prediction.sqrt_covar @ \
                hypothesis.measurement_prediction.cross_covar @ \
                np.linalg.inv(hypothesis.measurement_prediction.covar)
            # Square root of the innovation covariance
            sqrt_innov_cov = la.sqrtm(hypothesis.measurement_prediction.covar)
            # Posterior covariance
            post_cov = hypothesis.prediction.sqrt_covar @ \
                (np.identity(hypothesis.prediction.ndim) -
                 hypothesis.measurement_prediction.cross_covar @ np.linalg.inv(sqrt_innov_cov.T) @
                 np.linalg.inv(sqrt_innov_cov + sqrt_noise_cov) @
                 hypothesis.measurement_prediction.cross_covar.T)

        return post_cov, kalman_gain


class IteratedKalmanUpdater(ExtendedKalmanUpdater):
    r"""This version of the Kalman updater runs an iteration over the linearisation of the
    sensor function in order to refine the posterior state estimate. Specifically,

    .. math::

        \mathbf{x}_{k,i+1} &= \mathbf{x}_{k|k-1} + K_i [\mathbf{z} - h(\mathbf{x}_{k,i}) -
        H_i (\mathbf{x}_{k|k-1} - \mathbf{x}_{k,i}) ]

        P_{k,i+1} &= (I - K_i H_i) P_{k|k-1}

    where,

    .. math::

        H_i &= h^{\prime}(\mathbf{x}_{k,i}),

        K_i &= P_{k|k-1} H_i^T (H_i P_{k|k-1} H_i^T + R)^{-1}

    and

    .. math::

        \mathbf{x}_{k,0} &= \mathbf{x}_{k|k-1}

        P_{k,0} &= P_{k|k-1}

    It inherits from the ExtendedKalmanUpdater as it uses the same linearisation of the sensor
    function via the :meth:`_measurement_matrix()` function.
    """

    def get_measurement_prediction(self, state_prediction,
                                   measurement_model=None, **kwargs):
        """Extended Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.\
            Should be used in cases where the measurement model is dependent\
            on the received measurement.\
            (the default is ``None``, in which case the updater will use the\
            measurement model specified on initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """

        # Measurement model parameters
        if measurement_model is None:
            measurement_model = self.measurement_model
        measurement_matrix, measurement_noise_covar, measurement_function = \
            self._extract_model_parameters(measurement_model,
                                           state_prediction.state_vector)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_function,
                                                     measurement_matrix,
                                                     measurement_noise_covar)

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

    def update(self, hypothesis, **kwargs):
        """ Extended Kalman Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.GaussianState`
            The state posterior
        """
        post_mean = predicted_state.state_vector.copy()
        post_mean[np.ix_(~self.consider)] += \
            kalman_gain @ (measurement.state_vector - measurement_prediction.state_vector)
        return post_mean.view(StateVector)

        # Extract model parameters
        measurement_matrix, measurement_noise_covar, measurement_function = \
            self._extract_model_parameters(self.measurement_model,
                                           hypothesis.prediction.state_vector,
                                           hypothesis.measurement)

        # If no measurement prediction is provided with hypothesis
        if hypothesis.measurement_prediction is None:
            # Perform full update step
            posterior_mean, posterior_covar, meas_pred_mean,\
                meas_pred_covar, cross_covar, _ = \
                self.update_lowlevel(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    measurement_function,
                    measurement_matrix,
                    measurement_noise_covar,
                    hypothesis.measurement.state_vector
                )
            # Augment hypothesis with measurement prediction
            hypothesis = Hypothesis(hypothesis.prediction,
                                    hypothesis.measurement,
                                    GaussianMeasurementPrediction(
                                        meas_pred_mean, meas_pred_covar,
                                        hypothesis.prediction.timestamp,
                                        cross_covar)
                                    )
        else:
            posterior_mean, posterior_covar, _ = \
                self.update_on_measurement_prediction(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    hypothesis.measurement.state_vector,
                    hypothesis.measurement_prediction.mean,
                    hypothesis.measurement_prediction.covar,
                    hypothesis.measurement_prediction.cross_covar,
                    measurement_matrix,
                    measurement_noise_covar
                )

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)

    @staticmethod
    def update_lowlevel(x_pred, P_pred, h, H, R, y):
        """Low level Extended Kalman Filter update

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        h : function handle
            The (non-linear) measurement model function
            Must be of the form "y = fun(x)"
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model jacobian matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The posterior covariance matrix rendered via the Schmidt-Kalman update process.
        : numpy.ndarray
            The reduced form of the Kalman gain,
            :math:`K_s = (P_{ss,k|k-1} H_{s,k}^T + P_{sp,k|k-1} H_{p,k}^T) S^{-1}`

        """
        # Intermediate matrices P_p and H.
        pp = hypothesis.prediction.covar[np.tile(self.consider, (len(self.consider), 1))]
        pp = pp.reshape((len(self.consider), np.sum(self.consider)))
        hh = self._measurement_matrix(predicted_state=hypothesis.prediction)

        y_pred, S, Pxy = \
            ExtendedKalmanUpdater.get_measurement_prediction_lowlevel(x_pred,
                                                                      P_pred,
                                                                      h,
                                                                      H,
                                                                      R)

        x_post, P_post, K = \
            ExtendedKalmanUpdater.update_on_measurement_prediction(x_pred,
                                                                   P_pred,
                                                                   y,
                                                                   y_pred,
                                                                   S,
                                                                   Pxy,
                                                                   H,
                                                                   R)

        return x_post, P_post, y_pred, S, Pxy, K

    @staticmethod
    def get_measurement_prediction_lowlevel(x_pred, P_pred, h, H, R):
        """Low level Extended Kalman Filter measurement prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        h : function handle
            The (non-linear) measurement model function
            Must be of the form "y = fun(x)"
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model jacobian matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction

        """
        y_pred = h(x_pred)
        S = H@P_pred@H.T + R
        Pxy = P_pred@H.T

        return y_pred, S, Pxy

    @staticmethod
    def update_on_measurement_prediction(x_pred, P_pred, y,
                                         y_pred, S, Pxy, H=None, R=None):
        """Low level Extended Kalman Filter update, based on a measurement\
        prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector
        y_pred: :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        S: :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        return KalmanUpdater.update_on_measurement_prediction(x_pred, P_pred,
                                                              y, y_pred, S,
                                                              Pxy, H, R)

    @staticmethod
    def _extract_model_parameters(measurement_model, state_vector=None,
                                  measurement=None, **kwargs):
        """Extract measurement model parameters

        Parameters
        ----------
        measurement_model: :class:`~.MeasurementModel`
            A measurement model whose parameters are to be extracted
        measurement : :class:`~.Detection`, optional
            If provided and `measurement.measurement_model` is not `None`,\
            then its parameters will be returned instead\
            (the default is `None`, in which case `self.measurement_model`'s\
            parameters will be returned)

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model transformation matrix
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model covariance matrix
        """

        if(measurement is not None
           and measurement.measurement_model is not None):
            return ExtendedKalmanUpdater._extract_model_parameters(
                measurement.measurement_model, state_vector=state_vector)
        else:
            try:
                # Attempt to extract matrix from a LinearModel
                measurement_matrix = measurement_model.matrix(**kwargs)
            except AttributeError:
                # Else read jacobian from a NonLinearModel
                measurement_matrix = \
                    measurement_model.jacobian(state_vector,
                                               **kwargs)

            def measurement_function(x):
                return measurement_model.function(x, noise=0, **kwargs)

            measurement_noise_covar = measurement_model.covar(**kwargs)

        return measurement_matrix, measurement_noise_covar, \
            measurement_function


class UnscentedKalmanUpdater(KalmanUpdater):
    """Unscented Kalman Updater

    Perform measurement update step in the Unscented Kalman Filter.
    """

    alpha = Property(float, default=0.5,
                     doc="Primary sigma point spread scalling parameter.\
                         Typically 1e-3.")
    beta = Property(float, default=2,
                    doc="Used to incorporate prior knowledge of the distribution.\
                        If the true distribution is Gaussian, the value of 2\
                        is optimal.")
    kappa = Property(float, default=0,
                     doc="Secondary spread scaling parameter\
                        (default is calculated as 3-Ns)")

    def get_measurement_prediction(self, state_prediction,
                                   measurement_model=None, **kwargs):
        """Unscented Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.\
            Should be used in cases where the measurement model is dependent\
            on the received measurement.\
            (the default is ``None``, in which case the updater will use the\
            measurement model specified on initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """

        # Measurement model parameters
        if measurement_model is None:
            measurement_model = self.measurement_model

        measurement_function, measurement_noise_covar = \
            self._extract_model_parameters(measurement_model)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_function,
                                                     measurement_noise_covar,
                                                     self.alpha, self.beta,
                                                     self.kappa)

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_function,
                                                     measurement_noise_covar,
                                                     self.alpha, self.beta,
                                                     self.kappa)
        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

    def update(self, hypothesis, **kwargs):
        """ Unscented Kalman Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.GaussianState`
            The state posterior
        """

        # Extract model parameters
        measurement_function, measurement_noise_covar = \
            self._extract_model_parameters(self.measurement_model,
                                           hypothesis.measurement)

        # If no measurement prediction is provided with hypothesis
        if hypothesis.measurement_prediction is None:
            # Perform full update step
            posterior_mean, posterior_covar, meas_pred_mean,\
                meas_pred_covar, cross_covar, _ = \
                self.update_lowlevel(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    measurement_function,
                    measurement_noise_covar,
                    hypothesis.measurement.state_vector,
                    self.alpha, self.beta, self.kappa
                )
            # Augment hypothesis with measurement prediction
            hypothesis = Hypothesis(hypothesis.prediction,
                                    hypothesis.measurement,
                                    GaussianMeasurementPrediction(
                                        meas_pred_mean, meas_pred_covar,
                                        hypothesis.prediction.timestamp,
                                        cross_covar)
                                    )
        else:
            posterior_mean, posterior_covar, _ =\
                self.update_on_measurement_prediction(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    hypothesis.measurement.state_vector,
                    hypothesis.measurement_prediction.mean,
                    hypothesis.measurement_prediction.covar,
                    hypothesis.measurement_prediction.cross_covar
                )

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)

    @staticmethod
    def update_lowlevel(x_pred, P_pred, h, R, y, alpha, beta, kappa):
        """Low level Unscented Kalman Filter update

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        h : function handle
            The (non-linear) measurement model function
            Must be of the form "y = fun(x,w)"
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector
        alpha : float
            Spread of the sigma points.
        beta : float
            Used to incorporate prior knowledge of the distribution
            2 is optimal is the state is normally distributed.
        kappa : float
            Secondary spread scaling parameter

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        y_pred, S, Pxy = \
            UnscentedKalmanUpdater.get_measurement_prediction_lowlevel(
                x_pred, P_pred, h, R,
                alpha, beta, kappa)

        x_post, P_post, K = \
            UnscentedKalmanUpdater.update_on_measurement_prediction(
                x_pred, P_pred, y, y_pred, S, Pxy)

        return x_post, P_post, y_pred, S, Pxy, K

    @staticmethod
    def get_measurement_prediction_lowlevel(x_pred, P_pred, h, R,
                                            alpha, beta, kappa):
        """Low level Unscented Kalman Filter measurement prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        h : function handle
            The (non-linear) measurement model function
            Must be of the form "y = fun(x,w)"
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix
        alpha : float
            Spread of the sigma points.
        beta : float
            Used to incorporate prior knowledge of the distribution
            2 is optimal is the state is normally distributed.
        kappa : float
            Secondary spread scaling parameter

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        : :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance
        """

        sigma_points, mean_weights, covar_weights = \
            gauss2sigma(x_pred, P_pred, alpha, beta, kappa)

        y_pred, S, Pxy, _, _, _ = unscented_transform(sigma_points,
                                                      mean_weights,
                                                      covar_weights,
                                                      h, covar_noise=R)

        return y_pred, S, Pxy

    @staticmethod
    def update_on_measurement_prediction(x_pred, P_pred, y,
                                         y_pred, S, Pxy):
        """Low level Unscented Kalman Filter update, based on a measurement\
        prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector
        y_pred: :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        S: :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        return KalmanUpdater.update_on_measurement_prediction(x_pred, P_pred,
                                                              y, y_pred, S,
                                                              Pxy)

    @staticmethod
    def _extract_model_parameters(measurement_model, measurement=None,
                                  **kwargs):
        """Extract measurement model parameters

        Parameters
        ----------
        measurement_model: :class:`~.MeasurementModel`
            A measurement model whose parameters are to be extracted
        measurement : :class:`~.Detection`, optional
            If provided and `measurement.measurement_model` is not `None`,\
            then its parameters will be returned instead\
            (the default is `None`, in which case `self.measurement_model`'s\
            parameters will be returned)

        Returns
        -------
        : function handle
            The (non-linear) measurement model function
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model covariance matrix
        """

        if(measurement is not None
           and measurement.measurement_model is not None):
            return UnscentedKalmanUpdater._extract_model_parameters(
                measurement.measurement_model)
        else:
            def measurement_function(x, w=0):
                return measurement_model.function(x, w, **kwargs)

            measurement_noise_covar = measurement_model.covar(**kwargs)

        return measurement_function, measurement_noise_covar

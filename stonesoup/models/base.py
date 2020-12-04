from abc import abstractmethod
from typing import TYPE_CHECKING, Union, Optional
from collections import namedtuple
from datetime import timedelta
import numpy as np
from scipy.stats import multivariate_normal

import numpy as np
from scipy.stats import multivariate_normal

from ..base import Base
from ..functions import jacobian as compute_jac
from ..types.array import StateVector, StateVectors
from ..types.numeric import Probability


class Model(Base):
    """Model type

    Base/Abstract class for all models."""

    @property
    @abstractmethod
    def ndim(self):
        """Number of dimensions of model"""
        pass

    @abstractmethod
    def function(self, state_vector, noise=False):
        """ Model function"""
        pass

    @abstractmethod
    def rvs(self, num_samples=1):
        """Model noise/sample generation method"""
        pass

    @abstractmethod
    def pdf(self, state1, state2):
        """Model pdf/likelihood evaluator method"""
        pass


class LinearModel(Model):
    """LinearModel class

    Base/Abstract class for all linear models"""

    @abstractmethod
    def matrix(self, **kwargs) -> np.ndarray:
        """Model matrix"""
        raise NotImplementedError

    def function(self, state_vector, noise=False, **kwargs):
        """Model linear function :math:`f_k(x(k),w(k)) = F_k(x_k) + w_k`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        : :class:`State`
            The updated State with the model function evaluated.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(**kwargs)
            else:
                noise = 0

        return self.matrix(**kwargs) @ state.state_vector + noise


class ReversibleModel(Model):
    """Non-linear model containing sufficient co-ordinate
    information such that the linear co-ordinate conversions
    can be calculated from the non-linear counterparts.

    Contains an inverse function which computes the reverse
    of the relevant linear-to-non-linear function"""

    def jacobian(self, state, **kwargs):
        """Model jacobian matrix :math:`H_{jac}`

        Parameters
        ----------
        state : :class:`~.State`
            An input state

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, \
        :py:attr:`~ndim_state`)
            The model jacobian matrix evaluated around the given state vector.
        """

        def fun(x):
            return self.function(x, noise=False)

        return compute_jac(fun, state)

    @abstractmethod
    def function(self, state_vector, noise=False, **kwargs):
        """Model function :math:`f(t,x(t),w(t))`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        : :class:`numpy.ndarray`
            The model function evaluated.
        """
        pass


class ReversibleModel(NonLinearModel):
    """Non-linear model containing sufficient co-ordinate
    information such that the linear co-ordinate conversions
    can be calculated from the non-linear counterparts.

    Contains an inverse function which computes the reverse
    of the relevant linear-to-non-linear function"""

    @abstractmethod
    def inverse_function(self, detection, **kwargs):
        """Takes in the result of the function and
        computes the inverse function, returning the initial
        input of the function.

        Parameters
        ----------
        detection: :class:`~.Detection`
            Input state (non-linear format)

        Returns
        -------
        : :class:`numpy.ndarray`
            The linear co-ordinates
        """
        pass


class TimeVariantModel(Model):
    """TimeVariantModel class

    Base/Abstract class for all time-variant models"""


class TimeInvariantModel(Model):
    """TimeInvariantModel class

    Base/Abstract class for all time-invariant models"""


class GaussianModel(Model):
    """GaussianModel class

    Base/Abstract class for all Gaussian models"""
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(self.seed) if self.seed is not None else None

    def rvs(self, num_samples: int = 1, random_state=None, **kwargs) ->\
            Union[StateVector, StateVectors]:
        r"""Model noise/sample generation function

        Generates noise samples from the model.

        In mathematical terms, this can be written as:

        .. math::

            v_t \sim \mathcal{N}(0,Q)

        where :math:`v_t =` ``noise`` and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        covar = self.covar(**kwargs)

        # If model has None-type covariance or contains None, it does not represent a Gaussian
        if covar is None or None in covar:
            raise ValueError("Cannot generate rvs from None-type covariance")

        random_state = random_state if random_state is not None else self.random_state

        noise = multivariate_normal.rvs(
            np.zeros(self.ndim), covar, num_samples, random_state=random_state)

        noise = np.atleast_2d(noise)

        if self.ndim > 1:
            noise = noise.T  # numpy.rvs method squeezes 1-dimensional matrices to integers

        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)

    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        In mathematical terms, this can be written as:

        .. math::

            p = p(y_t | x_t) = \mathcal{N}(y_t; x_t, Q)

        where :math:`y_t` = ``state_vector1``, :math:`x_t` = ``state_vector2``
        and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        state1 : State
        state2 : State

        Returns
        -------
        : :class:`~.Probability` or :class:`~.numpy.ndarray` of :class:`~.Probability`
            The likelihood of ``state1``, given ``state2``
        """
        return Probability.from_log_ufunc(self.logpdf(state1, state2, **kwargs))

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[float, np.ndarray]:
        r"""Model log pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        In mathematical terms, this can be written as:

        .. math::

            p = p(y_t | x_t) = \mathcal{N}(y_t; x_t, Q)

        where :math:`y_t` = ``state_vector1``, :math:`x_t` = ``state_vector2``
        and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        state1 : State
        state2 : State

        Returns
        -------
        :  float or :class:`~.numpy.ndarray`
            The log likelihood of ``state1``, given ``state2``
        """
        covar = self.covar(**kwargs)

        # If model has None-type covariance or contains None, it does not represent a Gaussian
        if covar is None or None in covar:
            raise ValueError("Cannot generate pdf from None-type covariance")

        # Calculate difference before to handle custom types (mean defaults to zero)
        # This is required as log pdf coverts arrays to floats
        likelihood = np.atleast_1d(
            multivariate_normal.logpdf((state1.state_vector - self.function(state2, **kwargs)).T,
                                       cov=covar))

        if len(likelihood) == 1:
            likelihood = likelihood[0]

        return likelihood

    def rvs(self, num_samples=1, **kwargs):
        r"""Model noise/sample generation function

        Generates noise samples from the model.

        In mathematical terms, this can be written as:

        .. math::

            v_t \sim \mathcal{N}(0,Q)

        where :math:`v_t =` ``noise`` and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        covar = self.covar(**kwargs)

        # If model has None-type covariance or contains None, it does not represent a Gaussian
        if covar is None or None in covar:
            raise ValueError("Cannot generate rvs from None-type covariance")

        noise = multivariate_normal.rvs(
            np.zeros(self.ndim), covar, num_samples)

        noise = np.atleast_2d(noise)

        if self.ndim > 1:
            noise = noise.T  # numpy.rvs method squeezes 1-dimensional matrices to integers

        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)

    def pdf(self, state1, state2, **kwargs):
        r"""Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        In mathematical terms, this can be written as:

        .. math::

            p = p(y_t | x_t) = \mathcal{N}(y_t; x_t, Q)

        where :math:`y_t` = ``state_vector1``, :math:`x_t` = ``state_vector2``
        and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        : :class:`~.Probability`
            The likelihood of ``state1``, given ``state2``
        """

        covar = self.covar(**kwargs)

        # If model has None-type covariance or contains None, it does not represent a Gaussian
        if covar is None or None in covar:
            raise ValueError("Cannot generate pdf from None-type covariance")

        # Calculate difference before to handle custom types (mean defaults to zero)
        # This is required as log pdf coverts arrays to floats
        likelihood = multivariate_normal.logpdf(
            (state1.state_vector - self.function(state2, **kwargs)).ravel(),
            cov=covar
        )
        return Probability(likelihood, log_value=True)

    @abstractmethod
    def covar(self):
        """Model covariance"""

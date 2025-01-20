from abc import abstractmethod
import copy
from typing import Union, Optional
import numpy as np
from stonesoup.base import Property
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State, GaussianState
from typing import Callable, Generator, Optional, Tuple, Union, List, Iterable, Sequence
from stonesoup.models.driver import GaussianDriver, LevyDriver
from stonesoup.types.array import StateVector, StateVectors, CovarianceMatrix, CovarianceMatrices
from stonesoup.models.base import Model, Latents, GaussianModel
from stonesoup.models.base_driver import NoiseCase
from stonesoup.models.transition.base import TransitionModel
from datetime import timedelta
from scipy.integrate import quad_vec
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag



class ConditionallyGaussianDriver(LevyDriver):
    """Conditional Gaussian Levy noise driver.

    Noise samples are generated according to the Levy State-Space Model by Godsill et al.
    """

    c: np.double = Property(doc="Truncation parameter, expected no. jumps per unit time.")
    mu_W: float = Property(default=0.0, doc="Default conditional Gaussian mean")
    sigma_W2: float = Property(default=1.0, doc="Default conditional Gaussian variance")
    noise_case: NoiseCase = Property(
        default=NoiseCase.GAUSSIAN_APPROX,
        doc="Cases for compensating residuals from series truncation",
    )
    mu_W_transition_model: Optional[Callable] = Property(
        default=None, doc="Optional transition model for mu_W"
    )
    mu_W_array: Optional[np.ndarray] = None  # Cache the computed mu_W_array


    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        """Thinning probabilities for accept-reject sampling of latent variables."""
        # Accept all
        return np.ones_like(jsizes)  # (n_jumps, n_samples)

    def _accept_reject(
        self, jsizes: np.ndarray, random_state: Optional[Generator]
    ) -> np.ndarray:
        """Accept reject sampling to thin out sampled latents (jumps).

        Args:
            jsizes (np.ndarray): Jump sizes to apply accept-reject sampling
            random_state (Generator, optional): Random state to use. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        probabilities = self._thinning_probabilities(jsizes)
        u = random_state.uniform(low=0.0, high=1.0, size=probabilities.shape)
        jsizes = np.where(u <= probabilities, jsizes, 0)
        return jsizes

    def sample_latents(
        self, dt: float, num_samples: int, random_state: Optional[Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Samples the non-linear and possible non-Gaussian latent variables.

        Args:
            dt (float): _description_
            num_samples (int): Number of different jump sequences to sample. Each jump sequence
                               consist of a multiple jumps where the number of jumps depends
                               on the truncation parameter `self.c`
            random_state (Optional[Generator], optional): Random state to use. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A Tuple consisting of the jump sizes and jump times.
        """
        if random_state is None:
            random_state = self.random_state
        # Sample latents pairs
        # num_samples = 1 # TODO: constraned num_samples to 1 always.
        epochs = random_state.exponential(
            scale=1 / dt, size=(int(self.c * dt), num_samples)
        )
        epochs = epochs.cumsum(axis=0)
        # Accept reject sampling
        jsizes = self._hfunc(epochs=epochs)
        jsizes = self._accept_reject(jsizes=jsizes, random_state=random_state)
        # Generate jump times
        jtimes = random_state.uniform(low=0.0, high=dt, size=jsizes.shape)
        return jsizes, jtimes

    @abstractmethod
    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        """H function to be used an direct or indirect evaluation of the inverse upper tail
        probability of the Levy density. For indirect approaches, accept reject sampling is
        as an additional step is needed.
        """

    @abstractmethod
    def _centering(self, e_ft: np.ndarray, truncation: float) -> StateVector:
        """Compensation term for skewed Levy density.

        Args:
            e_ft (np.ndarray): Expectation of Levy stochastic integral over a unit time axis.
            truncation (float): Truncation parameter or no. expected
                                Possion jumps per unit time.

        Returns:
            StateVector: Vectorised form of compensation term.
        """

    @abstractmethod
    def _jump_power(self, jszies: np.ndarray) -> np.ndarray:
        """Raises the latent jump sizes to the desired power .

        Args:
            jszies (np.ndarray): Latent jump sizes to raise.

        Returns:
            np.ndarray: Latent jump sizes raised to the desired power.
        """

    @abstractmethod
    def _first_moment(self, truncation: float) -> float:
        """Computes first moment of the underlying subordinator process up to
        an upper limit defined by h(c), whereby h is the H function and c represents
        the truncation parameter.

        Args:
            truncation (float): Truncation parameter which defines the upper limit
                                of the associated integral.


        Returns:
            float: First moment of subordinator process up to limit h(c).
        """

    @abstractmethod
    def _second_moment(self, truncation: float) -> float:
        """Computes second moment of the underlying subordinator process up to
        an upper limit defined by h(c), whereby h is the H function and c represents
        the truncation parameter.

        Args:
            truncation (float): Truncation parameter which defines the upper limit
                                of the associated integral.


        Returns:
            float: Second moment of subordinator process up to limit h(c).
        """

    @abstractmethod
    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float
    ) -> CovarianceMatrix:
        """Calculates the covariance of Gaussian approximate residuals. Residuals
        arises from the truncated series representation of the Levy stotchastic
        integral.

        Args:
            e_ft (np.ndarray): Levy stochastic integral over a unit time axia.
            truncation (float): Truncation parameter.
            mu_W (float): Gaussian mean of Levy density when conditioned over the
                          latent variables (jumps).
            sigma_W2 (float): Gaussian variance of Levy density when conditioned
                              over the latent variables (jumps).

        Returns:
            CovarianceMatrix: Covariance matrix of the Gaussian approximated
                              residuals.
        """

    def _residual_mean(
        self, e_ft: np.ndarray, truncation: float, mu_W: float
    ) -> StateVector:
        """Calculates the mean of Gaussian approximate residuals. Residuals
        arises from the truncated series representation of the Levy stotchastic
        integral.

        Args:
            e_ft (np.ndarray): Levy stochastic integral over a unit time axia.
            truncation (float): Truncation parameter.
            mu_W (float): Gaussian mean of Levy density when conditioned over the
                          latent variables (jumps).

        Returns:
            StateVector: Mean vector of the Gaussian approximated residuals.
        """
        if self.noise_case == NoiseCase.TRUNCATED or self.mu_W_transition_model is not None: #automatically truncate if time-varying
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
        elif (
            self.noise_case == NoiseCase.GAUSSIAN_APPROX
            or self.noise_case == NoiseCase.PARTIAL_GAUSSIAN_APPROX
        ):
            r_mean = e_ft * mu_W  # (m, 1)
        else:
            raise AttributeError("invalid noise case")
        return self._first_moment(truncation=truncation) * r_mean  # (m, 1)
    
    def _mu_W_array(self, jtimes: np.ndarray, dt: float, mu_W: Optional[float] = None, **kwargs) -> Tuple[np.ndarray, float]: 
        """
        Computes the time-varying mu_W based on the transition model or returns the constant mu_W.

        Args:
            jtimes (np.ndarray): Array of jump times.
            dt (float): The total time interval.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the array of mu_W values and the final mu_W value.
        """
        if self.mu_W_transition_model is None:
            return self.mu_W, np.full_like(jtimes, self.mu_W)

        n_mu_dim=self.mu_W_transition_model.ndim
        covar=np.zeros((n_mu_dim,n_mu_dim))

        mu_W_array = np.zeros_like(jtimes)
        sorted_jtimes = np.sort(jtimes, axis=0)        
        # print(sorted_jtimes, sorted_jtimes.shape)

        num_samples = jtimes.shape[1]
        num_jumps=jtimes.shape[0]

        mu_W = np.atleast_2d(mu_W if mu_W is not None else self.mu_W)
        padded_mu_W=np.full(max(num_samples,2), mu_W)

        for j in range(num_samples):
            initial_mu_W = GaussianState(state_vector=np.array([[padded_mu_W[j]]]), covar=covar)
            print(initial_mu_W)
            prev_mu_W = initial_mu_W
            for i in range(num_jumps):
                if i == 0:
                    # The first jump time interval is just jtime - 0
                    interval = timedelta(seconds=sorted_jtimes[i][j])  # Convert to timedelta
                else:
                    interval = timedelta(seconds=sorted_jtimes[i][j] - sorted_jtimes[i - 1][j])  # Convert difference to timedelta

                prev_mu_W_state_vector = self.mu_W_transition_model.function(prev_mu_W, noise=True, time_interval=interval) 
                mu_W_array[i][j] = prev_mu_W_state_vector[0] #add the float of the state vector value to the array
                prev_mu_W =  GaussianState(state_vector=prev_mu_W_state_vector, covar=covar) #transform float to gauss state for next update
                #TODO: use some form of prediction model to add mu_W's covariance (even though not used)
        
            # For the last interval, update with (dt - jtimes[-1])
            final_interval = timedelta(seconds=dt - sorted_jtimes[-1][j]) 
            prev_mu_W_state_vector = self.mu_W_transition_model.function(prev_mu_W, noise=True, time_interval=final_interval) 
            last_mu_W =  prev_mu_W_state_vector[0] #add the float to the output
            padded_mu_W[j]=last_mu_W
        if num_samples>1:
            last_mu_W=padded_mu_W
        print(last_mu_W)
        return last_mu_W, mu_W_array
    
    def mean(
        self,
        jsizes: np.array,
        jtimes: np.array,
        ft_func: Callable[..., np.ndarray],
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        mu_W: Optional[float] = None,
        mu_W_array: Optional[np.ndarray] = None,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        """Computes mean vectors. The number of mean vectors is dependent on the
        number of samples in the jump sizes/times. Each jump sequence results in
        an unique mean vector.

        Args:
            jsizes (np.array): Latents corresponding to jump sizes.
            jtimes (np.array): Latents corresponding to jump times.
            ft_func (Callable[..., np.ndarray]): The function f consisting of the
                state transtion matrix multiplied by the control matrix h as denoted
                by Godsill et. al. (2020).
            e_ft_func (Callable[..., np.ndarray]): The expectation of ft_func.
            dt (float): The time interval.
            mu_W (Optional[float], optional): The conditionally Gaussian mean vector.
                Defaults to None and the default mu_W specified during initialisation
                is used.

        Returns:
            Union[StateVector, StateVectors]: The resulting mean vectors.
        """
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)
        assert jsizes.shape == jtimes.shape
        num_samples = jsizes.shape[1]
        truncation = self.c * dt
        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        if self.mu_W_transition_model is None:
            series = np.sum(jsizes[..., None, None] * ft, axis=0)  # (n_samples, m, 1)
            m = series * mu_W
        else:
            m = np.sum(jsizes[..., None, None] * ft * mu_W_array[..., None, None], axis=0)  # (n_samples, m, 1) sum over varying mean terms directly

        e_ft = e_ft_func(dt=dt)  # (m, 1)
        residual_mean = self._residual_mean(e_ft=e_ft, mu_W=mu_W, truncation=truncation)[
            None, ...
        ]
        centering = (
            dt * self._centering(e_ft=e_ft, mu_W=mu_W, truncation=truncation)[None, ...]
        )
        mean = m - centering + residual_mean
        if num_samples == 1:
            return mean[0].view(StateVector)
        else:
            return mean.view(StateVectors)

    def covar(
        self,
        jsizes: np.array,
        jtimes: np.array,
        ft_func: Callable[..., np.ndarray],
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        mu_W: Optional[float] = None,
        mu_W_array: Optional[np.ndarray] = None,
        sigma_W2: Optional[float] = None,
        **kwargs
    ) -> Union[CovarianceMatrix, CovarianceMatrices]:
        """Computes covariance matrices. The number of covariance matrices is dependent
        on the number of samples in the jump sizes/times. Each jump sequence results
        in an unique covariance matrix.

        Args:
            jsizes (np.array): Latents corresponding to jump sizes.
            jtimes (np.array): Latents corresponding to jump times.
            ft_func (Callable[..., np.ndarray]): The function f consisting of the
                state transtion matrix multiplied by the control matrix h as denoted
                by Godstill et. al. (2020).
            e_ft_func (Callable[..., np.ndarray]): The expectation of ft_func.
            dt (float): The time interval.
            mu_W (Optional[float], optional): The conditionally Gaussian mean.
                Defaults to None and the default mu_W specified during initialisation
                is used.
            sigma_W2 (Optional[float], optional): The conditionally Gaussian variance.
                Defaults to None and the default sigma_W2 specified during initialisation
                is used.

        Returns:
            Union[CovarianceMatrix, CovarianceMatrices]: The resulting covariance matrices.
        """
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)
        sigma_W2 = (
            np.atleast_2d(self.sigma_W2) if sigma_W2 is None else np.atleast_2d(sigma_W2)
        )

        assert jsizes.shape == jtimes.shape
        num_samples = jsizes.shape[1]
        jsizes = self._jump_power(jsizes)  # (n_jumps, n_samples)
        truncation = self._hfunc(self.c * dt)

        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        ft2 = np.einsum("ijkl, ijml -> ijkm", ft, ft)  # (n_jumps, n_samples, m, m)
        series = np.sum(jsizes[..., None, None] * ft2, axis=0)  # (n_samples, m, m)
        s = sigma_W2 * series

        e_ft = e_ft_func(dt=dt)  # (m, 1)
        residual_cov = self._residual_covar(
            e_ft=e_ft, mu_W=mu_W, sigma_W2=sigma_W2, truncation=truncation
        )
        covar = s + residual_cov
        if num_samples == 1:
            return covar[0].view(CovarianceMatrix)  # (m, m)
        else:
            return covar.view(CovarianceMatrices)  # (n_samples, m, m)

    def rvs(
        self,
        mean: StateVector,
        covar: CovarianceMatrices,
        random_state: Optional[np.random.RandomState] = None,
        num_samples: int = 1,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        """Computes the driving noise term given the mean and covariance matrix specified.


        Args:
            mean (StateVector): The mean vector.
            covar (CovarianceMatrices): The covariance matrix.
            random_state (Optional[np.random.RandomState], optional): RNG to use. Defaults to None.
            num_samples (int, optional): Number of driving noise samples. Defaults to 1.

        Returns:
            Union[StateVector, StateVectors]: Driving noise samples.
        """
        assert isinstance(mean, StateVector)
        assert isinstance(covar, CovarianceMatrix)
        if random_state is None:
            random_state = self.random_state

        noise = random_state.multivariate_normal(mean.flatten(), covar, size=num_samples)
        noise = noise.T
        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)


class NormalSigmaMeanDriver(ConditionallyGaussianDriver):
    """Implements the class of Normal Sigma Mean (NSM) Levy models."""

    def _jump_power(self, jsizes: np.ndarray) -> np.ndarray:
        return jsizes**2

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float, **kwargs
    ) -> CovarianceMatrix:
        mu_W = mu_W
        sigma_W2 = sigma_W2
        if self.noise_case == NoiseCase.TRUNCATED or self.mu_W_transition_model is not None: #automatically truncate if mu_W time-varying. given 2nd case formula, cld implement easily
            m = e_ft.shape[0]
            r_cov = np.zeros((m, m))
        elif self.noise_case == NoiseCase.GAUSSIAN_APPROX:
            r_cov = (
                e_ft
                @ e_ft.T
                * self._second_moment(truncation=truncation)
                * (mu_W**2 + sigma_W2)
            )
        elif self.noise_case == NoiseCase.PARTIAL_GAUSSIAN_APPROX:
            r_cov = e_ft @ e_ft.T * self._second_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("Invalid noise case.")
        return r_cov  # (m, m)
    
class AlphaStableNSMDriver(NormalSigmaMeanDriver):
    """Implements the Alpha Stable NSM noise driver to be used with :class:`~.LevyModel`."""

    alpha: float = Property(doc="Alpha parameter.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            np.isclose(self.alpha, 0.0)
            or np.isclose(self.alpha, 1.0)
            or np.isclose(self.alpha, 2.0)
        ):
            raise AttributeError("alpha must be 0 < alpha < 1 or 1 < alpha < 2.")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return np.power(epochs, -1.0 / self.alpha)

    def _first_moment(self, **kwargs) -> float:
        return self.alpha / (1.0 - self.alpha) * np.power(self.c, 1.0 - 1.0 / self.alpha)

    def _second_moment(self, **kwargs) -> float:
        return self.alpha / (2.0 - self.alpha) * np.power(self.c, 1.0 - 2.0 / self.alpha)

    def _residual_mean(
        self, e_ft: np.ndarray, truncation: float, mu_W: float
    ) -> StateVector:
        if 1 < self.alpha < 2:
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
            return r_mean
        return super()._residual_mean(e_ft=e_ft, mu_W=mu_W, truncation=truncation)

    def _centering(self, e_ft: np.ndarray, truncation: float, mu_W: float) -> StateVector:
        if 0 < self.alpha < 1 or self.mu_W_transition_model is not None: #paper doesn't cover centering terms so set to zero if time-varying
            m = e_ft.shape[0]
            return np.zeros((m, 1))
        elif 1 < self.alpha < 2:
            term = e_ft * mu_W  # (m, 1) 
            return -self._first_moment(truncation=truncation) * term  # (m, 1) [should be implementable as dt/T *E[mu_W(Vi)*jsizes[i]]*e_ft for time-varying mu_W ]
        else:
            raise AttributeError("alpha must be 0 < alpha < 2")

    def characteristic_func(self):
        # TODO
        raise NotImplementedError

class LevyModel(Model):
    """
    Class to be derived from for Levy models.
    For now, we consider only conditionally Gaussian ones
    """

    driver: Union[ConditionallyGaussianDriver, GaussianDriver] = Property(
        doc="Conditional Gaussian process noise driver"
    )
    mu_W: Optional[float] = Property(default=None, doc="Condtional Gaussian mean")
    sigma_W2: Optional[float] = Property(default=None, doc="Conditional Gaussian variance")
    mu_W_transition_model: Optional[Callable] = Property(
        default=None, doc="Optional transition model for mu_W"
    )
    mu_W_array: Optional[np.ndarray] = None  # Cache the computed mu_W_array

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _integrand(self, dt: float, jtimes: np.ndarray) -> np.ndarray:
        pass

    def _integrate(self, func: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res, err = quad_vec(func, a=a, b=b)
        return res

    def _integral(self, dt: float) -> np.ndarray:
        def func(dt: int):
            return self._integrand(dt, jtimes=np.zeros((1, 1)))[0, 0, :]  # currying
        return self._integrate(func, a=0, b=dt)
    
    def _mu_W_array(self,latents:Latents, time_interval:timedelta, mu_W: Optional[float] = None,**kwargs) -> np.ndarray:
        """Model covariance"""
        assert latents is not None
        dt = time_interval.total_seconds()
        if latents.exists(self.driver):
            jsizes = latents.sizes(self.driver)
            jtimes = latents.times(self.driver)
        else:
            jsizes, jtimes = None, None
        return self.driver._mu_W_array(         #returns mu_W_array, mu_prev where mu_prev is mu_value for most recent timestep
            jtimes=jtimes,
            dt=dt,
            num_samples=latents.num_samples,
            mu_W=mu_W
        )
    
    def mean(
        self, latents: Latents, time_interval: timedelta, **kwargs
    ) -> Union[StateVector, StateVectors]:
        """Model mean"""
        assert latents is not None
        dt = time_interval.total_seconds()
        if latents.exists(self.driver):
            jsizes = latents.sizes(self.driver)
            jtimes = latents.times(self.driver)
        else:
            jsizes, jtimes = None, None    
        if self.driver.mu_W_transition_model is not None:
            self.mu_W, self.mu_W_array=self._mu_W_array(latents=latents,
                                                        time_interval=time_interval,
                                                        mu_W=self.mu_W,)
        return self.driver.mean(
            jsizes=jsizes,
            jtimes=jtimes,
            dt=dt,
            e_ft_func=self._integral,
            ft_func=self._integrand,
            mu_W=self.mu_W,
            mu_W_array=self.mu_W_array,
            num_samples=latents.num_samples,
        )

    def covar(
        self, latents: Latents, time_interval: timedelta, **kwargs
    ) -> Union[CovarianceMatrix, CovarianceMatrices]:
        """Model covariance"""
        assert latents is not None
        dt = time_interval.total_seconds()
        if latents.exists(self.driver):
            jsizes = latents.sizes(self.driver)
            jtimes = latents.times(self.driver)
        else:
            jsizes, jtimes = None, None
        if self.driver.mu_W_transition_model is not None:
            self.mu_W, self.mu_W_array=self._mu_W_array(latents=latents,
                                                        time_interval=time_interval,
                                                        mu_W=self.mu_W,)
        return self.driver.covar(
            jsizes=jsizes,
            jtimes=jtimes,
            dt=dt,
            e_ft_func=self._integral,
            ft_func=self._integrand,
            mu_W=self.mu_W,
            mu_W_array=self.mu_W_array,
            sigma_W2=self.sigma_W2,
            num_samples=latents.num_samples,
        )

    def sample_latents(
        self,
        time_interval: timedelta,
        num_samples: int,
        random_state: Optional[np.random.RandomState] = None,
    ) -> Latents:
        dt = time_interval.total_seconds()
        latents = Latents(num_samples=num_samples)
        if isinstance(self.driver, ConditionallyGaussianDriver):
            jsizes, jtimes = self.driver.sample_latents(
                dt=dt, num_samples=num_samples, random_state=random_state
            )
            latents.add(driver=self.driver, jsizes=jsizes, jtimes=jtimes)
        return latents

    def rvs(
        self,
        latents: Optional[Latents] = None,
        n_rvs_samples_for_each_mean_covar_pair: int = 1,
        random_state: Optional[np.random.RandomState] = None,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        noise = 0
        n_mean_covar_pair = 1
        if not latents:
            latents = self.sample_latents(
                num_samples=n_mean_covar_pair, random_state=random_state, **kwargs
            )
        mean = self.mean(latents=latents, **kwargs)
        if mean is None or None in mean:
            raise ValueError("Cannot generate rvs from None-type mean")
        assert isinstance(mean, StateVector)

        covar = self.covar(latents=latents, **kwargs)
        if covar is None or None in covar:
            raise ValueError("Cannot generate rvs from None-type covariance")
        assert isinstance(covar, CovarianceMatrix)

        noise += self.driver.rvs(
            mean=mean,
            covar=covar,
            random_state=random_state,
            num_samples=n_rvs_samples_for_each_mean_covar_pair,
            **kwargs
        )
        return noise

    def condpdf(
        self, state1: State, state2: State, latents: Optional[Latents] = None, **kwargs
    ) -> Union[Probability, np.ndarray]:
        r"""Model conditional pdf/likelihood evaluation function"""
        return Probability.from_log_ufunc(
            self.logcondpdf(state1, state2, latents=latents, **kwargs)
        )

    def logcondpdf(
        self, state1: State, state2: State, latents: Optional[Latents] = None, **kwargs
    ) -> Union[float, np.ndarray]:
        r"""Model log conditional pdf/likelihood evaluation function"""
        if latents is None:
            raise ValueError("Latents cannot be none.")

        mean = self.mean(latents=latents, **kwargs)
        if mean is None or None in mean:
            raise ValueError("Cannot generate pdf from None-type mean")
        assert isinstance(mean, StateVector)

        covar = self.covar(latents=latents, **kwargs)
        if covar is None or None in covar:
            raise ValueError("Cannot generate pdf from None-type covariance")
        assert isinstance(covar, CovarianceMatrix)

        likelihood = np.atleast_1d(
            multivariate_normal.logpdf(
                (state1.state_vector - self.function(state2, **kwargs)).T, mean=mean, cov=covar
            )
        )

        if len(likelihood) == 1:
            likelihood = likelihood[0]

        return likelihood

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Model log pdf/likelihood evaluation function"""
        return NotImplementedError

    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Model pdf/likelihood evaluation function"""
        return Probability.from_log_ufunc(self.logpdf(state1, state2, **kwargs))
    
class CombinedLevyTransitionModel(TransitionModel, LevyModel):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Gaussian.
    Time Variant, and Time Invariant models can be combined together.
    If any of the models are time variant the keyword argument "time_interval"
    must be supplied to all methods
    """
    model_list: Sequence[GaussianModel] = Property(doc="List of Transition Models.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.model_list) != 0

    def _integrand(self, dt: float, jtimes: np.ndarray):
        return NotImplementedError

    @property
    def driver(self) -> List[Iterable]:
        return [model.driver for model in self.model_list]

    @property
    def mu_W(self):
        mu = [m.mu_W if m.mu_W is not None else m.driver.mu_W for m in self.model_list]
        return np.atleast_2d(mu).T

    @property
    def mu_W_transition_model(self): 
        mu_W_transition_model = [m.mu_W_transition_model if m.mu_W_transition_model is not None
                                  else m.driver.mu_W_transition_model for m in self.model_list]
        return mu_W_transition_model
    
    @property
    def mu_W_array(self):
        mu_W_array = [m.mu_W_array if m.mu_W_array is not None
                                  else m.driver.mu_W_array for m in self.model_list]
        return mu_W_array
        
    @property
    def sigma_W2(self):
        sigma2 = [
            m.sigma_W2 if m.sigma_W2 is not None else m.driver.sigma_W2
            for m in self.model_list
        ]
        return np.diag(sigma2)

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list)

    def mean(self, **kwargs) -> Union[StateVector, StateVectors]:
        """Returns the transition model noise mean vector.

        Returns
        -------
        : :class:`stonesoup.types.state.StateVector` of shape\
        (:py:attr:`~ndim_state`, 1)
            The process noise mean.
        """
        mean_list = [model.mean(**kwargs) for _, model in enumerate(self.model_list)]
        if len(mean_list[0].shape) == 2:
            return np.vstack(mean_list).view(StateVector)
        else:
            return np.concatenate(mean_list, axis=1).view(StateVectors)
    def covar(self, **kwargs) -> Union[CovarianceMatrix, CovarianceMatrices]:
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        covar_list = [model.covar(**kwargs) for _, model in enumerate(self.model_list)]
        if len(covar_list[0].shape) == 2:
            return block_diag(*covar_list).view(CovarianceMatrix)
        else:
            N = covar_list[0].shape[0]
            ret = []
            for n in range(N):
                tmp = []
                for tensor in covar_list:  # D
                    tmp.append(tensor[n])
                ret.append(block_diag(*tmp))
            return np.array(ret).view(CovarianceMatrices)

    def function(
        self, state, time_interval: timedelta, noise=False, **kwargs
    ) -> StateVector:
        """Applies each transition model in :py:attr:`~model_list` in turn to the state's
        corresponding state vector components.
        For example, in a 3D state space, with :py:attr:`~model_list` = [modelA(ndim_state=2),
        modelB(ndim_state=1)], this would apply modelA to the state vector's 1st and 2nd elements,
        then modelB to the remaining 3rd element.

        Parameters
        ----------
        state : :class:`stonesoup.state.State`
            The state to be transitioned according to the models in :py:attr:`~model_list`.
        time_interval : :class:`timestamp.timedelta`
            The time interval between two observations.
        noise : :class:`bool`


        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_state, 1`). The resultant state vector of the transition.
        """

        temp_state = copy.copy(state)
        ndim_count = 0
        if state.state_vector.shape[1] == 1:
            state_vector = np.zeros(state.state_vector.shape).view(StateVector)
        else:
            state_vector = np.zeros(state.state_vector.shape).view(StateVectors)
        # To handle explicit noise vector(s) passed in we set the noise for the individual models
        # to False and add the noise later. When noise is Boolean, we just pass in that value.
        if noise is None:
            noise = False
        if isinstance(noise, bool):
            noise_loop = noise
        else:
            noise_loop = False
        latents = self.sample_latents(time_interval=time_interval, num_samples=1)
        for model in self.model_list:
            temp_state.state_vector = state.state_vector[
                ndim_count: model.ndim_state + ndim_count, :
            ]
            state_vector[ndim_count: model.ndim_state + ndim_count, :] += model.function(
                state=temp_state,
                latents=latents,
                time_interval=time_interval,
                noise=noise_loop,
                **kwargs
            )
            ndim_count += model.ndim_state

        if isinstance(noise, bool):
            noise = 0
        return state_vector + noise

    def sample_latents(
        self,
        time_interval: timedelta,
        num_samples: int,
        random_state: Optional[np.random.RandomState] = None,
    ) -> Latents:
        dt = time_interval.total_seconds()
        latents = Latents(num_samples=num_samples)
        for m in self.model_list:
            if (
                m.driver
                and isinstance(m.driver, ConditionallyGaussianDriver)
                and not latents.exists(m.driver)
            ):
                jsizes, jtimes = m.driver.sample_latents(
                    dt=dt, num_samples=num_samples, random_state=random_state
                )
                latents.add(driver=m.driver, jsizes=jsizes, jtimes=jtimes)
        return latents

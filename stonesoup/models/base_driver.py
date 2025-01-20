from datetime import timedelta
from typing import Union, Optional, Tuple, Callable, Dict, NamedTuple, Generator
from collections import namedtuple
from scipy.stats import multivariate_normal
from abc import abstractmethod
from enum import Enum
from typing import Callable, Generator, Optional, Tuple, Union
from datetime import timedelta
import numpy as np
from stonesoup.types.state import   GaussianState
from stonesoup.base import Base, Property
from stonesoup.types.array import (
    CovarianceMatrices,
    CovarianceMatrix,
    StateVector,
    StateVectors,
)


class NoiseCase(Enum):
    """Different methods of approximating the residuals for
    the truncated series representation of the associated
    Levy integrals
    """

    TRUNCATED = 0
    GAUSSIAN_APPROX = 1
    PARTIAL_GAUSSIAN_APPROX = 2


class Driver(Base):
    pass


class NoiseCase(Driver):
    pass


class TruncatedCase(NoiseCase):
    pass


class GaussianResidualApproxCase(NoiseCase):
    pass


class PartialGaussianResidualApproxCase(NoiseCase):
    pass


class Latents:
    def __init__(self, num_samples: int) -> None:
        self.store: Dict[Driver, NamedTuple] = dict()
        self.Data = namedtuple("Data", ["sizes", "times"])
        self._num_samples = num_samples

    def exists(self, driver) -> bool:
        return driver in self.store

    def add(self, driver: Driver, jsizes: np.ndarray, jtimes: np.ndarray) -> None:
        assert jsizes.shape == jtimes.shape
        assert jsizes.shape[1] == self._num_samples
        data = self.Data(jsizes, jtimes)
        self.store[driver] = data

    def sizes(self, driver) -> np.ndarray:
        assert driver in self.store
        # dimensions of sizes are (n_jumps, n_samples)
        return self.store[driver].sizes

    def times(self, driver) -> np.ndarray:
        assert driver in self.store
        # dimensions of times are (n_times, n_samples)
        return self.store[driver].times

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, num_samples) -> None:
        self._num_samples = num_samples


class LevyDriver(Driver):
    """Driver type

    Base/Abstract class for all conditional Gaussian noise driving processes."""

    seed: Optional[int] = Property(default=None, doc="Seed for random number generation")
    c: np.double = Property(doc="Truncation parameter, expected no. jumps per unit time.")
    noise_case: NoiseCase = Property(
        default=GaussianResidualApproxCase(),
        doc="Cases for compensating residuals from series truncation",
    )
    mu_W: float = Property(default=0.0, doc="Default Gaussian mean")
    sigma_W2: float = Property(default=1.0, doc="Default Gaussian variance")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_state = np.random.default_rng(self.seed)

    def rvs(
        self,
        mean: StateVector,
        covar: CovarianceMatrix,
        random_state: Generator,
        num_samples: int = 1,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        """
        returns driving noise term
        """
        if random_state is None:
            random_state = self.random_state
        noise = random_state.multivariate_normal(
            mean.flatten(), covar, size=num_samples
        )
        noise = noise.T
        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)

    @abstractmethod
    def characteristic_func():
        pass

    @abstractmethod
    def _centering(self, e_ft: np.ndarray, truncation: float) -> StateVector:
        pass

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


    @abstractmethod
    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        """Calculate thinning probabilities for accept-reject sampling"""
        pass

    @abstractmethod
    def _jump_power(self, jszies: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _first_moment(self, truncation: float) -> float:
        pass

    @abstractmethod
    def _second_moment(self, truncation: float) -> float:
        pass

    @abstractmethod
    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float
    ) -> CovarianceMatrix:
        pass

    def _residual_mean(self, e_ft: np.ndarray, truncation: float, mu_W: float) -> CovarianceMatrix:
        if isinstance(self.noise_case, TruncatedCase):
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
        elif isinstance(self.noise_case, GaussianResidualApproxCase) or isinstance(
            self.noise_case, PartialGaussianResidualApproxCase
        ):
            r_mean = e_ft * mu_W  # (m, 1)
        else:
            raise AttributeError("invalid noise case")
        return self._first_moment(truncation=truncation) * r_mean  # (m, 1)

    def _accept_reject(self, jsizes: np.ndarray, random_state: Generator) -> np.ndarray:
        probabilities = self._thinning_probabilities(jsizes)
        u = random_state.uniform(low=0.0, high=1.0, size=probabilities.shape)
        jsizes = np.where(u <= probabilities, jsizes, 0)
        return jsizes

    def sample_latents(self, dt: float, num_samples: int, random_state: Optional[Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        if random_state is None:
            random_state = self.random_state
        # Sample latents pairs
        epochs = random_state.exponential(scale=1 / dt, size=(int(self.c * dt), num_samples))
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
        self, e_ft: np.ndarray, truncation: float, mu_W: float, # TODO: add mu_W_array: np.array and method
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
        return last_mu_W, mu_W_array
    
    def mean(
        self,
        latents: Latents,
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

        jtimes = latents.times(driver=self)  # (n_jumps, n_samples)
        jsizes = latents.sizes(driver=self)  # (n_jumps, n_samples)
        num_samples = latents.num_samples
        assert(jsizes.shape[1] == (num_samples) and jtimes.shape[1] == (num_samples))
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
        latents: Latents,
        ft_func: Callable[..., np.ndarray],
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        mu_W: Optional[float] = None,
        mu_W_array: Optional[np.ndarray] = None,
        sigma_W2: Optional[float] = None,
        **kwargs
    ) -> Union[CovarianceMatrix, CovarianceMatrices]:
        """Computes covariance matrix / matrices"""
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)
        sigma_W2 = np.atleast_2d(self.sigma_W2) if sigma_W2 is None else np.atleast_2d(sigma_W2)     

        jsizes = self._jump_power(latents.sizes(driver=self))  # (n_jumps, n_samples)
        jtimes = latents.times(driver=self)
        num_samples = latents.num_samples
        assert(jsizes.shape[1] == (num_samples) and jtimes.shape[1] == (num_samples))

        truncation = self._hfunc(self.c * dt)

        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        ft2 = np.einsum("ijkl, ijml -> ijkm", ft, ft)  # (n_jumps, n_samples, m, m)
        series = np.sum(jsizes[..., None, None] * ft2, axis=0)  # (n_samples, m, m)
        s = sigma_W2 * series

        e_ft = e_ft_func(dt=dt)  # (m, 1)
        residual_cov = self._residual_covar(e_ft=e_ft, mu_W=mu_W, sigma_W2=sigma_W2, truncation=truncation)
        covar = s + residual_cov
        if num_samples == 1:
            return covar[0].view(CovarianceMatrix)  # (m, m)
        else:
            return covar.view(CovarianceMatrices)  # (n_samples, m, m)


class NormalSigmaMeanDriver(LevyDriver):
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
        elif isinstance(self.noise_case, GaussianResidualApproxCase):
            r_cov = (
                e_ft @ e_ft.T * self._second_moment(truncation=truncation) * (mu_W**2 + sigma_W2)
            )
        elif isinstance(self.noise_case, PartialGaussianResidualApproxCase):
            r_cov = e_ft @ e_ft.T * self._second_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("Invalid noise case.")
        return r_cov  # (m, m)
    
    def characteristic_func(self):
        # TODO
        raise NotImplementedError


class NormalVarianceMeanDriver(LevyDriver):
    def _jump_power(self, jsizes: np.ndarray) -> np.ndarray:
        return jsizes

    def _centering(
        self, e_ft: np.ndarray, truncation: float, mu_W: float
    ) -> StateVector:
        m = e_ft.shape[0]
        return np.zeros((m, 1))

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float, **kwargs
    ) -> CovarianceMatrix:
        mu_W = mu_W
        sigma_W2 = sigma_W2
        if isinstance(self.noise_case, TruncatedCase):
            m = e_ft.shape[0]
            r_cov = np.zeros((m, m))
        elif isinstance(self.noise_case, GaussianResidualApproxCase):
            r_cov = (
                e_ft
                @ e_ft.T
                * (
                    self._second_moment(truncation=truncation) * mu_W**2
                    + self._first_moment(truncation=truncation) * sigma_W2
                )
            )
        elif isinstance(self.noise_case, PartialGaussianResidualApproxCase):
            r_cov = e_ft @ e_ft.T * self._first_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("Invalid noise case.")
        return r_cov  # (m, m)

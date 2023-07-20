import copy
import numpy as np
from scipy.stats import multivariate_normal, uniform

from .base import Regulariser
from ..functions import cholesky_eps
from ..types.state import ParticleState


class MCMCRegulariser(Regulariser):
    """Markov chain Monte-Carlo (MCMC) move steps, or regularisation steps, can be implemented in
    particle filters to prevent sample impoverishment that results from resampling.
    One way of avoiding this is to only perform resampling when deemed necessary by some measure
    of effectiveness. Sometimes this is not desirable, or possible, when a particular algorithm
    requires the introduction of new samples as part of the filtering process for example.

    This is a particular implementation of a MCMC move step that uses the Metropolis-Hastings
    algorithm [1]_. After resampling, particles are moved a small amount, according do a Gaussian
    kernel, to a new state only if the Metropolis-Hastings acceptance probability is met by a
    random number assigned to each particle from a uniform random distribution, otherwise they
    remain the same. Further details on the implementation are given in [2]_.

    References
    ----------
    .. [1] Robert, Christian P. & Casella, George, Monte Carlo Statistical Methods, Springer, 1999.

    .. [2] Ristic, Branko & Arulampalam, Sanjeev & Gordon, Neil, Beyond the Kalman Filter:
        Particle Filters for Target Tracking Applications, Artech House, 2004. """

    def regularise(self, prior, posterior, detections, transition_model=None):
        """Regularise the particles

        Parameters
        ----------
        prior : :class:`~.ParticleState` type or list of :class:`~.Particle`
            prior particle distribution.
        posterior : :class:`~.ParticleState` type or list of :class:`~.Particle`
            posterior particle distribution
        detections : set of :class:`~.Detection`
            set of detections containing clutter,
            true detections or both
        transition_model : :class:`~.TransitionModel`
            Transition model used in the prediction step.

        Returns
        -------
        particle state: :class:`~.ParticleState`
           The particle state after regularisation
        """

        if not isinstance(posterior, ParticleState):
            posterior = ParticleState(None, particle_list=posterior)

        if not isinstance(prior, ParticleState):
            prior = ParticleState(None, particle_list=prior)

        regularised_particles = copy.copy(posterior)
        moved_particles = copy.copy(posterior)
        transitioned_prior = copy.copy(prior)

        if transition_model is not None:
            time_interval = posterior.timestamp - prior.timestamp
            new_state_vector = transition_model.function(prior,
                                                         noise=False,
                                                         time_interval=time_interval)
            transitioned_prior.state_vector = new_state_vector

        if detections is not None:
            ndim = prior.state_vector.shape[0]
            nparticles = len(posterior)

            measurement_model = next(iter(detections)).measurement_model

            # calculate the optimal bandwidth for the Gaussian kernel
            hopt = (4/(ndim+2))**(1/(ndim+4))*nparticles**(-1/(ndim+4))
            covar_est = posterior.covar

            # move particles
            moved_particles.state_vector = moved_particles.state_vector + \
                hopt * cholesky_eps(covar_est) @ np.random.randn(ndim, nparticles)

            # Evaluate likelihoods
            part_diff = moved_particles.state_vector - transitioned_prior.state_vector
            move_likelihood = multivariate_normal.logpdf(part_diff.T,
                                                         cov=covar_est)
            post_part_diff = posterior.state_vector - transitioned_prior.state_vector
            post_likelihood = multivariate_normal.logpdf(post_part_diff.T,
                                                         cov=covar_est)

            # Evaluate measurement likelihoods
            move_meas_likelihood = []
            post_meas_likelihood = []
            for detection in detections:
                move_meas_likelihood.append(measurement_model.logpdf(detection, moved_particles))
                post_meas_likelihood.append(measurement_model.logpdf(detection, posterior))

            # In the case that there are multiple measurements,
            # we select the highest overall likelihood.
            max_likelihood_idx = np.argmax(np.sum(move_meas_likelihood, axis=1))

            # Calculate acceptance probability (alpha)
            alpha = np.exp((move_meas_likelihood[max_likelihood_idx] + move_likelihood) -
                           (post_meas_likelihood[max_likelihood_idx] + post_likelihood))

            # All 'jittered' particles that are above the alpha threshold are kept, the rest are
            # rejected and the original posterior used
            selector = uniform.rvs(size=nparticles)
            index = alpha > selector

            regularised_particles.state_vector[:, index] = moved_particles.state_vector[:, index]

        return regularised_particles

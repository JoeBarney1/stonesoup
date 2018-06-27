from scipy.stats import multivariate_normal

from .base import Initiator
from ..base import Property
from ..dataassociator import DataAssociator
from ..deleter import Deleter
from ..models.base import LinearModel, ReversibleModel
from ..models.measurement import MeasurementModel
from ..types.hypothesis import SingleHypothesis
from ..types.mixture import GaussianMixture
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.state import State, GaussianState, ParticleState, TaggedWeightedGaussianState, \
    ASDGaussianState, EnsembleState
from ..types.track import Track
from ..types.state import GaussianState, ParticleState
from ..types.particle import Particle


class SinglePointInitiator(GaussianInitiator):
    """SinglePointInitiator class

    This uses an :class:`~.ExtendedKalmanUpdater` to carry out an update using
    provided :attr:`prior_state` for each unassociated detection.
    """

    def initiate(self, unassociated_detections, **kwargs):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        unassociated_detections : list of \
        :class:`stonesoup.types.detection.Detection`
            A list of unassociated detections
        timestamp: datetime.datetime
            Current timestamp

        Returns
        -------
        : :class:`sets.Set` of :class:`stonesoup.types.track.Track`
            A list of new tracks with an initial :class:`~.GaussianState`
        """

        tracks = set()
        for detection in unassociated_detections:
            post_state_vec, post_state_covar, _ = \
                KalmanUpdater.update_lowlevel(self.prior_state.state_vector,
                                              self.prior_state.covar,
                                              self.measurement_model.matrix(),
                                              self.measurement_model.covar(),
                                              detection.state_vector)

        tracks = set()
        for detection in detections:
            measurement_prediction = updater.predict_measurement(
                self.prior_state, detection.measurement_model)
            track_state = updater.update(SingleHypothesis(
                self.prior_state, detection, measurement_prediction))
            track = Track([track_state])
            tracks.add(track)

        return tracks


class SinglePointParticleInitiator(SinglePointInitiator):
    """SinglePointParticleInitiator class

    Utilising the SinglePointInitiator, sample from the resultant track's state
    to generate a number of particles, overwriting with a
    :class:`~.ParticleState`.
    """

    number_particles = Property(
        float, default=200, doc="Number of particles for initial track state")

    def initiate(self, unassociated_detections, **kwargs):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        unassociated_detections : list of :class:`~.Detection`
            A list of unassociated detections

        Returns
        -------
        : set of :class:`~.Track`
            A list of new tracks with a initial :class:`~.ParticleState`
        """
        tracks = super().initiate(unassociated_detections, **kwargs)
        for track in tracks:
            samples = multivariate_normal.rvs(track.state_vector.ravel(),
                                              track.covar,
                                              size=self.number_particles)
            particles = [
                Particle(sample.reshape(-1, 1), weight=1/self.number_particles)
                for sample in samples]
            track.states[-1] = ParticleState(particles,
                                             timestamp=track.timestamp)

        return tracks

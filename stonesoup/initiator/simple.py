from scipy.stats import multivariate_normal

from .base import Initiator, GaussianInitiator
from ..base import Property
from ..updater import KalmanUpdater
from ..models.measurement import MeasurementModel
from ..types.track import Track
from ..types.state import GaussianState, ParticleState
from ..types.particle import Particle


class SinglePointInitiator(GaussianInitiator):
    """ SinglePointInitiator class"""

    prior_state = Property(GaussianState, doc="Prior state information")
    measurement_model = Property(MeasurementModel, doc="Measurement model")

    def initiate(self, unassociated_detections, **kwargs):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        unassociated_detections : list of \
        :class:`stonesoup.types.detection.Detection`
            A list of unassociated detections

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

            track_state = GaussianState(
                post_state_vec,
                post_state_covar,
                timestamp=detection.timestamp)
            track = Track([track_state])
            tracks.add(track)

        return tracks


class GaussianParticleInitiator(Initiator):
    """Gaussian Particle Initiator class

    Utilising Gaussian Initiator, sample from the resultant track's state
    to generate a number of particles, overwriting with a
    :class:`~.ParticleState`.
    """

    initiator = Property(
        GaussianInitiator,
        doc="Gaussian Initiator which will be used to generate tracks.")
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
        tracks = self.initiator.initiate(unassociated_detections, **kwargs)
        for track in tracks:
            samples = multivariate_normal.rvs(track.state_vector.ravel(),
                                              track.covar,
                                              size=self.number_particles)
            particles = [
                Particle(sample.reshape(-1, 1), weight=1/self.number_particles)
                for sample in samples]
            track[-1] = ParticleState(particles,
                                      timestamp=track.timestamp)

        return tracks

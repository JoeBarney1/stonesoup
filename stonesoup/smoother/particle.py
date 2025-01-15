#for generic PS
from stonesoup.types.track import Track
from stonesoup.types.state import MarginalisedParticleState
from stonesoup.types.update import MarginalisedParticleStateUpdate
from stonesoup.types.numeric import Probability
from stonesoup.smoother.base import Property, Smoother
import numpy as np
from stonesoup.models.transition.linear import LinearGaussianTimeInvariantTransitionModel
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.state import GaussianState
from stonesoup.smoother.kalman import KalmanSmoother
from stonesoup.types.hypothesis import SingleHypothesis

   
class ParticleSmoother(Smoother):
    """
    Approximate Particle Smoother to reconstruct particle paths using the resampling history.
    This smoother traces back the origin of each particle using resampling indices and smooths  
    the resulting paths by calculating descendant-based weights.
    """

    # Making track a property
    track: Track = Property(doc="The filtered track resulting from the forward pass")
    def smooth(self, max_lag=None, window_length=1, **kwargs):
        # max_lag is how far forward we want to use data. 
        # if proof of independence some steps forward, we can use this to
        # avoid excessively long lag times which only weight a single particle
        """
        Generate particle paths using precomputed indices.

        Parameters
        ----------
        earliest_t : int
            The earliest timestep to consider.
        final_timestep : int, optional
            The final timestep to consider. Defaults to the last timestep in the track.

        Returns
        -------
        tuple
            - List of Tracks for each particle.
            - 2D array of indices for each particle at each timestep.
        """
        if window_length != 1:
            raise NotImplementedError
        
        num_particles = len(self.track[0])
        track_length = len(self.track)
        particle_track_list = [Track() for _ in range(num_particles)]
       
        if max_lag is None: #if unspecified, take all future data into account
            max_lag = track_length - 1

        # Build the particle tracks using the indices
        for t in range(track_length):
            state_t = self.track[t]

            if t + max_lag <= track_length - 1:
                # Recalculate particle indices array if final_timestep still changing
                final_timestep = t + max_lag
                particle_indices = self.get_particle_track_indices(earliest_t=t, final_timestep=final_timestep)

            for i in range(num_particles):
                current_index = particle_indices[i, t]
                prev_index= state_t.resample_index[current_index]  

                particle_prediction = MarginalisedParticleState(state_vector=state_t.hypothesis.prediction.state_vector[...,prev_index],
                                          covariance=state_t.hypothesis.prediction.covariance[...,prev_index])
                hypothesis = SingleHypothesis(
                                measurement=state_t.hypothesis.measurement,
                                prediction=particle_prediction)
                particle_state=MarginalisedParticleStateUpdate(
                            state_vector=state_t.state_vector[...,current_index],
                            covariance=state_t.covariance[...,current_index],
                            weight=np.array([Probability(1)]),
                            hypothesis=hypothesis,
                            timestamp=state_t.timestamp)
                
                particle_track_list[i].append(particle_state)
        
        return particle_track_list
    
    def get_particle_track_indices(self, earliest_t=0, final_timestep=None):
        """
        Compute the particle track indices array based on resampling history.

        Parameters
        ----------
        earliest_t : int
            The earliest timestep to consider.
        final_timestep : int, optional
            The final timestep to consider. Defaults to the last timestep in the track.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_particles, track_length), where each row corresponds
            to a particle's track indices within the filtered track, through time.

            track[t][particle_indices[i,t]] gives the particle at time 't' corresponding to the 'ith'
            descendant/sub-track from time 'final_timestep'. 
        """
        track_length = len(self.track)
        if final_timestep is None:
            final_timestep = track_length - 1
        elif final_timestep >= track_length:
            raise ValueError("final timestep index out of bounds, must be less than track length")

        num_particles = len(self.track[0])
        particle_indices = np.full((num_particles, track_length), None, dtype=object)

        # Initialize the indices for the last timestep
        particle_indices[:, final_timestep] = np.arange(num_particles)

        # Fill indices backward in time
        for t in range(final_timestep - 1, earliest_t - 1, -1):
            for i in range(num_particles):
                particle_indices[i, t] = self.track[t + 1].resample_index[particle_indices[i, t + 1]]

        return particle_indices        

    
    def mean_track(self):
        """
        Compute the mean track from a list of tracks, with optional exclusion of extreme values element-wise.

        Parameters
        ----------
        track_list : list of Tracks
            List of tracks, where each track is a list of states.
        exclude_percent : float, optional
            Percentage of extreme values to exclude from the top and bottom for each element.
            For example, `exclude_percent=10` excludes the top and bottom 10% of values element-wise.

        Returns
        -------
        Track
            A track representing the mean of all input tracks.
        """

        track_list=self.smooth()
        track_length = len(track_list[0])

        if any(len(track) != track_length for track in track_list):
            raise ValueError("All tracks must have the same length.")

        mean_track = Track()

        for t in range(track_length):
            # Collect state vectors and covariances at timestep t
            state_vectors = np.array([track[t].state_vector.flatten() for track in track_list])  # Shape: (num_tracks, state_dim)
            covariances = np.array([track[t].covar for track in track_list])  # Shape: (num_tracks, state_dim, state_dim)

            # Create a GaussianState for the mean state at this timestep
            mean_state = GaussianState(
                state_vector=np.mean(state_vectors, axis=0)[:, np.newaxis],  # Ensure the vector is column-shaped
                covar= np.mean(covariances, axis=0),
                timestamp=track_list[0][t].timestamp  # Use the timestamp from the first track
            )

            mean_track.append(mean_state)

        return mean_track
    
    # def get_descendant_count(self, earliest_t=0, final_timestep=None, particle_indices=None, **kwargs):
    #     [NB: currently redundant due to mean_track funct and us always keeping duplicate particles. if latter changes,
    #     this will be used to scale weight of particles (and thus smooth in computationally efficient manner)]
    #     """
    #     Computes descendant counts for particles at `earliest_t` with respect to `final_timestep`.

    #     Parameters
    #     ----------
    #     earliest_t : int
    #         The earliest timestep to consider.
    #     final_timestep : int, optional
    #         The final timestep to consider. Defaults to the last timestep in the track.

    #     Returns
    #     -------
    #     np.ndarray
    #         A 1D array where each element corresponds to the number of descendants
    #         at `final_timestep` for the respective particle at `earliest_t`.
    #     """
    #     num_particles = len(self.track[0])

    #     # Avoid recomputing particle indices if possible (if final_timestep is const.)
    #     if particle_indices is None:
    #         particle_indices = self.get_particle_track_indices(earliest_t=earliest_t, final_timestep=final_timestep)

    #     # Count occurrences of each particle index at earliest_timestep
    #     earliest_indices = particle_indices[:, earliest_t]
    #     descendant_counts = np.zeros((num_particles,), dtype=int)

    #     # Use np.unique to count occurrences
    #     unique, counts = np.unique(earliest_indices, return_counts=True)
    #     descendant_counts[unique.astype(int)] = counts

    #     return descendant_counts


class MarginalisedKalmanSmoother(ParticleSmoother,KalmanSmoother):
    
    track :Track = Property()

    def particle_paths_to_gaussian_paths(self, max_lag=None, window_length=1, **kwargs):
        """
        Converts particle tracks into GaussianStateUpdate tracks for Kalman smoothing.

        Parameters
        ----------
        lag_length : int, optional
            Specifies how far back in time to consider for smoothing (default: None).

        Returns
        -------
        list of Tracks
            A list of GaussianStateUpdate tracks corresponding to the particle tracks.
        """
        track=self.track        
        particlesmoother=ParticleSmoother(self.track)
        particle_paths=particlesmoother.smooth(max_lag=None, window_length=1, **kwargs)

        F = track[0].hypothesis.prediction.linear_transition_matrix
        transition_model=LinearGaussianTimeInvariantTransitionModel(
                        transition_matrix=F,
                        covariance_matrix=None
                    )
        
        gaussian_track_list = [Track() for path in particle_paths]

        for i,path in enumerate(particle_paths):
            gaussian_state=None
            for t, state in enumerate(path):
                timestamp=state.timestamp
                particle_prediction=state.hypothesis.prediction

                prediction = GaussianStatePrediction(
                    state_vector=particle_prediction.state_vector,
                    covar=particle_prediction.covariance,
                    timestamp=timestamp,
                    transition_model=transition_model,
                    prior=gaussian_state
                )

                gaussian_state=GaussianStateUpdate(
                    state_vector=state.state_vector,
                    covar=state.covariance,
                    timestamp=timestamp,
                    #need to change so that other hypotheses types are allowed.
                    hypothesis=SingleHypothesis(measurement=state.hypothesis.measurement,
                                                prediction=prediction)
                )
                gaussian_track_list[i].append(gaussian_state)

        return gaussian_track_list        
    
    def smooth(self, **kwargs):
        """
        Perform Kalman smoothing on the GaussianStateUpdate tracks.

        Parameters
        ----------
        **kwargs : various
            Additional arguments to pass to the Kalman smoother.

        Returns
        -------
        list of Tracks
            A list of smoothed tracks containing GaussianState objects.
        """

        # Convert particle tracks to GaussianStateUpdate tracks
        gaussian_tracks = self.particle_paths_to_gaussian_paths()

        # Use the KalmanSmoother to smooth each Gaussian track
        F=self.track[0].hypothesis.prediction.linear_transition_matrix
        kalman_smoother = KalmanSmoother(transition_model=F)
        smoothed_tracks = []

        for gaussian_track in gaussian_tracks:
            smoothed_track = kalman_smoother.smooth(gaussian_track, **kwargs)
            smoothed_tracks.append(smoothed_track)

        return smoothed_tracks

class CarterKohnParticleSmoother(MarginalisedKalmanSmoother):
    """
    A smoother implementing the Carter and Kohn (1994) method for Gaussian state-space models.
    It samples states recursively backward using Kalman smoothing principles.
        1. **Forward pass (Kalman filter):**
        for \( t = 1 \) to \( T \):
            Compute the Kalman filter updates:
            - \( x_{t|t-1} = F_t x_{t-1|t-1} \)
            - \( P_{t|t-1} = F_t P_{t-1|t-1} F_t^T + Q_t \)
            - \( K_t = P_{t|t-1} H_t^T (H_t P_{t|t-1} H_t^T + R_t)^{-1} \)
            - \( x_{t|t} = x_{t|t-1} + K_t (y_t - H_t x_{t|t-1}) \)
            - \( P_{t|t} = P_{t|t-1} - K_t H_t P_{t|t-1} \)

        2. **Backward sampling (Carter-Kohn):**
        Initialize \( x_T^{(i)} \sim N(x_{T|T}, P_{T|T}) \)
        for \( t = T-1 \) to \( 1 \):
            - \( x_{t|T}^{(i)} = \mathbb{E}[x_t | x_{t+1}, y_{1:T}] + \text{noise} \)
            - Compute:
                \( J_t = P_{t|t} F_{t+1}^T P_{t+1|t}^{-1} \)
                \( \mu_{t|T} = x_{t|t} + J_t (x_{t+1}^{(i)} - F_{t+1} x_{t|t}) \)
                \( \Sigma_{t|T} = P_{t|t} - J_t P_{t+1|t} J_t^T \)
            - Sample \( x_t^{(i)} \sim N(\mu_{t|T}, \Sigma_{t|T}) \)

        3. **Return smoothed samples** \( x_{1:T}^{(i)} \).
        """

    track: Track = Property(doc="Filtered track resulting from the forward pass")

    def smooth(self, lag_length=None, **kwargs):
        """
        Perform Carter-Kohn smoothing.

        Returns
        -------
        smoothed_track_list : list of Tracks
            Smoothed tracks for each particle.
        """
        # Convert particle tracks to GaussianStateUpdate tracks
        gaussian_tracks = self.particle_paths_to_gaussian_paths(lag_length)

        smoothed_track_list = []

        for gaussian_track in gaussian_tracks:
            track_length = len(gaussian_track)
            subsq_state = gaussian_track[-1]

            smoothed_track = Track()
            smoothed_track.append(subsq_state)

            # Backward sampling using Carter-Kohn
            for t in range(track_length - 2, -1, -1):
                current_state = gaussian_track[t]
                F = current_state.hypothesis.prediction.transition_model.matrix()

                # Posterior mean and covariance
                post_sv = current_state.state_vector
                post_covar = current_state.covar

                # Compute smoothing gain
                pred_covar = subsq_state.covar
                ksmooth_gain = post_covar @ F.T @ np.linalg.inv(pred_covar)

                # Smoothed mean and covariance
                smooth_mean = post_sv + ksmooth_gain @ (subsq_state.state_vector -
                                                        current_state.hypothesis.prediction.state_vector)
                smooth_covar = post_covar - ksmooth_gain @ pred_covar @ ksmooth_gain.T

                # Sample from the smoothed Gaussian
                smooth_sample = np.random.multivariate_normal(
                    smooth_mean.flatten(), smooth_covar
                ).reshape(-1, 1)

                # Update the subsq_state
                subsq_state = GaussianState(
                    state_vector=smooth_sample,
                    covar=smooth_covar,
                    timestamp=current_state.timestamp,
                )

                smoothed_track.insert(0, subsq_state)

            smoothed_track_list.append(smoothed_track)

        return smoothed_track_list
    
class MCparticle_smoother(ParticleSmoother):
    track: Track=Property()
    def MCparticle_smooth(self):
        raise NotImplementedError
        """
        Perform the backward pass as described in Algorithm 5.
        """
        smoothed_track = Track()
        track_length=len(self.track)
        num_particles=len(self.track[0])

        # Initialize smoothed track with the last filtered state, acting as xT:T|T:T (is just xT|y0:T), where T=track_length-1
        x_T=self.track[-1]
        smoothed_track.append(x_T)
        
        # Initialize weights with the last state's weights (normalized)
        weights = np.array([particle.weight for particle in self.track[-1]])
        weights /= np.sum(weights)
        
        # Iterate backward through time (T-1 to 0)
        for t in range(track_length - 2, -1, -1):
            new_weights = np.zeros(num_particles)
            
            # Get the current state and particles
            current_state = self.track[t]
            
            # Iterate over each particle
            for x_t in current_state:
                # Placeholder for f(x_{t+1} | x_t)
                def transition_prob(x_t_next, x_t):
                    # TODO: Implement f(x_{t+1} | x_t) here based on your model
                    return 1.0  # Placeholder value, replace with actual function

                # Compute the new weights using Eq (29)
                for j in range(num_particles):
                    x_t_plus_1 = smoothed_track[0][j].state_vector
                    weight = weights[j] * transition_prob(x_t_plus_1, x_t)
                    new_weights[i] += weight
            
            # Normalize the weights
            new_weights /= np.sum(new_weights)
            
            # Resample particles based on new weights
            sampled_indices = np.random.choice(num_particles, size=num_particles, p=new_weights)
            
            # Construct new smoothed state
            new_smoothed_state = []
            for idx in sampled_indices:
                particle = current_state[idx]
                particle.weight = Probability(new_weights[idx])
                new_smoothed_state.append(particle)
            
            # Append to smoothed track
            smoothed_track.insert(0, MarginalisedParticleState.from_list(new_smoothed_state, current_state.timestamp))
        
        return smoothed_track

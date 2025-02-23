#for generic PS
from ast import FunctionType
import copy
from sre_parse import State

from scipy.stats import multivariate_normal
from stonesoup.models.transition.categorical import MarkovianTransitionModel
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.particle import Particle
from stonesoup.types.prediction import MarginalisedParticleStatePrediction
from stonesoup.types.state import CategoricalState
from stonesoup.types.track import Track
from stonesoup.types.update import MarginalisedParticleStateUpdate
from stonesoup.types.array import CovarianceMatrices, StateVector,StateVectors
from stonesoup.smoother.base import Property, Smoother
import numpy as np
from stonesoup.smoother.kalman import KalmanSmoother
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.models.transition.categorical import MarkovianTransitionModel
   
class ParticleSmoother(Smoother):
    """
    Approximate Particle Smoother to reconstruct particle paths using the resampling history.
    This smoother traces back the origin of each particle using resampling indices and smooths  
    the resulting paths by calculating descendant-based weights.
    """

    # Making track a property
    track: Track = Property(doc="The filtered track resulting from the forward pass",default=None)

    def smooth():
        raise NotImplementedError

    def particle_paths(self,track=None, max_lag=None):
        """
        Create a single Track containing all particle states for times t in
        [earliest_t..earliest_t+max_lag], each as a MarginalisedParticleState.
        """
        if track is None:
            track=self.track

        track_length = len(track)
        num_particles = len(track[0])
        particle_indices,prior_idx_t = self.get_particle_track_indices(track=track,earliest_t=0)
        combined_track = Track()
        prev_idx_t=prior_idx_t

        for t in range(track_length):
            # Gather the 'resample' indices for all particles at time t
            idx_t = particle_indices[:, t].astype(int)  # shape (num_particles,)
            state_t = track[t]
            timestamp = state_t.timestamp
            prediction = state_t.hypothesis.prediction

            # Build a single MarginalisedParticleState for time t
            reordered_prediction = MarginalisedParticleStatePrediction(
                state_vector=prediction.state_vector[..., prev_idx_t], 
                covariance=prediction.covariance[..., prev_idx_t],
                timestamp=prediction.timestamp,
                linear_transition_matrix=prediction.linear_transition_matrix,
                process_mean=prediction.process_mean[..., prev_idx_t],
                process_covar=prediction.process_covar[..., prev_idx_t],
                log_weight=np.array([np.log(1 / num_particles)] * num_particles)
            )

            hypothesis = SingleHypothesis(
                measurement=state_t.hypothesis.measurement,
                prediction=reordered_prediction)

            combined_state = MarginalisedParticleStateUpdate(
                state_vector=state_t.state_vector[..., idx_t], # shape (M, num_particles)
                covariance=state_t.covariance[..., idx_t] , # shape (M, M, num_particles)
                log_weight=np.array([np.log(1 / num_particles)] * num_particles),
                hypothesis=hypothesis,
                timestamp=timestamp
            )

            combined_track.append(combined_state)
            prev_idx_t=idx_t

        return combined_track
    
    def get_particle_track_indices(self,track=None, earliest_t=0, final_timestep=None,store_particle_nums=False,**kwargs):
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
        if track is None:
            track=self.track

        # if store_particle_nums:
        #     particle_nums=np.zeros_like(track)

        track_length = len(track)
        if final_timestep is None or final_timestep >= track_length: 
            final_timestep = track_length - 1

        num_particles = len(track[0])
        particle_indices = np.full((num_particles, track_length), 
                                   num_particles, #will throw an index error if any elements unassigned
                                   dtype=int) 

        # Initialize the indices for the last timestep
        particle_indices[:, final_timestep] = np.arange(num_particles)

        # Fill indices backward in time
        for t in range(final_timestep - 1, earliest_t - 1, -1):
            particle_indices[:, t] = track[t + 1].resample_index[particle_indices[:, t + 1]]
            # if store_particle_nums:
            #     particle_nums[t]=len(np.unique(particle_indices[:,t]))
            # print(f"{len(np.unique(particle_indices[:,t]))} unique particles at time t={track[t].timestamp}")
                
        if earliest_t>0:
            prior_indices=None
        else:
            prior_indices = track[0].resample_index[particle_indices[:, 0]]
        
        return particle_indices, prior_indices

class MarginalisedKalmanSmoother(ParticleSmoother,KalmanSmoother):
    
    r"""
    The linear-Gaussian or Rauch-Tung-Striebel smoother, colloquially the Kalman smoother [1]_. The
    transition model is therefore linear-Gaussian. No control model is currently implemented.

    TODO: Include a control model

    The smooth function undertakes the backward algorithm on a :class:`~.Track()` object. This is
    done by starting at the final index in the track, :math:`K` and proceeding from
    :math:`K \rightarrow 1` via:

    .. math::

        \mathbf{x}_{k|k-1} &= F_{k} \mathbf{x}_{k-1}

        P_{k|k-1} &= F_{k} P_{k-1} F_{k}^T + Q_{k}

        G_k &= P_{k-1} F_{k}^T P_{k|k-1}^{-1}

        \mathbf{x}_{k-1}^s &= \mathbf{x}_{k-1} + G_k (\mathbf{x}_{k}^s - \mathbf{x}_{k|k-1})

        P_{k-1}^s &= P_{k-1} + G_k (P_{k}^s - P_{k|k-1}) G_k^T

    where :math:`\mathbf{x}_{K}^s = \mathbf{x}_{K}` and :math:`P_K^s = P_K`.

    The predicted state vector and covariance are retrieved from the Track via predicted state or
    updated state via the links therein. Note that this means that the first two equations are not
    calculated, the results merely retrieved. This smoother is therefore strictly Kalman only in
    the backward portion. The prediction might have come by any number of means. If present, the
    transition model (providing :math:`F` and :math:`Q`) in the prediction is used. This allows for
    a dynamic transition model (i.e. one that changes with :math:`k`). Otherwise, the (static)
    transition model is used, defined on smoother initialisation.

    References

    .. [1] Särkä S. 2013, Bayesian filtering and smoothing, Cambridge University Press

    """
    transition_matrix : np.ndarray=Property(default=None)

    def _prediction(self, state):
        """ Return the predicted state, either from the prediction directly, or from the attached
        hypothesis if the queried state is an Update. If not a :class:`~.GaussianStatePrediction`
        or :class:`~.GaussianStateUpdate` a :class:`~.TypeError` is thrown.

        Parameters
        ----------
        state : :class:`~.GaussianStatePrediction` or :class:`~.GaussianStateUpdate`

        Returns
        -------
         : :class:`~.GaussianStatePrediction`
            The prediction associated with the prediction (i.e. itself), or the prediction from the
            hypothesis used to generate an update.
        """
        if isinstance(state, MarginalisedParticleStatePrediction):
            return state
        elif isinstance(state, MarginalisedParticleStateUpdate):
            if isinstance(state.hypothesis, MultipleHypothesis):
                predictions = {hypothesis.prediction for hypothesis in state.hypothesis}
                if len(predictions) == 1:
                    # One predictions, this is fine to use.
                    return predictions.pop()
                else:
                    # Multiple predictions, so can't process this.
                    raise ValueError(
                        "Track has MultipleHypothesis updates with multiple predictions.")
            else:
                return state.hypothesis.prediction
        else:
            raise TypeError("States must be MarginalisedParticlePredictions or MarginalisedParticleUpdates.")

    def _transition_matrix(self, prediction):
        """ Return the transition matrix

        Parameters
        ----------
        state : :class:`~.State`
            The input state (to check for a linked prediction)
        **kwargs
            These are passed to the :meth:`matrix()` function

        Returns
        -------
         : :class:`numpy.ndarray`
            The transition matrix
        """
        # Is there a transition model linked to the prediction?
        transition_matrix = getattr(prediction, "linear_transition_matrix", None)

        if transition_matrix is None:
            transition_matrix=self.transition_matrix
        if transition_matrix is None:
            raise ValueError('neither input transition matrix, nor one attached to prediction')
        return transition_matrix

    def _smooth_gain(self, state, prediction, **kwargs):
        """Calculate the smoothing gain

        Parameters
        ----------
        state : :class:`~.State`
            The input state
        prediction : :class:`~.GaussianStatePrediction`
            The prediction (from the subsequent state)

        Returns
        -------
         : Matrix
            The smoothing gain

        """

        covar_state = np.moveaxis(state.covariance, 2, 0)   # (N, M, M)
        F = self._transition_matrix(prediction) #MxM
        Ft = F.T #MxM
        covar_pred = np.moveaxis(prediction.covariance, 2, 0)  # (N, M, M)

        # Multiply in batch form
        covarF = np.einsum("nij,jk->nik", covar_state, Ft)            # (N, M, M)
        covar_pred_inv = np.linalg.inv(covar_pred)                    # (N, M, M)
        ksmooth_gain = np.einsum("nij,njk->nik", covarF, covar_pred_inv)  # (N, M, M)

        return np.moveaxis(ksmooth_gain, 0, 2)  # (M, M, N)
    
    def smooth(self, culled_track=None, **kwargs):
        """
        Perform the backward recursion to smooth the track.

        Parameters
        ----------
        track : :class:`~.Track`
            The input track.

        Returns
        -------
         : :class:`~.Track`
            Smoothed track

        """
        if culled_track is None:
            track=self.track
            culled_track=ParticleSmoother().particle_paths(track=track)
        track=culled_track

        try:
            self._prediction(track[0])
            start = 0
        except (ValueError, TypeError):
            start = 1

        subsq_state = track[-1]
        smoothed_states = [subsq_state]
        for state in reversed(track[start:-1]):

            # Delta t
            time_interval = subsq_state.timestamp - state.timestamp

            # Retrieve the prediction from the subsequent (k+1th) timestep accessed previously
            prediction = self._prediction(subsq_state)

            # The smoothing gain, mean and covariance. operations here will be done on padded states
            ksmooth_gain = self._smooth_gain(
                state, prediction, time_interval=time_interval, **kwargs)
            
            diff = subsq_state.state_vector - prediction.state_vector #(MxN)
            smooth_mean = state.state_vector + np.einsum("imn,mn->in", ksmooth_gain, diff) #MxMxN x Mx1xN--> Mx1xN
            resid_covar = subsq_state.covariance - prediction.covariance #MxMxN
            
            smooth_covar = state.covariance + np.einsum(
                "abn,bcn,cdn->adn",
                ksmooth_gain,
                resid_covar,
                ksmooth_gain.transpose((1, 0, 2))
            )

            #Generate state with updated state vector and covariance
            subsq_state = type(state).from_state(state, 
                                                state_vector=smooth_mean, 
                                                covariance=smooth_covar,
                                                hypothesis=state.hypothesis,
                                                timestamp=state.timestamp)

            smoothed_states.insert(0, subsq_state)

        if start == 1:
            smoothed_states.insert(0, track[0])

        # Deep copy existing track, but avoid copying original states, as this would be super
        # expensive. This works by informing deepcopy that the smoothed states are the
        # replacement object for the original track states.
        smoothed_track = copy.deepcopy(track, {id(track.states): smoothed_states})
        return smoothed_track   

class CarterKohnSmoother(MarginalisedKalmanSmoother):
    """
    Carter-Kohn smoother implementation in Stone Soup framework.
    """
    # If MCMCsample=None, we only need these:
    measurements: list = Property(default=None)

    def smooth(self, culled_track=None, **kwargs):
        """
        Vectorised backward pass for Carter-Kohn or standard smoothing,
        handling shapes (M,N) for state vectors and (M,M,N) for covariances.
        If N=1, it degenerates to the single-track case.
        """
        if culled_track is None:
            track=self.track
            if track is None:
                raise ValueError("If MCMCsample=None and you are smoothing, you must provide 'track'") 
            culled_track=ParticleSmoother().particle_paths(track=track)

        track=culled_track

        smoothed_track = Track()

        # Start from last state
        final_state = track[-1]
        smoothed_track.append(final_state)
        subsq_state = final_state

        x_t_plus_1 = subsq_state.state_vector  # (M, N)
        m, ncols = x_t_plus_1.shape  # N == ncols

        self.sigma_residuals_term = 0
        self.tau_likelihood_term = 0

        # Walk backwards
        for t in range(len(track) - 2, -1, -1):
            current_state = track[t]
            time_interval = subsq_state.timestamp - current_state.timestamp

            x_t_given_t = current_state.state_vector.copy()   # (M, N)
            
            # Attempt to get the forward prediction
            # (If normal Stone Soup usage, this is in subsq_state.hypothesis.prediction)
            try:
                prediction_t_1 = subsq_state.hypothesis.prediction
                F_t_1 = prediction_t_1.linear_transition_matrix  # (M, M)
                U_t_1 = prediction_t_1.process_covar            # (M, M, N)
                S_t_given_t = current_state.covariance.copy()   # (M, M, N)
                process_mean = prediction_t_1.process_mean       # (M, N)
            except:
                # Fallback if it's a direct transition model
                transition_model = subsq_state.hypothesis.prediction.transition_model
                F_t_1 = np.atleast_2d(transition_model.matrix(time_interval=time_interval))
                U_t_1 = np.atleast_3d(transition_model.covar(time_interval=time_interval))
                S_t_given_t = np.atleast_3d(current_state.covar.copy())
                # If no separate process_mean is stored, assume zero
                process_mean = np.atleast_2d(np.zeros_like(x_t_plus_1))

            # Move U_t_1 to shape (N, M, M) for Cholesky
            U_batch = np.moveaxis(np.atleast_3d(U_t_1), 2, 0)      # (N, M, M)
            L_batch = np.linalg.cholesky(U_batch)   # (N, M, M)
            inv_L_batch = np.linalg.inv(L_batch)    # (N, M, M)

            # Move back to (M, M, N) so we have parallel slices
            inv_L_t_1 = np.moveaxis(inv_L_batch, 0, 2)

            # Delta_t_1 = inv_L * U * inv_L^T in batch
            U_batch_invLT = np.einsum("nij,nkj->nik", U_batch, inv_L_batch)  # (N,M,M)
            Delta_batch = np.einsum("nij,njk->nik", inv_L_batch, U_batch_invLT)  # (N,M,M)
            Delta_t_1 = np.moveaxis(Delta_batch, 0, 2)   # (M, M, N)

            # Transform x_{t+1}, process_mean, etc. using inv_L
            x_tilda_t_1 = np.einsum("imn,mn->in", inv_L_t_1, x_t_plus_1)    # (M, N)
            mean_tilda = np.einsum("imn,mn->in", inv_L_t_1, process_mean)   # (M, N)
            F_tilda_t_plus_1 = np.atleast_3d(np.einsum("imn,mk->ikn", inv_L_t_1, F_t_1))   # (M, M, N)

            # Row-by-row Carter–Kohn
            for i in range(m):
                F_i = np.atleast_2d(F_tilda_t_plus_1[i, :, :])   # (M, N)
                Delta_i = np.atleast_1d(Delta_t_1[i, i, :])      # (N,)
                mean_t_i = np.atleast_1d(mean_tilda[i, :])        # (N,)

                # e_t_i => shape (N,)
                # The dot product over M => np.einsum with x_t_given_t (M,N) and F_i (M,N)
                # We'll do e(t)= x_tilda(i,:)- sum_k(F_i(k)* x_t_given_t(k)) - mean_t_i
                # We'll treat F_i as shape (M,N) => multiply each row by x_t_given_t row, sum across M
                e_t_i = x_tilda_t_1[i] - np.einsum("mn,mn->n", x_t_given_t, F_i) - mean_t_i

                # R_t_i => sum_{m1,m2} F_i(m1,n)*S_t(m1,m2,n)* F_i(m2,n) + Delta_i(n)
                R_t_i = np.einsum("in, ijn, jn-> n", F_i, S_t_given_t, F_i) + Delta_i

                gain = np.einsum("lmn,ln->mn",S_t_given_t, F_i)/ R_t_i  # broadcast (M,N)/(N,)

                # update x_t_given_t => shape (M,N)
                x_t_given_t += gain * e_t_i 

                # update S_t_given_t => subtract gain*gai^T * R_t_i
                S_t_given_t-=np.einsum("lmn,n->lmn",np.einsum("mn, ln->mln",gain,gain),R_t_i) 

            # Sample each column => shape (M,N)
            sampled_state = np.zeros_like(x_t_given_t)
            for idx in range(ncols):
                mean_ = x_t_given_t[:, idx]
                cov_ = S_t_given_t[:, :, idx]
                sampled_state[:, idx] = np.random.multivariate_normal(mean_, cov_)

            new_state = type(current_state).from_state(
                current_state,
                state_vector=sampled_state,
                covariance=S_t_given_t,
                hypothesis=current_state.hypothesis,
                timestamp=current_state.timestamp
            )

            smoothed_track.insert(0, new_state)
            subsq_state = new_state
            x_t_plus_1 = sampled_state

        return smoothed_track
    
# class CarterKohnSampler(MarginalisedKalmanSmoother):
    
#     # Tells us if we are using MCMC method or not (if None, we're just smoothing)
#     MCMCsample: bool = Property(default=False)

#     # If MCMCsample=None, we only need these:
#     measurements: list = Property(default=None)

#     # If MCMCsample=True, we need these:
#     parameters: list[tuple] = Property(default=None)
#     C: float = Property(default=None)
#     X_prior: State = Property(default=None)
#     K_prior: CategoricalState = Property(default=None)
#     K_transition_matrix: np.ndarray = Property(default=None)
#     conditional_models_funct: FunctionType = Property(default=None)
#     parameter_posteriors: list[callable] = Property(default=None)

#     # Updated as we go along
#     indicator_variables: list = Property(default=None)
#     generated_states: Track = Property(default=None)
#     conditional_models: list[callable] = Property(default=None)

#     # Carter–Kohn likelihood terms
#     tau_likelihood_term: float = Property(default=0)
#     sigma_residuals_term: float = Property(default=0)

#     def __init__(self, *args, **kwargs):
#         """Custom init to check required properties depending on MCMCsample."""
#         super().__init__(*args, **kwargs)
#         if self.MCMCsample is not None:
#             # If MCMCsample is True, we typically require all the others
#             required_props = [
#                 "parameters", "C", "X_prior", "K_prior", "K_transition_matrix",
#                 "conditional_models_funct", "parameter_posteriors"
#             ]
#             for propname in required_props:
#                 if getattr(self, propname) is None:
#                     raise ValueError(f"'{propname}' cannot be None if MCMCsample=True")
#             self.conditional_models=self.conditional_models_funct(self.parameters[-1],self.C)
    
#     def smooth(self, culled_track=None, **kwargs):
#         """
#         Vectorised backward pass for Carter-Kohn or standard smoothing,
#         handling shapes (M,N) for state vectors and (M,M,N) for covariances.
#         If N=1, it degenerates to the single-track case.
#         """
#         if culled_track is None:
#             track=self.track
#             if track is None:
#                 raise ValueError("If MCMCsample=None and you are smoothing, you must provide 'track'") 
#             culled_track=ParticleSmoother().particle_paths(track=track)

#         track=culled_track

#         smoothed_track = Track()

#         # Start from last state
#         final_state = track[-1]
#         smoothed_track.append(final_state)
#         subsq_state = final_state

#         x_t_plus_1 = subsq_state.state_vector  # (M, N)
#         m, ncols = x_t_plus_1.shape  # N == ncols

#         self.sigma_residuals_term = 0
#         self.tau_likelihood_term = 0

#         # Walk backwards
#         for t in range(len(track) - 2, -1, -1):
#             current_state = track[t]
#             time_interval = subsq_state.timestamp - current_state.timestamp

#             # If MCMCsample is True, get K(t) and pick measurement model
#             if self.MCMCsample:
#                 K_t = np.argmax(self.indicator_variables[t])
#                 measurement_model= self.conditional_models["measurement"][K_t]
#                 H =measurement_model.matrix()
#                 # v = measurement_model.noise_covar()
#                 C = 1 if K_t == 0 else 100
#             else:
#                 # If no Markov switching, just use measurement model from the forward pass
#                 H = current_state.hypothesis.measurement.measurement_model.matrix()
#                 C = 1

#             x_t_given_t = current_state.state_vector.copy()   # (M, N)
            
#             # Attempt to get the forward prediction
#             # (If normal Stone Soup usage, this is in subsq_state.hypothesis.prediction)
#             try:
#                 prediction_t_1 = subsq_state.hypothesis.prediction
#                 F_t_1 = prediction_t_1.linear_transition_matrix  # (M, M)
#                 U_t_1 = prediction_t_1.process_covar            # (M, M, N)
#                 S_t_given_t = current_state.covariance.copy()   # (M, M, N)
#                 process_mean = prediction_t_1.process_mean       # (M, N)
#             except:
#                 # Fallback if it's a direct transition model
#                 transition_model = subsq_state.hypothesis.prediction.transition_model
#                 F_t_1 = np.atleast_2d(transition_model.matrix(time_interval=time_interval))
#                 U_t_1 = np.atleast_3d(transition_model.covar(time_interval=time_interval))
#                 S_t_given_t = np.atleast_3d(current_state.covar.copy())
#                 # If no separate process_mean is stored, assume zero
#                 process_mean = np.atleast_2d(np.zeros_like(x_t_plus_1))

#             # Move U_t_1 to shape (N, M, M) for Cholesky
#             U_batch = np.moveaxis(np.atleast_3d(U_t_1), 2, 0)      # (N, M, M)
#             L_batch = np.linalg.cholesky(U_batch)   # (N, M, M)
#             inv_L_batch = np.linalg.inv(L_batch)    # (N, M, M)

#             # Move back to (M, M, N) so we have parallel slices
#             inv_L_t_1 = np.moveaxis(inv_L_batch, 0, 2)

#             # Delta_t_1 = inv_L * U * inv_L^T in batch
#             U_batch_invLT = np.einsum("nij,nkj->nik", U_batch, inv_L_batch)  # (N,M,M)
#             Delta_batch = np.einsum("nij,njk->nik", inv_L_batch, U_batch_invLT)  # (N,M,M)
#             Delta_t_1 = np.moveaxis(Delta_batch, 0, 2)   # (M, M, N)

#             # Transform x_{t+1}, process_mean, etc. using inv_L
#             x_tilda_t_1 = np.einsum("imn,mn->in", inv_L_t_1, x_t_plus_1)    # (M, N)
#             mean_tilda = np.einsum("imn,mn->in", inv_L_t_1, process_mean)   # (M, N)
#             F_tilda_t_plus_1 = np.atleast_3d(np.einsum("imn,mk->ikn", inv_L_t_1, F_t_1))   # (M, M, N)

#             # Row-by-row Carter–Kohn
#             for i in range(m):
#                 F_i = np.atleast_2d(F_tilda_t_plus_1[i, :, :])   # (M, N)
#                 Delta_i = np.atleast_1d(Delta_t_1[i, i, :])      # (N,)
#                 mean_t_i = np.atleast_1d(mean_tilda[i, :])        # (N,)

#                 # e_t_i => shape (N,)
#                 # The dot product over M => np.einsum with x_t_given_t (M,N) and F_i (M,N)
#                 # We'll do e(t)= x_tilda(i,:)- sum_k(F_i(k)* x_t_given_t(k)) - mean_t_i
#                 # We'll treat F_i as shape (M,N) => multiply each row by x_t_given_t row, sum across M
#                 e_t_i = x_tilda_t_1[i] - np.einsum("mn,mn->n", x_t_given_t, F_i) - mean_t_i

#                 # R_t_i => sum_{m1,m2} F_i(m1,n)*S_t(m1,m2,n)* F_i(m2,n) + Delta_i(n)
#                 R_t_i = np.einsum("in, ijn, jn-> n", F_i, S_t_given_t, F_i) + Delta_i

#                 gain = np.einsum("lmn,ln->mn",S_t_given_t, F_i)/ R_t_i  # broadcast (M,N)/(N,)

#                 # update x_t_given_t => shape (M,N)
#                 x_t_given_t += gain * e_t_i 

#                 # update S_t_given_t => subtract gain*gai^T * R_t_i
#                 S_t_given_t-=np.einsum("lmn,n->lmn",np.einsum("mn, ln->mln",gain,gain),R_t_i) 

#                 if self.MCMCsample:
#                     self.tau_likelihood_term += np.sum(e_t_i**2 / (R_t_i * C))

#             # Sample each column => shape (M,N)
#             sampled_state = np.zeros_like(x_t_given_t)
#             for idx in range(ncols):
#                 mean_ = x_t_given_t[:, idx]
#                 cov_ = S_t_given_t[:, :, idx]
#                 sampled_state[:, idx] = np.random.multivariate_normal(mean_, cov_)

#             if self.MCMCsample:
#                 # Residual => shape depends on measurement
#                 residual = (self.measurements[t].state_vector - H @ sampled_state) # same dim as measurement, call it k

#                 self.sigma_residuals_term += np.einsum("kn,kn->n",residual, residual)/C # np.einsum
#                 #Need to check this if ever used, but should be N sets of a constant, so residual.T@residual

#                 new_state = type(current_state).from_state(
#                     current_state,
#                     state_vector=sampled_state[...,0],
#                     covar=S_t_given_t[...,0],
#                     timestamp=current_state.timestamp
#                 )
#             else:
#                 new_state = type(current_state).from_state(
#                     current_state,
#                     state_vector=sampled_state,
#                     covariance=S_t_given_t,
#                     hypothesis=current_state.hypothesis,
#                     timestamp=current_state.timestamp
#                 )

#             smoothed_track.insert(0, new_state)
#             subsq_state = new_state
#             x_t_plus_1 = sampled_state

#         if self.MCMCsample:
#             self.generated_states = smoothed_track
#         return smoothed_track
          
    # def resample(self,num_iterations):
    #     num_params=len(self.parameters[0])
    #     parameter_index=0
    #     self.generate_indicator_variables()
    #     for n in range(num_iterations):
    #         forward_states=self.generate_states_forward_pass() #gen states forward using kalman
            
    #         self.smooth(track=forward_states) #gen states backwards using smoother
            
    #         self.generate_indicator_variables() #gen indicator variables
            
    #         self.update_parameters(parameter_index) #update parameter based on index

    #         parameter_index = (parameter_index + 1) % num_params
        
    #     return self.generated_states, self.parameters, self.indicator_variables

    # def generate_indicator_variables(self):
    #     """
    #     Generate the indicator variables K using the discrete filter (forward-backward) algorithm
    #     from Appendix 2, conditioned on the already-generated states self.generated_states 
    #     and measurements self.measurements.
    #     """
    #     num_timesteps = len(self.measurements)
    #     transition_matrix = self.K_transition_matrix
    #     m = transition_matrix.shape[0]  # number of possible indicator states

    #     K_list = Track()

    #     if self.indicator_variables is None:
    #         # ------------------------------------------------------------
    #         # If no existing indicator variables, simply sample forward 
    #         # from the prior Markov model (as in your original code).
    #         # ------------------------------------------------------------
    #         K_list.append(self.K_prior)
    #         for t in range(1, num_timesteps):
    #             transition_model = MarkovianTransitionModel(transition_matrix=transition_matrix)
    #             time_interval = self.measurements[t].timestamp - self.measurements[t-1].timestamp
    #             K_t_state_vector = transition_model.function(state=K_list[-1], 
    #                                             time_interval=time_interval, 
    #                                             noise=True)
    #             K_list.append(CategoricalState(state_vector=K_t_state_vector,timestamp=self.measurements[t].timestamp))
        
    #     else:
    #         # ------------------------------------------------------------
    #         # Once states are known, perform forward-backward to sample K.
    #         # ------------------------------------------------------------

    #         # 1) Forward Filtering
    #         #
    #         # forward_probs[t, k] will store p(K(t)=k | Y^(t), X^(t))
    #         # (the normalised filtering distribution).
    #         #
            
    #         forward_probs = np.zeros((num_timesteps, m))
            
    #         # For convenience
    #         measurements = self.measurements   # y(0), y(1), ...
    #         generated_states = self.generated_states  # x(0), x(1), ...
            
    #         # Step 0: initialise forward_probs[0] 
    #         # We multiply the prior by p(y(0)| x(0), k) * p(x(0)| x(-1), k) if that is part
    #         # of your model. Often x(0) has its own prior, so you might only use p(y(0)| x(0), k).
    #         # Below we skip p(x(0)| x(-1), k) for simplicity.
    #         for k in range(m):
    #             # If your model designates y(0) as "observed", multiply by p(y(0)| x(0), k).
    #             # If y(0) is unobserved, you can omit the measurement factor.
    #             p_y = self.p_y_given_x_K(measurements[0], generated_states[0], k)
    #             forward_probs[0, k] = self.K_prior.state_vector[k] * p_y
            
    #         # Normalise
    #         forward_probs[0, :] /= np.sum(forward_probs[0, :])

    #         # Now iterate for t = 1..(num_timesteps-1):
    #         for t in range(1, num_timesteps):
    #             for k in range(m):
    #                 # Step 1 (predictive sum):
    #                 # prior_tk = sum over j of p(K(t)=k|K(t-1)=j)*forward_probs[t-1,j]
    #                 prior_tk = 0.0
    #                 for j in range(m):
    #                     prior_tk += transition_matrix[j, k] * forward_probs[t-1, j]
                    
    #                 # Step 2 (multiply by measurement/transition likelihoods if relevant):
    #                 # If y(t) is "observed":
    #                 if measurements[t] is not None:  # or however you detect 'observed'
    #                     p_y = self.p_y_given_x_K(measurements[t], generated_states[t], k)
    #                     p_x = self.p_x_given_x_prev_K(generated_states[t],
    #                                                 generated_states[t-1], k)
    #                     forward_probs[t, k] = prior_tk * p_y * p_x
    #                 else:
    #                     # If y(t) is not observed:
    #                     p_x = self.p_x_given_x_prev_K(generated_states[t],
    #                                                 generated_states[t-1], k)
    #                     forward_probs[t, k] = prior_tk * p_x
                
    #             # Normalise
    #             normaliser = np.sum(forward_probs[t, :])
    #             if normaliser > 0:
    #                 forward_probs[t, :] /= normaliser
    #             else:
    #                 # If normaliser is 0, you may need a small fix or re-check model
    #                 forward_probs[t, :] = 1.0/m

    #         # 2) Backward Sampling / Smoothing
    #         # We sample K(num_timesteps-1) directly from forward_probs[num_timesteps-1].
    #         K_smooth = np.zeros(num_timesteps, dtype=int)
            
    #         # Sample K(T-1) from forward_probs[T-1]
    #         final_dist = forward_probs[num_timesteps-1, :]
    #         K_smooth[num_timesteps-1] = np.random.choice(m, p=final_dist)
            
    #         # Then proceed backwards from t = num_timesteps-2 down to 0
    #         for t in reversed(range(num_timesteps-1)):
    #             k_next = K_smooth[t+1]
    #             # We want p(K(t)=k | K(t+1)=k_next, Y^*, X^*)
    #             # = transition_matrix[k, k_next] * forward_probs[t,k] / 
    #             #   sum_{k'} [transition_matrix[k', k_next] * forward_probs[t,k']]
    #             numerators = transition_matrix[:, k_next] * forward_probs[t, :]
    #             denom = np.sum(numerators)
    #             if denom > 0:
    #                 cond_dist = numerators / denom
    #             else:
    #                 cond_dist = np.ones(m) / m
                
    #             # Sample from that distribution:
    #             K_smooth[t] = np.random.choice(m, p=cond_dist)

    #         # Finally, populate K_list with the sampled indicator values
    #         for t,k_val in enumerate(K_smooth):
    #             K_vector=np.zeros(m)
    #             K_vector[k_val]=1
    #             K_state=CategoricalState(state_vector=StateVector(K_vector),timestamp=measurements[t])
    #             K_list.append(K_state)
    #     self.indicator_variables = K_list
    
    # def p_y_given_x_K(self, y_t, x_t, K_t):
    #     """
    #     Calculate p(y(t)|x(t), K(t)).
    #     This function should be implemented based on the specific model.
    #     """
    #     H=self.conditional_models['measurement'][K_t].matrix()
    #     covar=self.conditional_models['measurement'][K_t].noise_covar
    #     likelihood = multivariate_normal.pdf(y_t.state_vector.flatten(), mean=H@x_t.state_vector.flatten(), cov=covar)
    #     return likelihood
    
    # def p_x_given_x_prev_K(self, x_t, x_t_prev, K_t):
    #     """
    #     Calculate p(x(t)|x(t-1), K(t)).
    #     This function should be implemented based on the specific model.
    #     """
    #     F=self.conditional_models['transition'][K_t].matrix(time_interval=x_t.timestamp-x_t_prev.timestamp)
    #     covar=self.conditional_models['transition'][K_t].covar(time_interval=x_t.timestamp-x_t_prev.timestamp)
    #     probability = multivariate_normal.pdf(x_t.state_vector.flatten(), mean=F@x_t_prev.state_vector.flatten(), cov=covar) 
    #     return probability
    
    # def generate_states_forward_pass(self):
    #     K_list=self.indicator_variables
    #     forward_states=Track()
    #     prior=self.X_prior

    #     for t, measurement in enumerate(self.measurements):
    #         timestamp=measurement.timestamp
    #         K_index=np.argmax(K_list[t].state_vector)
    #         transition_model=self.conditional_models['transition'][K_index] 
    #         measurement_model=self.conditional_models['measurement'][K_index]
    #         predictor = KalmanPredictor(transition_model)
    #         updater = KalmanUpdater(measurement_model=measurement_model)
    #         prediction = predictor.predict(prior, timestamp=timestamp)
    #         hypothesis = SingleHypothesis(prediction, measurement) 
    #         post = updater.update(hypothesis)
    #         forward_states.append(post)
    #         prior = forward_states[-1]
    #     return forward_states
    
    # def parameter_posterior(self, parameters, parameter_index):
    #     prior = self.parameter_priors[parameter_index](parameters)
    #     likelihood = self.parameter_likelihood_functions[parameter_index](parameters, self.measurements, self.generated_states)
    #     posterior = prior * likelihood
    #     return posterior

    # def update_parameters(self, parameter_index):
    #     """
    #     Update the parameter at the given index by sampling from its posterior distribution.
        
    #     Args:
    #         parameter_index (int): Index of the parameter to update (0 for sigma2, 1 for tau2).
    #     """
    #     updated_parameter = self.parameter_posteriors[parameter_index](
    #         self.measurements,
    #         self.generated_states,
    #         self.tau_likelihood_term
    #     )        
    #     new_parameters = list(self.parameters[-1])
    #     new_parameters[parameter_index] = updated_parameter
    #     self.parameters.append(tuple(new_parameters))
    #     self.conditional_models=self.conditional_models_funct(new_parameters,self.C)


# def particle_paths_to_gaussian_paths(track, max_lag=None,   **kwargs):
#     """
#     Converts particle tracks into GaussianStateUpdate tracks for Kalman smoothing.

#     Parameters
#     ----------
#     lag_length : int, optional
#         Specifies how far back in time to consider for smoothing (default: None).

#     Returns
#     -------
#     list of Tracks
#         A list of GaussianStateUpdate tracks corresponding to the particle tracks.
#     """
      
#     particlesmoother=ParticleSmoother(track)
#     particle_paths=particlesmoother.smooth(max_lag, **kwargs)
#     #Outdated!!!
#     F = track[0].hypothesis.prediction.linearised_transition_model
#     transition_model=LinearGaussianTimeInvariantTransitionModel(
#                     transition_matrix=F,
#                     covariance_matrix=None
#                 )
    
#     gaussian_track_list = [Track() for path in particle_paths]

#     for i,path in enumerate(particle_paths):
#         gaussian_state=None
#         for t, state in enumerate(path):
#             timestamp=state.timestamp
#             particle_prediction=state.hypothesis.prediction

#             prediction = GaussianStatePrediction(
#                 state_vector=particle_prediction.state_vector,
#                 covar=particle_prediction.covariance,
#                 timestamp=timestamp,
#                 transition_model=transition_model,
#                 prior=gaussian_state
#             )

#             gaussian_state=GaussianStateUpdate(
#                 state_vector=state.state_vector,
#                 covar=state.covariance,
#                 timestamp=timestamp,
#                 #need to change so that other hypotheses types are allowed.
#                 hypothesis=SingleHypothesis(measurement=state.hypothesis.measurement,
#                                             prediction=prediction)
#             )
#             gaussian_track_list[i].append(gaussian_state)

#     return gaussian_track_list 

# def particle_paths_separated(self, track=None, max_lag=None, **kwargs):
#     # max_lag is how far forward we want to use data. 
#     # if proof of independence some steps forward, we can use this to
#     # avoid excessively long lag times which only weight a single particle
#     """
#     Generate particle paths using precomputed indices.

#     Parameters
#     ----------
#     earliest_t : int
#         The earliest timestep to consider.
#     final_timestep : int, optional
#         The final timestep to consider. Defaults to the last timestep in the track.

#     Returns
#     -------
#     tuple
#         - List of Tracks for each particle.
#         - 2D array of indices for each particle at each timestep.
#     """
#     if track is None:
#         track=self.track

#     num_particles = len(track[0])
#     track_length = len(track)
#     particle_track_list = [Track() for _ in range(num_particles)]
    
#     #TODO: need to fix this max_lag logic as currently doesn't work with Kalman smoother 
#     # SHOULD CHECK: if we randomise the 'prev_index' based on the number of times particles were resampled
#     # will that mess things up? 
#     if max_lag is None: #if unspecified, take all future data into account
#         max_lag = track_length - 1

#     # Build the particle tracks using the indices
#     for t in range(track_length):
#         state_t = track[t]
#         prediction=state_t.hypothesis.prediction

#         # Recalculate particle indices array to reintroduce culled particles
#         if t + max_lag <= track_length - 1:
#             final_timestep = t + max_lag
#             particle_indices,_ = self.get_particle_track_indices(track=track,earliest_t=t, final_timestep=final_timestep)
        
#         current_indices=[]
#         for i in range(num_particles):
#             current_index = particle_indices[i, t]
#             current_indices.append(current_index)
#             prev_index=state_t.resample_index[current_index]
#             particle_prediction = MarginalisedParticleStatePrediction(state_vector=prediction.state_vector[...,prev_index],
#                                         covariance=prediction.covariance[...,prev_index],
#                                         timestamp=prediction.timestamp,
#                                         linear_transition_matrix=prediction.linear_transition_matrix,
#                                         process_mean=prediction.process_mean[...,prev_index],
#                                         process_covar=prediction.process_covar[...,prev_index])
            
#             hypothesis = SingleHypothesis(
#                             measurement=state_t.hypothesis.measurement,
#                             prediction=particle_prediction)
#             particle_state=MarginalisedParticleStateUpdate(
#                         state_vector=state_t.state_vector[...,current_index],
#                         covariance=state_t.covariance[...,current_index],
#                         log_weight=np.array([0]),
#                         hypothesis=hypothesis,
#                         timestamp=state_t.timestamp)
            
#             particle_track_list[i].append(particle_state)
#     return particle_track_list

# def collate_tracks(self,track_list,max_lag=None,**kwargs):
#     """
#     Compute the combined track from a list of tracks.
#     Parameters
#     ----------
#     track_list : list of Tracks
#         List of tracks, where each track is a list of states.
#     Returns
#     -------
#     Track
#         A track representing the combined input of all input tracks.
#     """

#     track_length = len(track_list[0])
#     if any(len(track) != track_length for track in track_list):
#         raise ValueError("All tracks must have the same length.")
#     if max_lag is None: #if unspecified, take all future data into account
#         max_lag = track_length - 1

#     num_tracks=len(track_list)
#     combined_track = Track()

#     track_length = len(track_list[0])
    
#     for t in range(track_length):
        
#         states_at_t = [track[t] for track in track_list]

#         # Each state is dimension (M,) or (M,1) for vectors, and (M,M) for covariances.

#         pred_vectors = np.column_stack([st.hypothesis.prediction.state_vector for st in states_at_t])
#         pred_covars = np.stack([st.hypothesis.prediction.covariance for st in states_at_t], axis=2)
        
#         proc_means = np.column_stack([st.hypothesis.prediction.process_mean for st in states_at_t])
#         proc_covars = np.stack([st.hypothesis.prediction.process_covar for st in states_at_t], axis=2)
        
#         state_vectors = np.column_stack([st.state_vector for st in states_at_t])
#         state_covars = np.stack([st.covariance for st in states_at_t], axis=2)
        
#         # The timestamp and linear_transition_matrix can come from any one of the states,
#         # e.g. the first in the list
#         state_t = states_at_t[0]
#         timestamp = state_t.timestamp
#         linear_transition_matrix = state_t.hypothesis.prediction.linear_transition_matrix
        
#         # Wrap them in Stone Soup containers
#         particle_prediction = MarginalisedParticleStatePrediction(
#             state_vector=StateVectors(pred_vectors),         # shape (M, num_tracks)
#             covariance=CovarianceMatrices(pred_covars),      # shape (M, M, num_tracks)
#             timestamp=timestamp,
#             linear_transition_matrix=linear_transition_matrix,
#             process_mean=StateVectors(proc_means),
#             process_covar=CovarianceMatrices(proc_covars),
#             log_weight=np.array([np.log(1 / num_tracks)] * num_tracks),
#         )
        
#         hypothesis = SingleHypothesis(
#             measurement=state_t.hypothesis.measurement,
#             prediction=particle_prediction
#         )
        
#         # Finally, create the combined update state
#         combined_state = MarginalisedParticleStateUpdate(
#             state_vector=StateVectors(state_vectors),
#             covariance=CovarianceMatrices(state_covars),
#             log_weight=np.array([np.log(1 / num_tracks)] * num_tracks),
#             hypothesis=hypothesis,
#             timestamp=timestamp
#         )
        
#         combined_track.append(combined_state)

#     return combined_track
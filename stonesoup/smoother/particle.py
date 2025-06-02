#for generic PS
import copy
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.prediction import MarginalisedParticleStatePrediction
from stonesoup.types.track import Track
from stonesoup.types.update import MarginalisedParticleStateUpdate
from stonesoup.smoother.base import  Smoother
import numpy as np
from stonesoup.smoother.kalman import KalmanSmoother
class ParticleSmoother(Smoother):
    r"""
    A. Overview
    -----------
    This class implements an Particle Smoother that reconstructs
    particle paths using the resampling histories. The algorithm traces
     back through ancestors in the resampled particle set (via a "backward pass")
     and re-weights these paths based on the number of descendants in the 
     current particle generation, to approximate the joint smoothing distribution. 
     Conceptually, each smoothed trajectory is obtained by linking its sampled ancestors over time.

    B. Key Mathematical Ideas
    -------------------------
    1. Ancestor Tracing:
       Each particle i at time t is associated with an ancestor index A_{t}^{(i)} from
       the previous time t-1. In the backward pass, the trajectory for particle i is
       recovered by following these ancestor indices until t=0.

    2. Descendant-based Weighting (Approximate Joint Smoother):
       Let \(\{ x_{0:t}^{(i)}, w_t^{(i)} \}\) be the filtered particles and weights at time t.
       .. math::
         p(x_{0:t} \mid y_{0:T}) \approx \sum_{i=1}^N \widetilde{W}_T^{(i)} \, \delta\bigl(x_{0:t} - x_{0:t}^{(i)}\bigr),

       where \(\widetilde{W}_T^{(i)}\) are "descendant-based" weights derived from
       tracing how many descendant particles lead from i at earlier times. See [1]_
       for details of such "backward smoothing" or "two-pass" particle smoothers.

    3. Path Merging and Degeneracy:
       Because of repeated resampling, many paths eventually coalesce to the same
       ancestor, which can lead to degeneracy. 

    C. References
    -------------
    .. [1] Doucet, A., Godsill, S., and Andrieu, C. 2000. "On Sequential Monte Carlo
           Sampling Methods for Bayesian Filtering." *Statistics and Computing,*
           10(3):197-208.
    .. [2] Cappe, O., Godsill, S.J., and Moulines, E. 2007. "An Overview of Existing
           Methods and Recent Advances in Sequential Monte Carlo," *Proceedings of
           the IEEE,* 95(5):899-924. (See smoothing discussion and figures)
    """

    def smooth(self,track):
        return self.particle_paths(track)
    
    def _prediction(self, state):
        """
        A. Comments
        -----------
        Retrieves the prediction from the state if it is a prediction type or
        from the hypothesis if the state is an update. This is used to find the
        forward predictive distribution required in the backward recursion.

        B. MATHS
        --------
        The forward prediction is p(x_k | x_{k-1}, ...) in a linear-Gaussian form,
        typically :math:`\mathcal{N}(F_{k} x_{k-1}, Q_k)` if static, or from
        conditional processes if dynamic.

        """
        if isinstance(state, MarginalisedParticleStatePrediction):
            return state
        elif isinstance(state, MarginalisedParticleStateUpdate):
            if isinstance(state.hypothesis, MultipleHypothesis):
                predictions = {hypothesis.prediction for hypothesis in state.hypothesis}
                if len(predictions) == 1:
                    return predictions.pop()
                else:
                    raise ValueError(
                        "Track has MultipleHypothesis updates with multiple predictions.")
            else:
                return state.hypothesis.prediction
        else:
            raise TypeError("States must be MarginalisedParticlePredictions or MarginalisedParticleUpdates.")

    def particle_paths(self, track):
        r"""
        A. Comments
        -----------
        Create a single :class:`Track` containing all MarginalisedParticleStates
        for the times in the provided track, essentially culling particles 
        from which no descendants have reached the most recent particle state, 
        and reordering the Track so each index of the State variable 
        corresponds to an ancestor path. This merges each step's states in a single pass,
        reordering or matching them according to the stored resample indices.

        B. MATHS
        --------
        Let \(particle\_indices[:, t]\) be the list of ancestor indices for each
        particle at time t. Then for each t, the state vectors and covariances
        are aligned so that
        \[
            combined\_state[i] = x_{t}^{( idx_{t}[i] )},
        \]

        C. References
        -------------
        See [2] (Section on smoothing).
        """

        track_length = len(track)
        num_particles = len(track[0])
        particle_indices, prior_idx_t = self.get_particle_track_indices(track=track, earliest_t=0)
        combined_track = Track()
        prev_idx_t = prior_idx_t

        for t in range(track_length):
            # Gather the 'resample' indices for all particles at time t
            idx_t = particle_indices[:, t].astype(int)  # shape (num_particles,)
            state_t = track[t]
            timestamp = state_t.timestamp
            prediction = self._prediction(state_t)

            reordered_prediction = MarginalisedParticleStatePrediction(
                state_vector=prediction.state_vector[..., prev_idx_t], 
                covariance=prediction.covariance[..., prev_idx_t],
                timestamp=prediction.timestamp,
                linear_transition_matrix=prediction.linear_transition_matrix,
                process_mean=prediction.process_mean[..., prev_idx_t],
                process_covar=prediction.process_covar[..., prev_idx_t],
                log_weight=np.array([np.log(1 / num_particles)] * num_particles)
            )
            if type(state_t)==MarginalisedParticleStatePrediction:
                combined_track.append(reordered_prediction)
            else:
                hypothesis = copy.copy(state_t.hypothesis)
                hypothesis.prediction=reordered_prediction

                combined_state = MarginalisedParticleStateUpdate(
                    state_vector=state_t.state_vector[..., idx_t], # shape (M, num_particles)
                    covariance=state_t.covariance[..., idx_t],     # shape (M, M, num_particles)
                    log_weight=np.array([np.log(1 / num_particles)] * num_particles),
                    hypothesis=hypothesis,
                    timestamp=timestamp
                )

                combined_track.append(combined_state)
            prev_idx_t=idx_t

        return combined_track
    
    def get_particle_track_indices(self,track, earliest_t=0, final_timestep=None, store_particle_nums=False,**kwargs):
        r"""
        A. Comments
        -----------
        Extract the array of ancestor/resample indices for each particle across time.
        By walking backward in time from a final_timestep to earliest_t, we link each
        particle to the one it descended from at the previous time.

        B. MATHS
        --------
        Let \(\mathrm{resample\_index}[t]\) be the array of size (num\_particles) storing
        each particle's ancestor. Then:
        \[
           \text{particle\_indices}[i, t-1]
           = \mathrm{resample\_index}[t][\, \text{particle\_indices}[i, t] \,].
        \]
        This is iterated backward from t = final_timestep down to earliest_t.
        The ability to choose final_timestep is in hopes of a subsequent fixed-lag
        smoother implementation for which the variable would be useful.
        """
        track_length = len(track)
        if final_timestep is None or final_timestep >= track_length: 
            final_timestep = track_length - 1

        num_particles = len(track[0])
        particle_indices = np.full((num_particles, track_length), 
                                   num_particles, # Will throw index error if unassigned
                                   dtype=int) 

        # Initialize the indices for the last timestep
        particle_indices[:, final_timestep] = np.arange(num_particles)

        # Fill indices backward in time
        for t in range(final_timestep - 1, earliest_t - 1, -1):
            try:
                particle_indices[:, t] = track[t + 1].resample_index[particle_indices[:, t + 1]]
            except AttributeError:
                # Fallback when resample_index doesn't exist (i.e. it's a prediction)
                particle_indices[:, t] = particle_indices[:, t + 1]
                        
        if earliest_t > 0:
            prior_indices = None
        else:
            try:
                prior_indices = track[0].resample_index[particle_indices[:, 0]]
            except AttributeError:
                # Fallback when resample_index doesn't exist (i.e. it's a prediction)
                 prior_indices = particle_indices[:, 0]
            
        
        return particle_indices, prior_indices


class MarginalisedKalmanSmoother(ParticleSmoother, KalmanSmoother):
    r"""
    A. Comments
    -----------
    This class performs a Marginalised Kalman Smoother, sometimes referred to
    as an RTS Smoother for conditionally linear-Gaussian state evolution. By
    viewing the model as conditionally Gaussian given certain particles, the
    linear updates are applied in a backward recursion (the standard RTS approach).

    B. MATHS
    --------
    The backward recursion from the standard Kalman Smoother is given by:
    .. math::
       \mathbf{G}_k &= \mathbf{P}_{k|k} \mathbf{F}_k^T \mathbf{P}_{k+1|k}^{-1}, \\
       \mathbf{x}_{k|T} &= \mathbf{x}_{k|k} + \mathbf{G}_k \bigl(\mathbf{x}_{k+1|T} - \mathbf{x}_{k+1|k}\bigr), \\
       \mathbf{P}_{k|T} &= \mathbf{P}_{k|k} + \mathbf{G}_k \bigl(\mathbf{P}_{k+1|T} - \mathbf{P}_{k+1|k}\bigr) \mathbf{G}_k^T.

    In this implementation, the conditional predictions (the F, Q, etc.) can be
    retrieved from the particle-based forward pass, and the backward pass is done
    with these standard recursions as in the KalmanSmoother Class.

    C. References
    -------------
    .. [1] Rauch, H. E., Tung, F., and Striebel, C.T. 1965. "Maximum Likelihood
           Estimates of Linear Dynamic Systems," *AIAA Journal,* 3(8):1445-1450.
    .. [2] "The Levy state space model" (See section on conditional Gaussian
           modelling and smoothing)
    """

    def _transition_matrix(self, prediction):
        """
        A. Comments
        -----------
        Obtain or fallback to a stored transition matrix from the conditional
        prediction if not static. If the transition matrix is absent, raise error.
        """
        transition_matrix = getattr(prediction, "linear_transition_matrix", None)
        if transition_matrix is None:
            transition_matrix = self.transition_matrix
        if transition_matrix is None:
            raise ValueError('neither input transition matrix, nor one attached to prediction')
        return transition_matrix

    def _smooth_gain(self, state, prediction, **kwargs):
        r"""
        A. Comments
        -----------
        Compute the smoothing gain, commonly denoted as \(G_k\), used in the
        Rauch-Tung-Striebel backward pass step. The covariance terms come from
        the forward pass (both the posterior at time t and the prior at t+1).

        B. MATHS
        --------
        .. math::
          \mathbf{G}_k = \mathbf{P}_{k|k} \mathbf{F}_k^T \bigl(\mathbf{P}_{k+1|k}\bigr)^{-1}.
        """
        covar_state = np.moveaxis(state.covariance, 2, 0)   # (N, M, M)
        F = self._transition_matrix(prediction) #MxM
        Ft = F.T #MxM
        covar_pred = np.moveaxis(prediction.covariance, 2, 0)  # (N, M, M)
        covarF = np.einsum("nij,jk->nik", covar_state, Ft) # (N, M, M)
        covar_pred_inv = np.linalg.inv(covar_pred) # (N, M, M)
        ksmooth_gain = np.einsum("nij,njk->nik", covarF, covar_pred_inv) # (N, M, M)
        return np.moveaxis(ksmooth_gain, 0, 2)

    def smooth(self, track=None, culled_track=None, **kwargs):
        r"""
        A. Comments
        -----------
        Runs the backward RTS-based smoothing pass on a track of MarginalisedParticle
        states. First, it generates a "culled" track using the `particle_paths()`
        logic from the parent class, then executes the standard backward recursion.

        B. MATHS
        --------
        The standard backward recursion for t = T-1..0:

        .. math::
          \mathbf{x}_{t|T} = \mathbf{x}_{t|t} + \mathbf{G}_t \left(\mathbf{x}_{t+1|T} - \mathbf{x}_{t+1|t}\right),
          \mathbf{P}_{t|T} = \mathbf{P}_{t|t} + \mathbf{G}_t \left(\mathbf{P}_{t+1|T} - \mathbf{P}_{t+1|t}\right) \mathbf{G}_t^T.

        where \(\mathbf{G}_t\) is given by :meth:`_smooth_gain`.

        C. References
        -------------
        .. [1] Rauch, H. E., Tung, F., and Striebel, C.T. (1965). "Maximum likelihood
               estimates of linear dynamic systems." *AIAA Journal*, 3(8):1445-1450.
        .. [2] S. Godsill, M. Riabiz, and I. Kontoyiannis, "The Lévy State Space Model," 
        Department of Engineering, University of Cambridge, 2020 for the conditional approach to Gaussian updates.
        """
        culled_track = ParticleSmoother().particle_paths(track=track)

        track=culled_track #rename culled_track variable for simplicity/ to avoid confusion. 

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
            # Retrieve prediction
            prediction = self._prediction(subsq_state)
            ksmooth_gain = self._smooth_gain(state, prediction, time_interval=time_interval, **kwargs)
            
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
            #clean up diff state type handling
            hypothesis = getattr(state, 'hypothesis', None)
            if hypothesis:
                subsq_state = type(state).from_state(state, 
                                                state_vector=smooth_mean, 
                                                covariance=smooth_covar,
                                                hypothesis=hypothesis,
                                                timestamp=state.timestamp)
            else:
                subsq_state = type(state).from_state(state, 
                                                state_vector=smooth_mean, 
                                                covariance=smooth_covar,
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
    Carter-Kohn (1994) Backward Sampling Algorithm
    ----------------------------------------------

    This algorithm efficiently samples state sequences in linear Gaussian
    state-space models using a backward-sampling method within a Gibbs sampling
    framework.

    A. Comments
    ------------
    - Implements the Carter-Kohn (1994) Gibbs-type backward-sampling-based
    smoother. This approach draws entire state sequences by sampling from
    p(x_{t} | x_{t+1}, y_{1:T}).
    - Uses matrix decompositions to directly sample states for linear or 
    conditionally linear processes.

    B. Algorithm Steps
    -------------------

    1. Initialization
    ------------------
    # Perform Kalman filter forward pass to obtain mean and covariance estimates:
        x_t[i] = E(x_t | Y*, x_{t+1}, ..., x_n)
        S_t[i] = Var(x_t | Y*, x_{t+1}, ..., x_n)

    2. Construct Future Observations
    ---------------------------------
    # Augment observations:
        x_t(i) = F_{t+1} x_t(i) + u_{t+1}(i)
    Where:
        - u_{t+1}(i) ~ N(0, Δ_{t+1}) (Δ_{t+1} diagonal)
        - Elements of u_{t+1}(i) are independent

    3. Recursive Kalman Update for Each Step
    ----------------------------------------
    For i = 1, ..., m:

    # Compute auxiliary variables:
        a_t(i) = S_{t+1}(i) F_{t+1} Y S_{t|t-1}^{-1} (x_t(i) - μ_{t|t-1})
        R_t(i) = F_{t+1} Y S_{t|t-1}^{-1} F_{t+1}^T + Δ_{t+1}

    4. Update Mean and Covariance
    ------------------------------
    # Compute smoothed mean and covariance:
        x_t(i) = x_t(i) + S_{t|t-1} F_{t+1}^T R_t(i)^{-1} a_t(i)
        S_t(i) = S_{t|t-1} - S_{t|t-1} F_{t+1}^T R_t(i)^{-1} F_{t+1} S_{t|t-1}

    5. Final Distribution
    ----------------------
    # At the terminal time step, the smoothed distribution is:
        x_t(i) ~ N(μ_{t|T}, Σ_{t|T})

    Where:
        μ_{t|T} = μ_{t|t} + P_{t|t} F_{t+1}^T P_{t+1|t}^{-1} (x_{t+1} - μ_{t+1|t})
        Σ_{t|T} = P_{t|t} - P_{t|t} F_{t+1}^T P_{t+1|t}^{-1} F_{t+1} P_{t|t}

    6. Sampling
    ------------
    # Sample from the smoothed conditional Gaussian distribution:
        x_t ~ N(μ_{t|T}, Σ_{t|T})

    C. Key Intuition
    -----------------
    - The backward pass reconstructs the state sequence by:
        - Using Kalman filtering principles for recursive updates
        - Leveraging conditional independence properties to simplify calculations
    - Efficiently computes state sequences in linear-Gaussian models, making it ideal
    for large-scale or complex state-space models.

    D. References
    --------------
    - Carter, C.K., and Kohn, R. (1994). "On Gibbs Sampling for State Space Models,"
    Biometrika, 81(3):541-553.
    """

    def smooth(self, track=None, measurements=None, **kwargs):
        r"""
        A. Comments
        -----------
        Carries out the vectorized backward pass from Carter & Kohn (1994),
        sampling each time slice (in a backward manner) from its conditional
        distribution given x_{t+1}, y_{1:T}. If the underlying model or
        forward pass is conditionally Gaussian, these draws are feasible
        through Cholesky / matrix manipulations. This effectively "draws"
        entire state paths from the smoothing distribution.

        B. MATHS
        --------
        Carter & Kohn propose a block backward-sampling approach:
        .. math::
           x_{t}| x_{t+1}, y \sim \mathcal{N}(m_t, C_t),
        where:
        .. math::
           m_t &= x_{t|t} + P_{t|t} F^T P_{t+1|t}^{-1} (x_{t+1} - x_{t+1|t}),\\
           C_t &= P_{t|t} - P_{t|t} F^T P_{t+1|t}^{-1} F P_{t|t}.
        The code includes the typical vectorization for batch form.

        C. References
        -------------
        .. [1] Carter, C.K. and Kohn, R. (1994). "On Gibbs Sampling for State Space
               Models," *Biometrika*, 81(3):541-553.
        """

        culled_track=ParticleSmoother().particle_paths(track=track)

        track=culled_track #rename culled_track variable for simplicity/ to avoid confusion. 

        if measurements is None:
            measurements = []
            for state in track: 
                measurement = state.hypothesis.measurement
                measurements.append(measurement)

        smoothed_track = Track()

        final_state = track[-1]
        smoothed_track.append(final_state)
        subsq_state = final_state

        x_t_plus_1 = subsq_state.state_vector
        m, ncols = x_t_plus_1.shape

        self.sigma_residuals_term = 0
        self.tau_likelihood_term = 0

        for t in range(len(track) - 2, -1, -1):
            current_state = track[t]
            time_interval = subsq_state.timestamp - current_state.timestamp

            x_t_given_t = current_state.state_vector.copy()   # (M, N)
            
            # Attempt to get the forward prediction
            # (If normal Stone Soup usage, this is in subsq_state.hypothesis.prediction)
            try:
                prediction_t_1 = self._prediction(subsq_state)
                F_t_1 = prediction_t_1.linear_transition_matrix  # (M, M)
                U_t_1 = prediction_t_1.process_covar            # (M, M, N)
                S_t_given_t = current_state.covariance.copy()   # (M, M, N)
                process_mean = prediction_t_1.process_mean       # (M, N)
            except:
                # Fallback if it's a direct transition model
                transition_model = self._prediction(subsq_state).transition_model
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
from stonesoup.types.track import Track
from stonesoup.types.update import MarginalisedParticleStateUpdate
from stonesoup.types.state import MarginalisedParticleState
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle
from .base import Property, Smoother
import numpy as np


    #TODO: NEED TO MODULARISE AND GENERALISE:
    # need to modularise this and make it self-contained. Mainly considering 'history' implementation,
    # potentially making it a property of an update object or similar so whole process is more generalised,
    # and the smoothing process, taking transition model etc. instead of relying on history being calculated as you go along.
   
    #TODO: INCORPORATE ONLINE SMOOTHING: 
    # this would allow for online smoothing, which i want to be incorporated.

class ParticleSmoother(Smoother):
    """
    Approximate Particle Smoother to reconstruct particle paths using the resampling history.
    This smoother traces back the origin of each particle using resampling indices and smooths 
    the resulting paths by calculating descendant-based weights.
    """
    
    def smooth(self, track, history, smooth_type='descendant', lag_length=1, window_length=1, constant_lag=None, keep_duplicates=None, **kwargs):
        """
        Perform smoothing on particle tracks based on descendant counts and weighted averages.

        Parameters
        ----------
        track : list of ParticleState
            List of particle states over time.
        history : dict
            Dictionary containing the 'resample_indices' which store the resampling history.
        on_off_line : str, optional
            Mode of operation ('off' for offline smoothing, 'on' for online).
        smooth_length : int, optional
            Smoothing lag length (L) to determine how far into the future to look.
        window_length : int, optional
            Window length (W) to smooth subsequences of particles.
        keep_duplicates : bool, optional
            If False, prevents duplicate paths.
        keep_constant_lag : bool, optional
            If True, raises an error if the smoothing lag cannot be maintained.

        Returns
        -------
        Track
            A smoothed track containing weighted averaged states.
        """
        if smooth_type == 'descendant':
            # Step 1: Perform offline smoothing using the assigned weights
            return self.descendant_smooth(track, history,lag_length=lag_length, window_length=window_length, constant_lag=constant_lag, **kwargs)
        
        elif smooth_type == 'particle_smoother':
            return self.particle_smoother(track=track, history=history,lag_length=lag_length, window_length=window_length, constant_lag=constant_lag, **kwargs)
        
        else:
            raise ValueError("Invalid mode. Use 'off' for offline or 'on' for online smoothing.")
    
    def particle_smoother(self, track, history,lag_length, window_length, constant_lag, **kwargs):

        # Placeholder for future implementation of particle smoothing
        raise NotImplementedError("Particle smoother is not yet implemented.")

    def MarginalisedParticleSmoother(self):
        # Placeholder for future implementation of conidtionally gaussian smoothing using kalman filter
        raise NotImplementedError("Particle smoother is not yet implemented.")


    def descendant_smooth(self, track, history,lag_length, window_length, constant_lag, **kwargs):
        track_length=len(track)
        
        smoothed_track=Track()

        for t in range(track_length-1):
            if constant_lag is not None and t>=track_length-lag_length:
                break
            
            if constant_lag is not None:
                final_timestep=t+lag_length
            elif constant_lag is None:
                final_timestep = min(t+lag_length,track_length-1)

            state_descendants_by_index=self.get_descendant_count(track, 
                                                                 history,earliest_t=t, 
                                                                 window_length=window_length,
                                                                 final_timestep=final_timestep,
                                                                 **kwargs)

            total_descendants = sum(state_descendants_by_index[t].values())
            
            weights_array=np.array([Probability(
                state_descendants_by_index[t][i]/total_descendants) 
                for i in state_descendants_by_index[t]])
            
            
            smoothed_track.append(
                MarginalisedParticleState(
                    state_vector=track[t].state_vector,
                    covariance=track[t].covariance,
                    weight=weights_array,
                    timestamp=track[t].timestamp))
    
        smoothed_track.append(track[-1])

        if constant_lag is None:
            [smoothed_track.append(track[-1])]
        elif constant_lag is not None:
            [smoothed_track.append(track[i]) for i in range(track_length-lag_length,track_length)]
        return smoothed_track
    
    def get_descendant_count(self, track, history,earliest_t=0,final_timestep=None, window_length=1, **kwargs):
            """
            finds number of descendants for track up to index t=final_timestep. 
            Goes back to time 't=earliest_timestep' rather than t=0 to save comp time.

            Parameters
            ----------
            track : list of ParticleState
                List of particle states over time.
            history : dict
                Dictionary containing the 'resample_indices' which store the resampling history.
            
            Returns
            -------
            dict
                State memory dictionary tracking counts of particle indices at each timestep.
            """
            
            track_length = len(track)
            if final_timestep is None:
                final_timestep = track_length-1
            elif final_timestep >=track_length:
                raise ValueError("final timestep index out of bounds, must be less than track length")

            number_particles=len(track[0])
            state_descendants_by_index = {j: {i:0 for i in range(number_particles)} for j in range(earliest_t,final_timestep)} #state_descendants_by_index= {t: {index_i:count of descendants at index, for all i}, for all 0<t<track_length-1}

            for i in range(number_particles):
                current_particle_index = i
                t = final_timestep
                while t > earliest_t:
                    new_particle_index = history['resample_indices'][t][0][current_particle_index]                               
                    state_descendants_by_index[t-1][new_particle_index] +=1
                    current_particle_index = new_particle_index
                    t-=1

            return state_descendants_by_index     
    
    def get_particle_paths(self, track, history, keep_duplicates=None,**kwargs):
        """
        Extract particle paths based on resampling history and generate the state memory for smoothing.
        Parameters
        ----------
        track : list of ParticleState
            List of particle states over time.
        history : dict
            Dictionary containing the 'resample_indices' which store the resampling history.
        keep_duplicates : bool, optional
            If False, prevents duplicate paths.

        Returns
        -------
        list
            A list of lists containing particle tracks, ending at t=T but of varying lengths (if keep_duplicates=None) so index backwards
            A list of lists containing particle indices,  ending at t=T but of varying lengths (if keep_duplicates=None) so index backwards
        """
        particle_track_list = []
        particle_track_indices_list = []
        track_length = len(track)
        number_particles=len(track[0])
        #states_by_index equals {t: {index_i:count of particles at index, for all i}, for all t}
        states_by_index = {j: [] for j in range(track_length)}
        for i in range(number_particles):
            ith_particle_track = Track()
            current_particle_index = i
            ith_particle_track_indices = []
            t = track_length

            # if at first set of particles in track (at t=0), parent particles are a prior, not a post (see notebook implementation), so won't be in track(...)
            # this logic is only given track.append(post). if you added the first prior it'd be different, so generalisation needs serious thought
            while t >= 1:
                t-=1
                state = track[t]
                
                # if at first set of particles in track (at t=0), parent particles are a prior, not a post (see notebook implementation), so won't be in track(...)
                # this logic is only given track.append(post). if you added the first prior it'd be different, so generalisation needs serious thought
                
                states_by_index[t].append(current_particle_index)
                    
                #only add particle to list if not in there already, or if keeping every particle in every list
                ith_particle_track.insert(0, 
                                          MarginalisedParticleState(
                                              state_vector=state[current_particle_index].state_vector,
                                              covariance=state.covariance[..., current_particle_index],
                                              weight=np.array([Probability(1)]),
                                              timestamp=state.timestamp
                                        ))

                ith_particle_track_indices.insert(0, current_particle_index)

                new_particle_index = history['resample_indices'][t][0][current_particle_index]
                
                if t>=1:
                    parent_already_visited=False
                    if new_particle_index in states_by_index[t-1]:
                        parent_already_visited = True
                    if parent_already_visited and keep_duplicates is None:
                        break
                    current_particle_index = new_particle_index


            #TODO: figure out how to pad the Track variable to enable more intuitive (forward) indexing. currently method is just to keep duplicates.
            # # make the particle tracks the same length as the total track to make forward indexing easier
            # len_left= track_length-len(ith_particle_track)
            # ith_particle_track = ith_particle_track #figure out how to pad Track variable
            # ith_particle_track_indices = [None] * len_left + ith_particle_track_indices

            particle_track_list.append(ith_particle_track)
            particle_track_indices_list.append(ith_particle_track_indices)

        return particle_track_list,particle_track_indices_list

    def modularised_get_particle_history(measurements,prior,predictor,hypothesis_type,updater,**kwargs):
        #TODO MODULARISE: need to generalise so that can get history without also needing to generate track as-you-go
        #TODO GENERALISE: some classes if defined differently to 'defaults', would need more/fewer arguments, and would thus raise errors.
        track = Track()
        history=True

        for measurement in measurements:
            prediction = predictor.predict(prior, timestamp=measurement.timestamp)
            hypothesis = hypothesis_type(prediction, measurement)
            post, history = updater.update(hypothesis, history=history)
            track.append(post)
            prior = track[-1]

        return track, history
        # raise(NotImplementedError)

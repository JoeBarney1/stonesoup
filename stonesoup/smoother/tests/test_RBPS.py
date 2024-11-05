import copy
from cv2 import invert
import numpy as np
from functools import partial

from ..base import Smoother
from ...base import Property
from ...models.base import LinearModel
from ...models.transition.base import TransitionModel
from ...models.transition.linear import LinearGaussianTransitionModel
from ...types.multihypothesis import MultipleHypothesis
from ...types.prediction import GaussianStatePrediction
from ...types.update import GaussianStateUpdate
from ...functions import gauss2sigma, unscented_transform

def get_past_weights(Particle):
        # find parent particle
        # parent particle.weight on to the START of the sequence
        # repeat on particle until Particle.parent = None

        ##need to work on below if recursive method interesting
        # weights= []
        # while Particle != None:
        #     weights.append(Particle.weight)
        #     print(weights[-1])
        #     Particle=Particle.parent
        # return reversed(weights)

        # final_states= predictions[-1] #go through the set of particle predictions at the most recent timestamp 
        # past_weights_all_particles =[] # prepare a list to hold w_t_i, the weights over time of each particle
        # for particle_i in final_states:
        #         past_weights_particle_i= get_past_weights(particle_i)
        #         # print(get_past_weights(particle_i))
        #         past_weights_all_particles.append(past_weights_particle_i)

        raise NotImplementedError

class RBParticleSmoother(Smoother):
    r"""
    Rao-Blackwellized Particle Smoother for conditionally linear Gaussian models. This class
    implements the RBPS algorithm, incorporating both hierarchical and mixed linear/non-linear
    models as transition models.
    """

    # Transition models to be defined
    transition_model_hierarchical = Property(doc="Hierarchical transition model.")
    transition_model_mixed = Property(doc="Mixed linear/non-linear transition model.")

    def forward_filter(self, track):
        """
        Perform the forward filtering using Rao-Blackwellized particle filter (RBPF).
        
        Parameters
        ----------
        track : :class:`~.Track`
            The input track to apply the forward filter.

        Returns
        -------
        list
            List of forward-filtered states
        """
        forward_states = []
        # TODO: Implement RBPF forward pass here
        # This involves particle generation and Kalman filtering for the linear states
        return forward_states

    def backward_simulation(self, forward_states):
        """
        Perform the backward simulation on forward-filtered states.
        
        Parameters
        ----------
        forward_states : list
            List of forward-filtered states.

        Returns
        -------
        list
            List of sampled backward trajectories for nonlinear states.
        """
        backward_trajectories = []
        # TODO: Implement backward sampling pass here
        # Sample nonlinear states based on forward-filtered states
        return backward_trajectories

    def smooth_linear_states(self, backward_trajectories, forward_states):
        """
        Smooth the linear states conditionally on backward-simulated nonlinear states.
        
        Parameters
        ----------
        backward_trajectories : list
            List of backward-simulated nonlinear state trajectories.
        forward_states : list
            List of forward-filtered states.

        Returns
        -------
        list
            List of smoothed states for both linear and nonlinear components.
        """
        smoothed_states = []
        # TODO: Implement smoothing for linear states conditionally on backward trajectories
        # Re-run Kalman filter conditionally on backward-simulated trajectories
        return smoothed_states

    def smooth(self, track, **kwargs):
        """
        Apply the RBPS algorithm to smooth the track.
        
        Parameters
        ----------
        track : :class:`~.Track`
            The input track to smooth.

        Returns
        -------
         : :class:`~.Track`
            Smoothed track
        """
        # Forward filtering step
        forward_states = self.forward_filter(track)

        # Backward simulation step
        backward_trajectories = self.backward_simulation(forward_states)

        # Smooth linear states with respect to backward trajectories
        smoothed_states = self.smooth_linear_states(backward_trajectories, forward_states)

        # Deep copy the track with smoothed states
        smoothed_track = copy.deepcopy(track, {id(track.states): smoothed_states})
        return smoothed_track

    def _transition_model(self, model_type):
        """ Choose the transition model based on model type.

        Parameters
        ----------
        model_type : str
            Type of model ("hierarchical" or "mixed").

        Returns
        -------
        : :class:`~.TransitionModel`
            The chosen transition model.
        """
        if model_type == "hierarchical":
            return self.transition_model_hierarchical
        elif model_type == "mixed":
            return self.transition_model_mixed
        else:
            raise ValueError("Invalid model type. Choose 'hierarchical' or 'mixed'.")


import weakref
from typing import Sequence

from stonesoup.types.numeric import Probability
from stonesoup.base import Property
from stonesoup.types.array import StateVector
from stonesoup.types.base import Type


class Particle(Type):
    """
    Particle type

    A particle type which contains a state and weight
    """
    state_vector: StateVector = Property(doc="State vector")
    weight: float = Property(doc='Weight of particle')
    parent: 'Particle' = Property(default=None, doc='Parent particle')
    history: list = Property(default=list, doc='History of previous weights')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize history directly as an instance attribute
        self.history = []
        
        if self.parent and self.parent.parent:
            self.parent.parent = weakref.ref(self.parent.parent)
        if self.state_vector is not None and not isinstance(self.state_vector, StateVector):
            self.state_vector = StateVector(self.state_vector)
        
    @property
    def ndim(self):
        return self.state_vector.shape[0]

    @parent.getter
    def parent(self):
        if isinstance(self._property_parent, weakref.ReferenceType):
            return self._property_parent()
        else:
            return self._property_parent
        
    #define a function to update the history of a particle given new weights
    # def update_history(self):
    #         """
    #         Update the weight of the particle and add the previous weight to the history.
            
    #         Parameters
    #         ----------
    #         new_weight : Probability
    #             The new weight to assign to the particle.
    #         """
    #         if self.weight is not None:
    #             self.history.append(self.weight)  # Store the current weight in history before updating
print(dir(Particle))
class MultiModelParticle(Particle):
    """
    Particle type

    A MultiModelParticle type which contains a state, weight and the dynamic_model
    """
    dynamic_model: int = Property(doc='Assigned dynamic model')
    parent: 'MultiModelParticle' = Property(default=None, doc='Parent particle')


class RaoBlackwellisedParticle(Particle):
    """
    Particle type

    A RaoBlackwellisedParticle type which contains a state, weight, dynamic_model and
    associated model probabilities
    """
    model_probabilities: Sequence[float] = Property(
        doc="The dynamic probabilities of changing models")
    parent: 'RaoBlackwellisedParticle' = Property(default=None, doc='Parent particle')

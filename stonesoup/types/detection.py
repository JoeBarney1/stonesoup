from typing import MutableMapping, Sequence

from .groundtruth import GroundTruthPath
from .state import CategoricalState, CompositeState
from .state import State, GaussianState, StateVector
from ..base import Property
from ..models.measurement import MeasurementModel


class Detection(State):
    """Detection type"""
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="The measurement model used to generate the detection (the default is ``None``)")

    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class GaussianDetection(Detection, GaussianState):
    """GaussianDetection type"""


class Clutter(Detection):
    """Clutter type for detections classed as clutter

    This is same as :class:`~.Detection`, but can be used to identify clutter
    for metrics and analysis purposes.
    """

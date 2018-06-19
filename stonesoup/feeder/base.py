"""Base classes for Stone Soup feeder"""
from ..base import Property
from ..reader import Reader, DetectionReader, GroundTruthReader
from ..buffered_generator import BufferedGenerator


class Feeder(Reader):
    """Feeder base class

    Feeder consumes and outputs :class:`.State` data and can be used to
    modify the sequence, duplicate or drop data.
    """

    reader: Reader = Property(doc="Source of detections")

    @abstractmethod
    @BufferedGenerator.generator_method
    def data_gen(self):
        raise NotImplementedError


class DetectionFeeder(Feeder, DetectionReader):
    """Detection feeder base class

    Feeder consumes and outputs :class:`.Detection` data and can be used to
    modify the sequence, duplicate or drop data.
    """

    detector = Property(DetectionReader, doc="Source of detections")

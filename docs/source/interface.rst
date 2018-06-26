Component Interfaces
====================

This sections contains the base classes for the different components within the
Stone Soup Framework.

Enabling Components
-------------------
.. autoclass:: stonesoup.detector.Detector
    :inherited-members:
.. autoclass:: stonesoup.feeder.Feeder
    :inherited-members:
.. autoclass:: stonesoup.metricgenerator.MetricGenerator
    :noindex:
.. autoclass:: stonesoup.tracker.Tracker
    :noindex:

Data Input
^^^^^^^^^^
.. autoclass:: stonesoup.reader.DetectionReader
    :noindex:
.. autoclass:: stonesoup.reader.GroundTruthReader
    :noindex:
.. autoclass:: stonesoup.reader.SensorDataReader
    :noindex:

Data Output
^^^^^^^^^^^
.. autoclass:: stonesoup.writer.MetricsWriter
    :noindex:
.. autoclass:: stonesoup.writer.TrackWriter
    :noindex:

Simulation
^^^^^^^^^^
.. autoclass:: stonesoup.simulator.DetectionSimulator
    :inherited-members:
.. autoclass:: stonesoup.simulator.GroundTruthSimulator
    :inherited-members:
.. autoclass:: stonesoup.simulator.SensorSimulator
    :inherited-members:


Algorithm Components
--------------------
.. autoclass:: stonesoup.dataassociator.DataAssociator
.. autoclass:: stonesoup.deleter.Deleter
.. autoclass:: stonesoup.hypothesiser.Hypothesiser
    :noindex:
.. autoclass:: stonesoup.gater.Gater
    :noindex:
.. autoclass:: stonesoup.initiator.Initiator
    :noindex:
.. autoclass:: stonesoup.mixturereducer.MixtureReducer
    :noindex:
.. autoclass:: stonesoup.predictor.Predictor
    :noindex:
.. autoclass:: stonesoup.resampler.Resampler
    :noindex:
.. autoclass:: stonesoup.smoother.Smoother
    :noindex:
.. autoclass:: stonesoup.updater.Updater
    :noindex:

Models
^^^^^^
.. autoclass:: stonesoup.models.control.ControlModel
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.models.measurement.MeasurementModel
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.models.transition.TransitionModel
    :inherited-members:
    :noindex:

from abc import abstractmethod

from ..base import Base, Property
from ..models.measurement import MeasurementModel


class Updater(Base):
    r"""Updater base class

    An updater is used to update the predicted state, utilising a measurement
    and a :class:`~.MeasurementModel`.  The general observation model is

    .. math::

        \mathbf{z} = h(\mathbf{x}, \mathbf{\sigma})

    where :math:`\mathbf{x}` is the state, :math:`\mathbf{\sigma}`, the
    measurement noise and :math:`\mathbf{z}` the resulting measurement.

    """

    measurement_model: MeasurementModel = Property(doc="measurement model")

    def _check_measurement_model(self, measurement_model):
        """Check that the measurement model passed actually exists. If not
        attach the one in the updater. If that one's not specified, return an
        error.

        Parameters
        ----------
        measurement_model : :class`~.MeasurementModel`
            A measurement model to be checked

        Returns
        -------
        : :class`~.MeasurementModel`
            The measurement model to be used

        """
        if measurement_model is None:
            if self.measurement_model is None:
                raise ValueError("No measurement model specified")
            else:
                measurement_model = self.measurement_model

        return measurement_model

    @abstractmethod
    def get_measurement_prediction(self, state_prediction, **kwargs):
        """Get measurement prediction from state prediction

        Parameters
        ----------
        state_prediction : :class:`~.StatePrediction`
            The state prediction

        Returns
        -------
        : :class:`~.MeasurementPrediction`
            The predicted measurement
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, prediction, measurement,
               measurement_prediction=None, **kwargs):
        """Update state using prediction and measurement.

        Parameters
        ----------
        prediction : :class:`~.StatePrediction`
            The state prediction
        measurement : :class:`~.Detection`
            The measurement
        measurement_prediction : :class:`~.MeasurementPrediction`
            The measurement prediction

        Returns
        -------
        : :class:`~.State`
            The state posterior
        """
        raise NotImplementedError

from scipy.stats import multivariate_normal as mn

from .base import Hypothesiser
from ..base import Property
from ..types import MissedDetection
from ..types.multihypothesis import \
    MultipleHypothesis
from ..types import SingleProbabilityHypothesis
from ..types.numeric import Probability
from ..predictor import Predictor
from ..updater import Updater


class PDAHypothesiser(Hypothesiser):
    """Hypothesiser based on Probabilistic Data Association (PDA)

    Generate track predictions at detection times and calculate probabilities
    for all prediction-detection pairs for single prediction and multiple
    detections.
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    clutter_spatial_density = Property(
        float,
        doc="Spatial density of clutter - tied to probability of false "
            "detection")
    prob_detect = Property(
        Probability,
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_gate = Property(
        Probability,
        default=Probability(0.95),
        doc="Gate Probability - prob. gate contains true measurement "
            "if detected")

    def hypothesise(self, track, detections, timestamp):
        r"""Hypothesise track and detection association

        For a given track and a set of N detections, return a
        MultipleHypothesis with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated probability.
        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`m, m\epsilon{1...N}`: detection m is associated
        with the track.

        The probabilities for these detections are calculated as follow:

        .. math::

          \beta_i(k) = \begin{cases}
                \frac{\mathcal{L}_{i}(k)}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=1,...,m(k) \\
                \frac{1-P_{D}P_{G}}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=0
                \end{cases}

        where

        .. math::

          \mathcal{L}_{i}(k) = \frac{\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),
          S(k)]P_{D}}{\lambda}

        :math:`\lambda` is the clutter density

        :math:`P_{D}` is the detection probability

        :math:`P_{G}` is the gate probability

        :math:`\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),S(k)]` is the likelihood
        ratio of the measurement :math:`z_{i}(k)` originating from the track
        target rather than the clutter.

        NOTE: Since all probabilities have the same denominator and are
        normalized later, the denominator can be discarded.

        References:

        [1] "The Probabilistic Data Association Filter: Estimation in the
        Presence of Measurement Origin Uncertainty" -
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5338565

        [2] "Robotics 2 Data Association" (Lecture notes) -
        http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/
        pdfs/rob2-15-dataassociation.pdf

        """

        probability_hypotheses = list()

        # items common to all SingleMeasurementHypotheses that compose
        # MultipleHypothesis
        prediction = self.predictor.predict(track.state, timestamp=timestamp)
        measurement_prediction = self.updater.get_measurement_prediction(
            prediction)

        # Missed detection hypothesis
        probability = Probability(1-(self.prob_detect * self.prob_gate))
        detection = MissedDetection(timestamp=timestamp)
        probability_hypotheses.append(
            SingleProbabilityHypothesis(
                prediction, detection,
                measurement_prediction=measurement_prediction,
                probability=probability))

        for detection in detections:
            measurement_prediction = self.updater.get_measurement_prediction(
                prediction, detection.measurement_model)

            # hypothesis that track and given detection are associated
            log_pdf = mn.logpdf(detection.state_vector.ravel(),
                                measurement_prediction.state_vector.ravel(),
                                measurement_prediction.covar)
            pdf = Probability(log_pdf, log_value=True)
            probability = (pdf * self.prob_detect)/self.clutter_spatial_density

            probability_hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction, detection,
                    measurement_prediction=measurement_prediction,
                    probability=probability))

        result = MultipleHypothesis(probability_hypotheses,
                                    normalise=True, total_weight=1)

        return result

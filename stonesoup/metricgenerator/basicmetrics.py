# -*- coding: utf-8 -*-
from itertools import chain

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import MetricGenerator
from ..base import Property
from ..models.measurement import MeasurementModel
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..types.state import State, StateMutableSequence
from ..types.time import TimeRange


class BasicMetrics(MetricGenerator):
    """Calculates simple metrics like number of tracks, truth and
    ratio of track-to-truth"""

    def compute_metric(self, manager, *args, **kwargs):
        """Compute the metric using the data in the metric manager

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)

        Returns
        ----------
        : list of :class:`~.Metric`
            Contains the metric information
        """

        metrics = []

        # Make a list of all the unique timestamps used
        timestamps = {state.timestamp for state in manager.tracks}
        timestamps |= {state.timestamp
                       for path in manager.groundtruth_paths
                       for state in path}

        # Number of tracks
        metrics.append(TimeRangeMetric(
            title='Number of targets',
            value=len(manager.groundtruth_paths),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        metrics.append(TimeRangeMetric(
            title='Number of tracks',
            value=len(manager.tracks),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        metrics.append(TimeRangeMetric(
            title='Track-to-target ratio',
            value=len(manager.tracks) / len(manager.groundtruth_paths),
            time_range=TimeRange(
                start_timestamp=min(timestamps),
                end_timestamp=max(timestamps)),
            generator=self))

        return metrics


class OSPAMetric(MetricGenerator):
    """Computes the OSPA distance for two sets of objects at each timestep that
    a state exists at.
    """
    c = Property(float, doc="Maximum distance for possible association")
    p = Property(float, doc="norm associated to distance")
    measurement_model_truth = Property(
        MeasurementModel,
        doc="Measurement model for the truth states to extract parameters to "
            "calculate distance over")
    measurement_model_track = Property(
        MeasurementModel,
        doc="Measurement model for the track states to extract parameters to "
            "calculate distance over")

    def compute_metric(self, manager):
        """Compute the metric using the data in the metric manager

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)

        Returns
        ----------
        : list of :class:`~.Metric`
            Contains the metric information at each timestamp
        """

        metric = self.process_datasets(manager.tracks,
                                       manager.groundtruth_paths)
        return metric

    def process_datasets(self, dataset_1, dataset_2):
        """Compute the OSPA distance between two datasets

        Parameters
        ----------
        dataset_1: object containing :class:`State`
        dataset_2: object containing :class:`State`

        Returns
        -------
        : list of :class:`~.Metric`
            Contains the OSPA distance at each timestamp
        """

        states_1 = self.extract_states(dataset_1)
        states_2 = self.extract_states(dataset_2)
        return self.compute_over_time(states_1, states_2)

    def extract_states(self, object_with_states):
        """Extracts a list of states from a list of (or single) object
        containing states

        Parameters
        ----------
        object_with_states: object containing a list of states
            Method of state extraction depends on the type of the object

        Returns
        ----------
        : list of :class:`~.State`
        """

        state_list = StateMutableSequence()
        for element in list(object_with_states):
            if isinstance(element, StateMutableSequence):
                state_list.extend(element.states)
            elif isinstance(element, State):
                state_list.append(element)
            else:
                raise ValueError(
                    "{!r} has no state extraction method".format(element))

        return state_list

    def compute_over_time(self, measured_states, truth_states):
        """Compute the OSPA metric at every timestep from a list of measured
        states and truth states

        Parameters
        ----------
        measured_states: list of :class:`~.State`
            Created by a filter
        truth_states: list of :class:`~.State`
            Truth states to compare against

        Returns
        -------
        TimeRangeMetric
            Covering the duration that states exist for in the parameters.
            Metric.value contains a list of metrics for the OSPA distance at
            each timestamp
        """

        # Make a sorted list of all the unique timestamps used
        timestamps = sorted({
            state.timestamp
            for state in chain(measured_states, truth_states)})

        ospa_distances = []

        for timestamp in timestamps:
            meas_points = [state
                           for state in measured_states
                           if state.timestamp == timestamp]
            truth_points = [state
                            for state in truth_states
                            if state.timestamp == timestamp]
            ospa_distances.append(
                self.compute_OSPA_distance(meas_points, truth_points))

        # If only one timestamp is present then return a SingleTimeMetric
        if len(timestamps) == 1:
            return ospa_distances[0]
        else:
            return TimeRangeMetric(
                title='OSPA distances',
                value=ospa_distances,
                time_range=TimeRange(min(timestamps), max(timestamps)),
                generator=self)

    def compute_OSPA_distance(self, track_states, truth_states):
        """Computes the OSPA distance between two sets of states

        Parameters
        ----------
        track_states: list of :class:`~.State`
        truth_states: list of :class:`~.State`

        Returns
        -------
        SingleTimeMetric
            The OSPA distance

        """

        timestamps = {
            state.timestamp
            for state in chain(truth_states, track_states)}
        if len(timestamps) != 1:
            raise ValueError(
                'All states must be from the same time to perform OSPA')

        if not track_states or not truth_states:
            distance = 0
        else:
            cost_matrix = self.compute_cost_matrix(track_states, truth_states)

            # Solve cost matrix with Hungarian/Munkres using
            # scipy.optimize.linear_sum_assignemnt
            # Length of longest set of states
            n = max(len(track_states), len(truth_states))
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Calculate metric
            distance = ((1 / n) * cost_matrix[row_ind, col_ind].sum()) ** (
                        1 / self.p)

        return SingleTimeMetric(title='OSPA distance', value=distance,
                                timestamp=timestamps.pop(), generator=self)

    def compute_cost_matrix(self, track_states, truth_states):
        """
        Creates the cost matrix between two lists of states

        Parameters
        ----------
        track_states: list of :class:`State`
        truth_states: list of :class:`State`

        Returns
        ----------
        np.ndarry
            Matrix of euclidian distance between each element in each list of
            states
        """

        cost_matrix = np.ones([len(track_states), len(truth_states)]) * self.c

        for i_track, track_state, in enumerate(track_states):
            for i_truth, truth_state in enumerate(truth_states):
                euc_distance = np.linalg.norm(
                    self.measurement_model_track.function(
                        track_state.state_vector, noise=0)
                    - self.measurement_model_truth.function(
                        truth_state.state_vector, noise=0))

                if euc_distance < self.c:
                    cost_matrix[i_track, i_truth] = euc_distance

        return cost_matrix

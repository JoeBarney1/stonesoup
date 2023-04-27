import json
import sys
import threading
from datetime import datetime, timedelta
from math import modf
from queue import Empty, Queue
from threading import Thread
from typing import Dict, List, Collection

import numpy as np
from confluent_kafka import Consumer
from dateutil.parser import parse
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader, Reader, GroundTruthReader
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


class _KafkaReader(Reader):
    topic: str = Property(doc="The Kafka topic on which to listen for messages")
    state_vector_fields: List[str] = Property(
        doc="List of columns names to be used in state vector")
    time_field: str = Property(
        doc="Name of column to be used as time field")
    time_field_format: str = Property(
        default=None, doc="Optional datetime format")
    timestamp: bool = Property(
        default=False, doc="Treat time field as a timestamp from epoch")
    metadata_fields: Collection[str] = Property(
        default=None, doc="List of columns to be saved as metadata, default all")
    kafka_config: Dict[str, str] = Property(
        default={}, doc="Keyword arguments for the underlying kafka consumer")
    buffer_size: int = Property(
        default=0,
        doc="Size of the frame buffer. The frame buffer is used to cache frames in "
            "cases where the stream generates messages faster than they are ingested "
            "by the reader. If `buffer_size` is less than or equal to zero, the buffer "
            "size is infinite.")
    timeout: bool = Property(
        default=None,
        doc="Timeout (in seconds) when reading from buffer. Defaults to None in which case the "
            "reader will block until new data becomes available.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer = Queue(maxsize=self.buffer_size)
        self._consumer = Consumer(self.kafka_config)
        self._consumer.subscribe(topics=[self.topic])
        self._non_metadata_fields = [*self.state_vector_fields, self.time_field]
        self._running = True
        self._consumer_thread = Thread(daemon=True, target=self._consume)
        self._consumer_thread.start()

    def stop(self):
        self._running = False
        self._consumer_thread.join()

    def _consume(self):
        while self._running:
            msg = self._consumer.poll(timeout=10.0)

            if msg.error():
                sys.stderr.write(f"kafka error: {msg.error()}")
            else:
                self._on_msg(msg)

    def _get_time(self, data):
        if self.time_field_format is not None:
            time_field_value = datetime.strptime(
                data[self.time_field], self.time_field_format
            )
        elif self.timestamp is True:
            fractional, timestamp = modf(float(data[self.time_field]))
            time_field_value = datetime.utcfromtimestamp(int(timestamp))
            time_field_value += timedelta(microseconds=fractional * 1e6)
        else:
            time_field_value = parse(data[self.time_field], ignoretz=True)
        return time_field_value

    def _get_metadata(self, data):
        metadata_fields = set(data.keys())
        if self.metadata_fields is None:
            metadata_fields -= set(self._non_metadata_fields)
        else:
            metadata_fields = metadata_fields.intersection(set(self.metadata_fields))
        local_metadata = {field: data[field] for field in metadata_fields}
        return local_metadata

    def _on_msg(self, msg):
        # Extract data from message
        data = json.loads(msg.value())
        self._buffer.put(data)


class KafkaDetectionReader(DetectionReader, _KafkaReader):
    """A detection reader that reads detections from a Kafka broker

    It is assumed that each message contains a single detection. The value of each message is a
    JSON object containing the detection data. The JSON object must contain a field for each
    element of the state vector and a timestamp. The JSON object may contain fields
    for the detection metadata.
    """

    @BufferedGenerator.generator_method
    def detections_gen(self):
        detections = set()
        previous_time = None
        while self._consumer_thread.is_alive():
            try:
                # Get data from buffer
                data = self._buffer.get(timeout=self.timeout)

                timestamp = self._get_time(data)
                if previous_time is not None and previous_time != timestamp:
                    yield previous_time, detections
                    detections = set()
                previous_time = timestamp

                state_vector = StateVector(
                    [[data[field_name]] for field_name in self.state_vector_fields],
                    dtype=np.float_,
                )

                detections.add(Detection(
                    state_vector=state_vector,
                    timestamp=timestamp,
                    metadata=self._get_metadata(data))
                )
            except Empty:
                yield previous_time, detections
                detections = set()


class KafkaGroundTruthReader(GroundTruthReader, _KafkaReader):
    """A ground truth reader that reads ground truths from a Kafka broker

    It is assumed that each message contains a single ground truth state. The value of each message
    is a JSON object containing the ground truth data. The JSON object must contain a field for
    each element of the state vector a timestamp. The JSON object must also contain a field for
    the path ID. The JSON object may contain fields for the ground truth metadata.
    """
    path_id_field: str = Property(doc="Name of column to be used as path ID")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._non_metadata_fields += [self.path_id_field]
        self._thread_lock = threading.Lock()
        self._groundtruth_dict = {}
        self._updated_paths = set()
        self._buffer = Queue(maxsize=self.buffer_size)

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        groundtruth_dict = {}
        updated_paths = set()
        previous_time = None
        while self._consumer_thread.is_alive():
            try:
                # Get data from buffer
                data = self._buffer.get(timeout=self.timeout)

                timestamp = self._get_time(data)
                if previous_time is not None and previous_time != timestamp:
                    yield previous_time, updated_paths
                    updated_paths = set()
                previous_time = timestamp

                # Create track state
                state = GroundTruthState(
                    StateVector([[data[field_name]] for field_name in self.state_vector_fields],
                                dtype=np.float_),
                    timestamp=timestamp,
                    metadata=self._get_metadata(data))

                # Update existing track or create new track
                path_id = data[self.path_id_field]
                try:
                    groundtruth_path = groundtruth_dict[path_id]
                except KeyError:
                    groundtruth_path = GroundTruthPath(id=path_id)
                    groundtruth_dict[path_id] = groundtruth_path

                groundtruth_path.append(state)
                updated_paths.add(groundtruth_path)
            except Empty:
                yield previous_time, updated_paths
                updated_paths = set()

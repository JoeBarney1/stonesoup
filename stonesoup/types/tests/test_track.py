import copy
import datetime

import numpy as np
import pytest

from ..detection import Detection
from ..hypothesis import SingleHypothesis
from ..numeric import Probability
from ..state import State, GaussianState, ParticleState
from ..track import Track
from ..update import Update


def test_track_empty():
    # Track initialisation without initial state
    track = Track()
    assert len(track) == 0


@pytest.mark.parametrize('state', [
    State(np.array([[0]]), datetime.datetime.now()),
    GaussianState(np.array([[0]]), np.array([[0]]), datetime.datetime.now()),
    ParticleState([[0]], datetime.datetime.now(), [Probability(1)]),
    ],
    ids=['State', 'GaussianState', 'ParticleState'])
def test_track_state(state):
    # Track initialisation with initial state
    track = Track([state])
    assert(len(track.states) == 1)
    assert(track.state == state)

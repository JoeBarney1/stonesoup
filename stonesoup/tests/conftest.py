import pytest

from ..base import Base, Property


@pytest.fixture(scope='session')
def base():
    return _TestBase

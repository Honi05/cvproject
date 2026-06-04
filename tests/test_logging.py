import logging
from cvchess.logging_utils import get_logger


def test_get_logger_returns_named_logger():
    log = get_logger("foo")
    assert isinstance(log, logging.Logger)
    assert log.name == "cvchess.foo"


def test_get_logger_is_idempotent():
    a = get_logger("bar")
    b = get_logger("bar")
    assert len(a.handlers) == len(b.handlers)  # no duplicate handlers

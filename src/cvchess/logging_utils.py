from __future__ import annotations
import logging
import sys

_CONFIGURED: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    full = f"cvchess.{name}"
    logger = logging.getLogger(full)
    if full not in _CONFIGURED:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(full)
    return logger

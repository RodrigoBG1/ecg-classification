"""Centralized logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Return a logger that writes to stdout (and optionally a file).

    Calling this multiple times with the same `name` returns the same logger
    without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    log = get_logger("test", log_file=Path("/tmp/ecg_test.log"))
    log.info("Logger initialized successfully.")
    log.warning("This is a warning.")
    print("logger.py smoke test passed.")

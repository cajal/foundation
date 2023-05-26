import logging
import sys
import os
from tqdm import tqdm as _tqdm
from functools import wraps
from contextlib import contextmanager


def get_logger():
    logger = logging.getLogger("foundation")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)-s -- %(levelname)-s -- %(filename)-10s%(lineno)4d:\t %(message)s",
        datefmt="%d-%m-%Y:%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


@contextmanager
def disable_tqdm(disable=True):
    """Context manager that disables tqdm"""

    prev = os.getenv("FOUNDATION_TQDM", "1")
    os.environ["FOUNDATION_TQDM"] = "0" if disable else prev
    try:
        yield
    finally:
        os.environ["FOUNDATION_TQDM"] = prev


@wraps(_tqdm)
def tqdm(iterable, *args, **kwargs):
    use = os.getenv("FOUNDATION_TQDM", "1")
    if int(use):
        return _tqdm(iterable, *args, **kwargs)
    else:
        return iterable

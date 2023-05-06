import os
import contextlib
import multiprocessing as mp


@contextlib.contextmanager
def multiprocess(processes=None):
    """
    Temporariliy uses specified number of processes. None uses all cpu cores.
    """
    environ = dict(os.environ)

    if processes is None:
        processes = mp.cpu_count()
    else:
        processes = min(mp.cpu_count(), processes)

    os.environ["FOUNDATION_MP"] = str(processes)

    try:
        yield

    finally:
        os.environ.clear()
        os.environ.update(environ)

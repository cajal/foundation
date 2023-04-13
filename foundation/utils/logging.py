import logging
import sys

logger = logging.getLogger("foundation")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(filename)-20s%(lineno)4d:\t %(message)s",
    datefmt="%d-%m-%Y:%H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

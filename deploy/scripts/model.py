from time import sleep
from random import randint
from foundation.fnn.model import ModelNetwork

if __name__ == "__main__":
    sleep(randint(0, 10))
    ModelNetwork.populate(reserve_jobs=True)

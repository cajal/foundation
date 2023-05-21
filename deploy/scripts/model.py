import time
import random
from foundation.fnn.model import ModelNetwork

if __name__ == "__main__":
    time.sleep(random.randint(0, 10))
    ModelNetwork.populate(reserve_jobs=True)

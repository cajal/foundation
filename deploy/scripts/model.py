import time
import random
from foundation.fnn.model import NetworkModel

if __name__ == "__main__":
    time.sleep(random.randint(0, 10))
    NetworkModel.populate(reserve_jobs=True)

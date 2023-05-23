from time import sleep
from random import randint
from foundation.fnn.model import Model, ModelNetwork


def run():
    sleep(randint(0, 10))
    ModelNetwork.populate(Model.NetworkModel, reserve_jobs=True)


if __name__ == "__main__":
    run()

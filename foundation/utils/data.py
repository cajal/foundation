import numpy as np
import pandas as pd


# -------------- Item --------------


class Item:
    """Data Item"""

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int | slice | 1D array
            used to index the data item
        """
        raise NotImplementedError()


class NpyFile(Item):
    def __init__(self, filepath, indexmap=None, transform=None):
        """
        Parameters
        ----------
        filepath : file-like object | str | pathlib.Path
            file to read
        indexmap : 1D array
            index map for the npy file
        """
        self.filepath = filepath
        self.indexmap = indexmap
        self.transform = transform

    def __getitem__(self, index):
        x = np.load(self.filepath, mmap_mode="r")
        i = index if self.indexmap is None else self.indexmap[index]
        t = np.array if self.transform is None else self.transform
        return t(x[i])

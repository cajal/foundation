import numpy as np
from PIL import Image
from .logging import logger


class Video:
    def __init__(self, frames, fixed=True):
        """
        Parameters
        ----------
        frames : Sequence[Image]
            stimulus frames
        fixed : bool
            fixed frame rate
        """
        self.frames = tuple(frames)
        self.fixed = bool(fixed)

        assert np.unique([f.mode for f in self.frames]).size == 1
        assert np.unique([f.height for f in self.frames]).size == 1
        assert np.unique([f.width for f in self.frames]).size == 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, key):
        return self.frames[key]

    @property
    def mode(self):
        """
        Returns
        -------
        str | None
            frame mode
        """
        return self.frames[0].mode

    @property
    def height(self):
        """
        Returns
        -------
        int | None
            frame height
        """
        return self.frames[0].height

    @property
    def width(self):
        """
        Returns
        -------
        int | None
            frame width
        """
        return self.frames[0].width

    @property
    def channels(self):
        """
        Returns
        -------
        int | None
            frame width
        """
        if self.mode == "L":
            return 1
        else:
            raise NotImplementedError(f"Mode {self.mode} has not yet been implemented.")

    @property
    def array(self):
        """
        Returns
        -------
        4D array | None
            shape = [frames, height, width, channels]
            dtype = np.uint8
        """
        if self.mode == "L":
            frames = np.stack([np.array(frame) for frame in self.frames], 0)
            return frames[:, :, :, None]
        else:
            raise NotImplementedError(f"Mode {self.mode} has not yet been implemented.")

    @staticmethod
    def fromarray(array, mode=None, fixed=True):
        """
        Returns
        -------
        array: 3D array | 4D array
            [frames, height, width] | [frames, height, width, channels]
        mode : bool
            frame mode
        fixed : bool
            fixed frame rate

        Returns
        -------
        Video
            new video with tranformed frames
        """
        if array.ndim == 4:
            array = array.squeeze(3)
        elif array.ndim != 3:
            raise ValueError("Array must be either 4D or 3D")

        return Video([Image.fromarray(frame, mode=mode) for frame in array], fixed=fixed)

    def apply(self, transform):
        """
        Parameters
        ----------
        tranform : Callable[[Image], Image]
            function that takes a Image and returns a transformed Image

        Returns
        -------
        Video
            new video with tranformed frames
        """
        return Video(map(transform, self.frames), self.fixed)

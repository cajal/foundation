import numpy as np
from .logging import logger


class Video:
    def __init__(self, frames, homogenous=True):
        """
        Parameters
        ----------
        frames : Sequence[PIL.Image]
            stimulus frames
        homogenous : bool
            all frames are the same mode, height, and width
        """
        self.frames = tuple(frames)
        self.homogenous = bool(homogenous)

        if self.homogenous:
            for frame in self.frames[1:]:
                assert frame.mode == self.frames[0].mode
                assert frame.height == self.frames[0].height
                assert frame.width == self.frames[0].width

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
        if self.homogenous:
            return self.frames[0].mode
        else:
            logger.warn("Only supported for homogenous videos.")

    @property
    def height(self):
        """
        Returns
        -------
        int | None
            frame height
        """
        if self.homogenous:
            return self.frames[0].height
        else:
            logger.warn("Only supported for homogenous videos.")

    @property
    def width(self):
        """
        Returns
        -------
        int | None
            frame width
        """
        if self.homogenous:
            return self.frames[0].width
        else:
            logger.warn("Only supported for homogenous videos.")

    @property
    def channels(self):
        """
        Returns
        -------
        int | None
            frame width
        """
        if not self.homogenous:
            logger.warn("Only supported for homogenous videos.")
            return

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
        if not self.homogenous:
            logger.warn("Only supported for homogenous videos.")
            return

        if self.mode == "L":
            frames = np.stack([np.array(frame) for frame in self.frames], 0)
            return frames[:, :, :, None]
        else:
            raise NotImplementedError(f"Mode {self.mode} has not yet been implemented.")

    def apply(self, transform):
        """
        Parameters
        ----------
        tranform : Callable[[PIL.Image], PIL.Image]
            function that takes a PIL.Image and returns a transformed PIL.Image

        Returns
        -------
        Video
            new video with tranformed frames
        """
        return Video(map(transform, self.frames), self.homogenous)

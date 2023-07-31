import numpy as np
from PIL import Image as Frame
from .resample import flip_index
from .logging import tqdm


class Video:
    def __init__(self, frames, period=None, times=None):
        """
        Parameters
        ----------
        frames : Sequence[Frame]
            stimulus frames
        period : None | float
            flip period (seconds)
        times : None | 1D array
            flip times (seconds)
        """
        self.frames = tuple(frames)

        assert np.unique([f.mode for f in self.frames]).size == 1
        assert np.unique([f.height for f in self.frames]).size == 1
        assert np.unique([f.width for f in self.frames]).size == 1

        if period is None and times is None:
            self.period = None
            self.times = None

        elif period is not None and times is None:
            self.period = float(period)
            self.times = self.period * np.arange(len(self))

        elif period is None and times is not None:
            self.period = None
            self.times = np.array(times, dtype=float)
            assert len(self.times) == len(self)

        else:
            raise ValueError("Either `period` or `times` can be provided, not both.")

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

    @classmethod
    def fromarray(cls, array, mode=None, period=None, times=None):
        """
        Returns
        -------
        array: 3D array | 4D array
            [frames, height, width] | [frames, height, width, channels]
        mode : bool
            frame mode
        period : None | float
            flip period (seconds)
        times : None | 1D array
            flip times (seconds)

        Returns
        -------
        Video
            video object from the provided arrays and attributes
        """
        if array.ndim == 4:
            array = array.squeeze(3)
        elif array.ndim != 3:
            raise ValueError("Array must be either 4D or 3D")

        return cls([Frame.fromarray(frame, mode=mode) for frame in array], period=period, times=times)

    def apply(self, transform):
        """
        Parameters
        ----------
        tranform : Callable[[Frame], Frame]
            function that takes a Frame and returns a transformed Frame

        Returns
        -------
        Video
            new video with tranformed frames
        """
        if self.period is not None:
            return self.__class__(map(transform, self.frames), period=self.period)

        elif self.times is not None:
            return self.__class__(map(transform, self.frames), times=self.times)

        else:
            return self.__class__(map(transform, self.frames))

    def animate(self, fps=30, vmin=0, vmax=255, cmap="gray", width=6, dpi=None, html=True):
        """
        Parameters
        ----------
        fps : float
            animation frames per second
        cmap : str | matplotlib.colors.Colormap
            colormap -- ignored if video is RGB(A)
        width : float
            width of the video in inches
        dpi : float | None
            dots per inch
        html : bool
            return HTML for IPython display | return pyplot FuncAnimation

        Returns
        -------
            IPython.display.HTML | matplotlib.animation.FuncAnimation
                animated video, either as HTML or pyplot object
        """
        from matplotlib import pyplot as plt
        from matplotlib import animation

        if self.times is None:
            raise ValueError("Cannot animate without timing information")

        index = flip_index(self.times, 1 / fps)
        frames = self.array[index]

        fig = plt.figure(figsize=(width, width / self.width * self.height), dpi=dpi)
        im = plt.imshow(frames[0], vmin=vmin, vmax=vmax, cmap=cmap)

        plt.axis("off")
        plt.tight_layout()
        plt.close()

        def init():
            im.set_data(frames[0])

        def animate(i):
            im.set_data(frames[i])
            return im

        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames), interval=1000 / fps)

        if html:
            from IPython.display import HTML

            return HTML(ani.to_jshtml())

        else:
            return ani

    def generate(self, period, array=True, display_progress=True):
        """
        Parameters
        ----------
        period : float
            sampling period (seconds)
        array : bool
            numpy array (True) | PIL Image (False)
        display_progress : bool
            display generation progress

        Yields
        -------
            np.array | PIL.Image
                video frame, either as numpy array or PIL Image
        """
        index = flip_index(self.times, period)

        if display_progress:
            index = tqdm(index, desc="Video Frames")

        for i in index:
            if array:
                yield np.array(self.frames[i])
            else:
                yield self.frames[i]

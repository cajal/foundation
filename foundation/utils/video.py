import numpy as np
from PIL import Image as Frame


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

    def animate(self, start=None, end=None, fps=30, vmin=0, vmax=255, cmap="gray", width=6, dpi=None, html=True):
        """
        Parameters
        ----------
        start : int | None
            start frame
        end : int | None
            end frame
        fps : float
            frames per second
        vmin : float
            data range minimum
        vmax : float
            data range maximum
        cmap : str | matplotlib.colors.Colormap
            colormap -- ignored if video is RGB(A)
        width : float
            width of the video in inches
        dpi : float | None
            dots per inch
        html : bool
            return HTML for IPython display | pyplot FuncAnimation
        """
        from matplotlib import pyplot as plt
        from matplotlib import animation

        frames = self.array[slice(start, end)]

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

        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self), interval=1000 / fps)

        if html:
            from IPython.display import HTML

            return HTML(ani.to_jshtml())

        else:
            return ani

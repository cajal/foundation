from PIL import Image


# ------- Resize Interface -------


class Resize:
    """Resize Base"""

    def __call__(self, video, height, width):
        """
        Parameters
        ----------
        video : foundation.utils.video.Video
            video to be resized
        height : int
            target height
        width : width
            target widht

        Returns
        -------
        foundation.utils.video.Video
            resized video
        """
        raise NotImplementedError()


# ------- Resize Types -------


class PilResize(Resize):
    """Resizes video via PIL"""

    def __init__(self, resample):
        """
        Parameters
        ----------
        resample : PIL.Image.Resampling.*
            PIL resampling method
        """
        self.resample = resample
        assert resample in Image.Resampling

    def __call__(self, video, height, width):

        if height == video.height and width == video.width:
            return video

        else:
            f = lambda img: img.resize(size=(width, height), resample=self.resample)
            return video.apply(f)

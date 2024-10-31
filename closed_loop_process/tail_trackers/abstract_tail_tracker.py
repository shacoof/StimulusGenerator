from abc import ABC, abstractmethod


class AbstractTailTracker(ABC):
    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.tail_angles = None
        self.tail_points = None
        self.current_frame = None

    @abstractmethod
    def tail_tracking(self, raw_frame):
        """
        Args:
            raw_frame: np array of the raw frame received by camera
        Returns:
        tail_points -  np array of dims m x 2 of 2d points in pixels where m is the number of points
        representing the tail
        tail_angles - np array of dims 98 x 1 of angles in radians representing the tail shape
        """
        self.current_frame = raw_frame


    def plot_tail(self):
        """plots the tail detection on the image"""
        if self.tail_points is None or self.current_frame is None:
            raise RuntimeError("need to run tail detect first")



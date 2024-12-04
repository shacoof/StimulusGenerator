from abc import ABC, abstractmethod

class AbstractTailTracker(ABC):
    def __init__(self, image_processor):
        '''
        Base class for tail tracking, all child classes must return the tail detection np.array of the tail point in
        pixels and the tail angles when implementing the abstract tail_tracking method
        Args:
            image_processor: image processor that preforms the preprocessing of the current raw frame to create a
            manipulated frame to do the tail fracking on
        '''
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




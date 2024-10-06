from collections import deque
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Suppress matplotlib font manager debug logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class RecognizeBout:
    def __init__(self, first_frame_from_calibration, bout_threshold_pixel,
                 sum_moving_threshold_percentage, frame_memory, to_plot, tail_tip, x_dist_from_tail = 13,
                 y_dist_from_tail = 30):
        """
        Recognizes bouts and ibis each frame
        :param first_frame_from_calibration: for the first frame compare to the mean frame from calibration
        :param threshold: threshold of diff between frame to decide if this is the start of a bout
        """
        first_frame_from_calibration = first_frame_from_calibration.astype('uint8')
        self.bout_threshold = bout_threshold_pixel
        self.sum_moving_threshold_percentage = sum_moving_threshold_percentage
        self.recent_frames_percentage = deque(maxlen=frame_memory)
        self.frame_memory = frame_memory
        self.previous_frame = None
        self.tail_tip = tail_tip
        self.x_dist_from_tail = x_dist_from_tail
        self.y_dist_from_tail = y_dist_from_tail
        self.current_frame = first_frame_from_calibration[self.tail_tip[1] - self.y_dist_from_tail: self.tail_tip[1] + self.y_dist_from_tail,
                             self.tail_tip[0] - self.x_dist_from_tail: self.tail_tip[0] + self.x_dist_from_tail]
        self.num_pixels = self.current_frame.shape[0]*self.current_frame.shape[1]
        self.to_plot = to_plot
        self.percentage = 0
        self.movement_mask = None



    def update(self, frame):
        cropped_frame = frame[self.tail_tip[1] - self.y_dist_from_tail: self.tail_tip[1] + self.y_dist_from_tail,
                             self.tail_tip[0] - self.x_dist_from_tail: self.tail_tip[0] + self.x_dist_from_tail]
        self.previous_frame = self.current_frame
        self.current_frame = cropped_frame
        # Compute the absolute difference between the frames
        diff = cv2.absdiff(self.previous_frame, self.current_frame)

        # Apply a threshold to get a binary image showing areas of movement
        _, movement_mask = cv2.threshold(diff, self.bout_threshold, 255, cv2.THRESH_BINARY)

        # Compute the percentage of movement
        self.movement_mask = movement_mask
        self.percentage = (np.sum(movement_mask == 255) / self.num_pixels) * 100
        self.recent_frames_percentage.append(self.percentage)

    def is_start_of_bout(self, frame_number):
        """
        Determines if the current frame signifies the start of a bout based on movement detection.

        :param frame_number: The current frame number (for context, included in the plot title).
        :return: Tuple (bool indicating if it's the start of a bout, percentage of movement)
        """

        # Check if the percentage of movement exceeds the threshold
        is_bout_start = self.percentage > self.sum_moving_threshold_percentage
        ibi = self._is_ibi()
        is_start_of_bout = is_bout_start & ibi

        if self.to_plot:
            # Clear the current axes
            plt.clf()

            # Display the current frame
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(self.current_frame, cmap='gray')
            ax1.set_title(f'Current Frame: {frame_number}')

            # Display the movement mask
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(self.movement_mask, cmap='gray')
            title_color = 'red' if is_start_of_bout else 'black'
            ax2.set_title(f'Movement Mask - Percentage: {self.percentage:.2f}%', color=title_color)

            # Update the title indicating bout start
            plt.suptitle("Bout Start Detected!" if is_start_of_bout else "", color='red', fontsize=16)

            plt.draw()  # Update the figure
            plt.pause(0.2)  # Pause to allow the plot to update

        return is_start_of_bout, self.percentage

    def _is_ibi(self):
        """
        See if the in the past frame_memory all diffs are less than still_threshold
        :return:
        """
        for i in range(len(self.recent_frames_percentage)-1):
            if self.recent_frames_percentage[i] > self.sum_moving_threshold_percentage:
                return False
        return True






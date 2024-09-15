from recognize_bout_start.RecognizeBout import RecognizeBout
from calibration.calibrate import Calibrator
from closed_loop_config import *
from renderer.Renderer import Renderer
import numpy as np
import os
import time

class ClosedLoop:
    def __init__(self,pca_and_predict, image_processor, tail_tracker, bout_recognizer, multiprocess_prediction_queue):
        """
        Preforms closed loop
        """
        self.pca_and_predict = pca_and_predict
        self.image_processor = image_processor
        self.tail_tracker = tail_tracker
        self.bout_recognizer = bout_recognizer
        self.is_bout = False
        self.bout_index = 0
        self.bout_frames = np.zeros((frames_from_bout, 105, 2))
        self.current_frame = 0
        self.renderer = Renderer()
        self.multiprocess_prediction_queue = multiprocess_prediction_queue
        self.time = 0

    def process_frame(self, frame):
        time_now = time.time()
        print(f"time {time_now - self.time}")
        self.time = time_now

        # time.sleep(1)
        # if self.current_frame % 2 ==0:
        #     new_angle, new_distance = self.renderer.calc_new_angle_and_size(40, 1)
        # else:
        #     new_angle, new_distance = self.renderer.calc_new_angle_and_size(-40, 1)
        # print(f"angle {new_angle} distance {new_distance}")
        # self.multiprocess_prediction_queue.put((new_angle, new_distance))
        # self.current_frame += 1


        # self.current_frame += 1
        # self.image_processor.load_mat(frame)
        # self.bout_recognizer.update(self.image_processor.get_image_matrix())
        # # if this is a bout frame
        # if self.is_bout:
        #     self.bout_index += 1
        #     binary_image = self.image_processor.preprocess_binary()
        #     self.tail_tracker.load_binary_image(binary_image)
        #     tail_points = self.tail_tracker.get_tail_points(self.current_frame)
        #     self.bout_frames[self.bout_index, :, :] = tail_points
        #     #last bout frame
        #     if self.bout_index == frames_from_bout:
        #         self.is_bout = False
        #         self.bout_index = 0
        #         angle, distance = self.pca_and_predict.reduce_dimensionality_and_predict(bout_frames, to_plot=debug_PCA)
        #         self.bout_frames = np.zeros((frames_from_bout, 105, 2))
        #         new_angle, new_distance = self.renderer.calc_new_angle_and_size(angle, distance)
        #         self.multiprocess_prediction_queue.put((angle, distance))
        #         print(
        #             f"frame {self.current_frame} predicted angle {angle}, predicted distance {distance}")
        # else:
        #     verdict, diff = self.bout_recognizer.is_start_of_bout(self.current_frame)
        #     if verdict:
        #         self.bout_index += 1
        #         binary_image = self.image_processor.preprocess_binary()
        #         self.tail_tracker.load_binary_image(binary_image)
        #         tail_points = self.tail_tracker.get_tail_points(self.current_frame)
        #         self.bout_frames[self.bout_index, :, :] = tail_points



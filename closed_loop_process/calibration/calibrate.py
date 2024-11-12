import time

from closed_loop_process.PCA_and_predict.PCA_166_Hz import PCA166Hz
from closed_loop_process.PCA_and_predict.PCA_500_Hz import PCA500Hz
from closed_loop_process.tail_trackers.tail_tracker_lab import LabTailTracker
from closed_loop_process.tail_trackers.tail_tracker_stytra import StytraTailTracker

from config_files.preprocess_config import *
from closed_loop_process.calibration.point_selector import PointSelector
from closed_loop_process.image_processor.ImageProcessor import ImageProcessor
from closed_loop_process.recognize_bout_start.RecognizeBout import RecognizeBout
from camera.Flir_camera import SpinnakerCamera
from config_files.closed_loop_config import *

import numpy as np
import os
from tqdm import tqdm



class Calibrator:
    def __init__(self, live_camera = True, images_path = None,
                start_frame = 0, end_frame = number_of_frames_calibration, calib_frame_ranges = None):
        """
        Args:
            live_camera: use live camera - in this case the FLIR camera needs to be connected
            images_path: for running on recorded data (this is the raw data path)
            plot_bout_detector: for debugging the bout detector
            start_frame: for running on recorded data start frame used in calib - either provide start and end frame or
             provide calib_frame_ranges
            end_frame: for running on recorded data start frame used in calib
            calib_frame_ranges: example [[30,80],[122, 300], [507, 600]] ranges of frames to use for calibration
            use_stytra_tracking: tail detection techniques
            fr_500: if true then the frame rate is 500 or else it is 166
        """
        self.live_camera = live_camera
        self.mean_frame = None
        self.min_frame = None
        self.head_origin = None
        self.head_dest = None
        self.tail_tip = None
        self.tail_mid = None
        self.focal_lim_x = None
        self.focal_lim_y = None
        self.pca_and_predict = None
        self.bout_recognizer = None
        self.images_paths = []
        self.current_frame = 0
        self.camera = None
        if live_camera:
            self.camera = SpinnakerCamera()
            self.camera.set_image_dimensions(width=camera_frame_width_in_pixels,height=camera_frame_height_in_pixels,
                                             offsetX=camera_frame_offsetX_in_pixels,offsetY=camera_frame_offsetY_in_pixels)
            self.camera.set_camera_settings(frame_rate=camera_frame_rate)
            self.num_frames = end_frame - start_frame
        self.image_processor = ImageProcessor(live_camera, self.camera)
        self.image_processor.image_processor_start_camera()
        if live_camera == False and images_path is None:
            raise RuntimeError("enter images directory")
        if not live_camera:
            # Load only the images within the specified range
            if calib_frame_ranges:
                for i in range(len(calib_frame_ranges)):
                    for j in range(calib_frame_ranges[i][0],calib_frame_ranges[i][1]):
                        # Format the image filename based on the numbering pattern
                        img_filename = f"img{str(j).zfill(12)}.jpg"
                        img_path = os.path.join(images_path, img_filename)
                        self.images_paths.append(img_path)
            else:
                for i in range(start_frame, end_frame):
                    # Format the image filename based on the numbering pattern
                    img_filename = f"img{str(i).zfill(12)}.jpg"
                    img_path = os.path.join(images_path, img_filename)
                    self.images_paths.append(img_path)
            self.num_frames = len(self.images_paths)
        self.first_image = self.load_image()
        self.current_frame = 0
        self.get_area_of_interest()
        self.start_frame = start_frame
        self.end_frame = end_frame

    def get_area_of_interest(self):
        first_img_arr = self.first_image
        selector = PointSelector(first_img_arr)
        points = selector.select_points()
        self.head_origin = [round(value) for value in list(points[0])]
        self.head_dest =  [round(value) for value in list(points[1])]
        self.tail_tip = [round(value) for value in list(points[2])]
        self.tail_mid = [round((self.head_origin[0] + self.tail_tip[0])/2),round((self.head_origin[1] + self.tail_tip[1])/2)]
        self.focal_lim_x = [self.head_origin[0] - FOCAL_LIM_X_MINUS, self.head_origin[0] + FOCAL_LIM_X_PLUS]
        self.focal_lim_y = [self.head_origin[1] - FOCAL_LIM_Y_MINUS, self.head_origin[1] + FOCAL_LIM_Y_PLUS]


    def load_image(self):
        """
        Loads the image from the given path and stores it as a NumPy array when not in camera mode, otherwise reads
        frame from the camera
        """
        if self.live_camera:
            self.image_processor.load_image()
        else:
            self.image_processor.load_image(self.images_paths[self.current_frame])
        self.current_frame += 1
        return self.image_processor.get_image_matrix()


    def start_calibrating(self, stimuli_queue = None):
        self.bout_recognizer = self._init_bout_recognizer()
        self._calc_mean_min_frame(stimuli_queue)
        if fr_500:
            self.pca_and_predict = PCA500Hz()
        else:
            self.pca_and_predict = PCA166Hz()

        if use_stytra_tracking:
            self.tail_tracker = StytraTailTracker(image_processor=self.image_processor,
                                                  head_origin=self.head_origin, tail_tip=self.tail_tip)
        else:
            self.tail_tracker = LabTailTracker(image_processor=self.image_processor,
                                               head_origin=self.head_origin)
        self.image_processor.camera = None
        return self.pca_and_predict, self.image_processor, self.bout_recognizer, self.tail_tracker

    def _calc_mean_min_frame(self,stimuli_queue):
        first_img_arr = self.first_image
        current_min = first_img_arr[self.focal_lim_y[0]:self.focal_lim_y[1], self.focal_lim_x[0]:self.focal_lim_x[1]]
        current_sum = first_img_arr
        start_angle_of_stimuli = start_angle
        end_angle_of_stimuli = end_angle
        is_start = True
        for i in tqdm(range(self.num_frames)):
            if i % 830 == 0:
                if stimuli_queue:
                    angle = start_angle_of_stimuli if is_start else end_angle_of_stimuli
                    stimuli_queue.put((angle, 0))
                    is_start = not is_start
            img_arr = self.load_image()
            self.bout_recognizer.update(img_arr)
            verdict, diff = self.bout_recognizer.is_start_of_bout(i)
            if verdict:
                print("bout detected!!!!")
            current_min = np.minimum(current_min, img_arr[self.focal_lim_y[0]:self.focal_lim_y[1],
                                                  self.focal_lim_x[0]:self.focal_lim_x[1]])
            current_sum = np.add(current_sum, img_arr, dtype=np.uint32)

        self.min_frame = current_min
        self.mean_frame = current_sum / self.num_frames
        self.image_processor.calc_masks(self.min_frame,self.mean_frame,self.head_origin,
                                        number_of_frames_used_in_calib=self.num_frames)
        self.image_processor.image_processor_stop_camera()


    def _init_bout_recognizer(self):
        first_img_arr = self.first_image
        if fr_500:
            bout_recognizer = RecognizeBout(first_img_arr, 10, 3,
                                            7, plot_bout_detector, self.tail_mid)
        else: #166 Hz
            bout_recognizer = RecognizeBout(first_img_arr, 10, 4,
                                        5, plot_bout_detector, self.tail_mid)
        return bout_recognizer













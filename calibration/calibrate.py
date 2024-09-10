from utils_closed_loop.machine_vision import load_image
from preprocess_config import *
from PCA_and_predict.PCA import PCA
from calibration.point_selector import PointSelector
from image_processor.ImageProcessor import ImageProcessor
from image_processor.tail_tracker import TailTracker
from recognize_bout_start.RecognizeBout import RecognizeBout
from camera.Flir_camera import SpinnakerCamera

import numpy as np
import os
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt



class Calibrator:
    def __init__(self, calculate_PCA = False, live_camera = True, images_path = None, num_frames = 1000,
                 plot_bout_detector = False):
        """
        Calcultes returns calibrated image processor, PCA predictor, tail tracker
        :param calculate_PCA: boolean value, calculate PCs on fixed fish or use previously calculated projection matrix
        :param live_camera: boolean value, input data for calibration is from live camera or camera frames
        :param num_frames: number of frames to use for calibration
        """
        self.calculate_PCA = calculate_PCA
        self.num_frames = num_frames
        self.live_camera = live_camera
        self.mean_frame = None
        self.min_frame = None
        self.head_origin = None
        self.head_dest = None
        self.focal_lim_x = None
        self.focal_lim_y = None
        self.pca_and_predict = None
        self.bout_recognizer = None
        self.images_paths = []
        self.current_frame = 0
        self.camera = None

        if live_camera:
            self.camera = SpinnakerCamera()
        self.image_processor = ImageProcessor(live_camera, self.camera)
        self.tail_tracker = None
        self.plot_bout_detector = plot_bout_detector

        if live_camera == False and images_path is None:
            raise RuntimeError("enter images directory")
        if not live_camera:
            sorted_filenames = sorted(
                [filename for filename in os.listdir(images_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
            )
            # Append full paths to self.images_paths
            for filename in sorted_filenames:
                self.images_paths.append(os.path.join(images_path, filename))
            if num_frames > len(self.images_paths):
                raise RuntimeError(f"Not enough frames in directory to calibrate with {num_frames} frames")


        self.first_image = self.load_image()
        self.current_frame = 0
        self.get_area_of_interest()
        self.tail_tracker = TailTracker(self.head_origin, self.head_dest)



    def get_area_of_interest(self):
        first_img_arr = self.first_image
        selector = PointSelector(first_img_arr)
        points = selector.select_points()
        #points = [(1,2),(93,4)]
        self.head_origin = [round(value) for value in list(points[0])]
        self.head_dest =  [round(value) for value in list(points[1])]
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


    def start_calibrating(self):
        self._calc_mean_min_frame()
        # Shai
        B_angle = scipy.io.loadmat("Z:\Lab-Shared\Data\ClosedLoop\movement_train_intermediate.mat")['angle_solution']
        B_distance = scipy.io.loadmat("Z:\Lab-Shared\Data\ClosedLoop\movement_train_intermediate.mat")['distance_solution']
        # Imri 500 Hz
        # B_angle = scipy.io.loadmat('Z:\Lab-Shared\Data\ClosedLoop\B_angle.mat')['angle_solution']
        # B_distance = scipy.io.loadmat('Z:\Lab-Shared\Data\ClosedLoop\B_distance.mat')['distance_solution']
        # Imri 500/3 Hz
        # B_angle = scipy.io.loadmat("Z:\Lab-Shared\Data\ClosedLoop\B_matrices_slow_imri.mat")['angle_solution']
        # B_distance = scipy.io.loadmat("Z:\Lab-Shared\Data\ClosedLoop\B_matrices_slow_imri.mat")['distance_solution']


        pca_and_predict = None
        if self.calculate_PCA:
            tail_data = self._get_tail_points_for_PCA()
            #tail_data = np.load('all_tail_data2.npy')
            pca_and_predict = PCA(prediction_matrix_angle=B_angle, prediction_matrix_distance=B_distance)
            pca_and_predict.calc_3_PCA(tail_data)
        else:
            V = scipy.io.loadmat('Z:\Lab-Shared\Data\ClosedLoop\V.mat')['V']
            S = scipy.io.loadmat('Z:\Lab-Shared\Data\ClosedLoop\S.mat')['S']
            pca_and_predict = PCA(prediction_matrix_angle=B_angle, prediction_matrix_distance=B_distance, V=V, S=S)
        self.pca_and_predict = pca_and_predict
        self.bout_recognizer = self._init_bout_recognizer()
        return self.pca_and_predict, self.image_processor, self.tail_tracker, self.bout_recognizer


    def _calc_mean_min_frame(self):
        first_img_arr = self.first_image
        current_min = first_img_arr[self.focal_lim_y[0]:self.focal_lim_y[1], self.focal_lim_x[0]:self.focal_lim_x[1]]
        current_sum = first_img_arr
        for _ in tqdm(range(self.num_frames)):
            img_arr = self.load_image()
            current_min = np.minimum(current_min, img_arr[self.focal_lim_y[0]:self.focal_lim_y[1],
                                                  self.focal_lim_x[0]:self.focal_lim_x[1]])
            current_sum = np.add(current_sum, img_arr, dtype=np.uint32)
        self.min_frame = current_min
        self.mean_frame = current_sum / self.num_frames
        # plt.imshow(current_min, cmap='gray')
        # plt.show()
        self.image_processor.calc_masks(self.min_frame,self.mean_frame,self.head_origin)


    def _get_tail_points_for_PCA(self):
        tail_data = np.zeros((self.num_frames, 105, 2))
        self.current_frame = 0
        for i in tqdm(range(self.num_frames)):
            img_arr = self.load_image()
            if self.calculate_PCA:
                binary_image = self.image_processor.preprocess_binary()
                self.tail_tracker.load_binary_image(binary_image)
                tail_points = self.tail_tracker.get_tail_points(i)
                tail_data[i, :, :] = tail_points
        #np.save('all_tail_data2.npy', tail_data)
        return tail_data

    def _init_bout_recognizer(self):
        first_img_arr = self.first_image
        bout_recognizer = RecognizeBout(first_img_arr, 10, 0.9,
                                        7, self.image_processor.get_tail_mask_x(),
                                        self.image_processor.get_tail_mask_y(), 30, self.plot_bout_detector)
        return bout_recognizer













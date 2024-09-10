from PIL import Image
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from camera.Flir_camera import SpinnakerCamera

from preprocess_config import *


class ImageProcessor:
    def __init__(self, is_live_camera=False, camera:SpinnakerCamera = None):
        """
        Initialize the ImageProcessor with the path to an image file.
        :param image_path: str, path to the image file
        """
        self.image_path = None
        self.image_matrix = None
        self.min_frame = None
        self.mean_frame = None
        self.focal_lim_x = None
        self.focal_lim_y = None
        self.tail_mask_x = None
        self.tail_mask_y = None
        self.kernel = None
        self.is_live_camera = is_live_camera
        if is_live_camera and camera is None:
            raise RuntimeError("need to supply camera")
        self.camera = camera


    def load_image(self, image_path=""):
        """
        Loads the image from the given path and stores it as a NumPy array.
        """
        if self.is_live_camera:
            self.image_matrix = self.camera.get_frame()
            return self.image_matrix
        else:
            try:
                with Image.open(image_path) as img:
                    self.image_matrix = np.array(img)
                    return self.image_matrix
            except Exception as e:
                print(f"Error loading image: {e}")

    def load_mat(self, mat):
        self.image_matrix = mat

    def get_image_matrix(self):
        """
        Returns the image matrix.
        :return: np.ndarray, the image matrix
        """
        if self.image_matrix is not None:
            return self.image_matrix
        else:
            print("Image not loaded yet. Please load an image first.")
            return None

    def display_image(self, text=''):
        """
        Displays the image using PIL's show method.
        """
        if self.image_matrix is not None:
            # Convert NumPy array back to PIL Image
            img = Image.fromarray(self.image_matrix)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            # Position for the text (bottom-right corner)
            text_position = (img.width - 100, img.height - 30)  # Adjust position as needed

            # Add text to the image
            draw.text(text_position, text, font=font, fill="white")
            img.show()
        else:
            print("Image not loaded yet. Please load an image first.")

    def calc_masks(self, min_frame, mean_frame, head_origin):
        # calc in advance - min_frame, focal_lim_x, focal_lim_y, tail_mask_y, tail_mask_x, kernel
        focal_lim_x = [head_origin[0] - FOCAL_LIM_X_MINUS, head_origin[0] + FOCAL_LIM_X_PLUS]
        focal_lim_y = [head_origin[1] - FOCAL_LIM_Y_MINUS, head_origin[1] + FOCAL_LIM_Y_PLUS]
        head_mask_x = [head_origin[0] - focal_lim_x[0] - HEAD_MASK_X_MINUS,
                       head_origin[0] - focal_lim_x[0] + HEAD_MASK_X_PLUS]
        head_mask_y = [head_origin[1] - focal_lim_y[0] - HEAD_MASK_Y_MINUS,
                       head_origin[1] - focal_lim_y[0] + HEAD_MASK_Y_PLUS]
        tail_mask_x = [head_origin[0] - FOCAL_LIM_X_MINUS, head_origin[0] + FOCAL_LIM_X_PLUS]
        tail_mask_y = [head_origin[1] + TAIL_MASK_Y_LOWER, head_origin[1] + TAIL_MASK_Y_UPPER]
        kernel = np.ones((5, 5), np.uint8)
        head_mask = min_frame[head_mask_y[0]:head_mask_y[1], head_mask_x[0]:head_mask_x[1]]
        head_mask[head_mask > HEAD_MASK_THRESHOLD] = 0
        min_frame[head_mask_y[0]:head_mask_y[1], head_mask_x[0]:head_mask_x[1]] = head_mask
        self.min_frame = min_frame
        self.focal_lim_x = focal_lim_x
        self.focal_lim_y = focal_lim_y
        self.tail_mask_x = tail_mask_x
        self.tail_mask_y = tail_mask_y
        self.kernel = kernel
        self.mean_frame = mean_frame

    def preprocess_binary(self):
        if self.image_matrix is None or self.min_frame is None:
            print("Please load an image first and run calc masks")
            return None
        img_arr = self.image_matrix
        tail_mask_x = self.tail_mask_x
        tail_mask_y = self.tail_mask_y
        focal_lim_x = self.focal_lim_x
        focal_lim_y = self.focal_lim_y
        subtracted_img = np.subtract(img_arr, self.mean_frame)
        subtracted_img[focal_lim_y[0]:focal_lim_y[1], focal_lim_x[0]:focal_lim_x[1]] = np.subtract(
            img_arr[focal_lim_y[0]:focal_lim_y[1], focal_lim_x[0]:focal_lim_x[1]], self.min_frame)
        binary_img = cv2.threshold(subtracted_img, IMG_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        binary_img = binary_img.astype('uint8')
        # tail dilation twice
        img_dilation = cv2.dilate(binary_img[tail_mask_y[0]:tail_mask_y[1], tail_mask_x[0]:tail_mask_x[1]], self.kernel,
                                  iterations=2)
        binary_img[tail_mask_y[0]:tail_mask_y[1], tail_mask_x[0]:tail_mask_x[1]] = img_dilation
        # tail erosion once
        img_erosion = cv2.erode(binary_img[tail_mask_y[0]:tail_mask_y[1], tail_mask_x[0]:tail_mask_x[1]], self.kernel,
                                iterations=1)
        binary_img[tail_mask_y[0]:tail_mask_y[1], tail_mask_x[0]:tail_mask_x[1]] = img_erosion
        self.image_matrix = binary_img
        return binary_img

    def get_tail_mask_x(self):
        if self.tail_mask_x is None:
            raise RuntimeError(f"need run run calc masks first")
        return self.tail_mask_x

    def get_tail_mask_y(self):
        if self.tail_mask_y is None:
            raise RuntimeError(f"need run run calc masks first")
        return self.tail_mask_y





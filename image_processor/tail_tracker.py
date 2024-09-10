from image_processor.RealTimeTailTracking import RealTimeTailTracking
import numpy as np
import matplotlib.pyplot as plt



class TailTracker:
    def __init__(self, head_origin, head_dest):
        """
        The class gets points from tail
        """
        self.binary_matrix = None
        self.tail_points = None
        self.head_origin = head_origin
        self.head_dest = head_dest
        self.real_time_tail_tracker = RealTimeTailTracking(True, False, False,
                                                      head_origin=self.head_origin, head_destination=self.head_dest)

    def load_binary_image(self, binary_image: np.array):
        self.binary_matrix = binary_image

    def get_tail_points(self, frame_num):
        if self.binary_matrix is None:
            raise RuntimeError("need to run load_binary_image first")
        binary_matrix = self.binary_matrix
        height, width = binary_matrix.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[binary_matrix == 0] = [0, 0, 0]  # For black
        rgb_image[binary_matrix == 255] = [255, 255, 255]  # For white

        annotated_frame, fish_analysis_output = self.real_time_tail_tracker._analyse(rgb_image, None, None, frame_num)
        if fish_analysis_output.tail_data is None:
            return np.full((105, 2), None, dtype=object)
        tail_points = fish_analysis_output.tail_data.tail_path
        interpolated_tail = TailTracker.get_smooth_tail(tail_points, self.head_origin)
        self.tail_points = interpolated_tail[1]
        return self.tail_points



    def plot_points(self, frame_num):
        """
        Plots a 3D RGB image with specified tail points and displays the frame number.
        :param frame_num: The frame number to display in the plot title.
        """
        if self.tail_points is None:
            raise RuntimeError("Need to run get_tail_points first")

        image = self.binary_matrix
        points = self.tail_points

        # Clear the current axes
        plt.clf()

        # Create subplot (or get existing one if already created)
        ax = plt.gca()

        # Plot the image
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)

        # Separate x and y coordinates
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # Overlay points on the image
        ax.scatter(x_coords, y_coords, c='red', marker='o', s=1)

        # Add the frame number as the title
        ax.set_title(f"Frame Number: {frame_num}")
        ax.axis('off')

        # Update the plot
        plt.draw()
        plt.pause(0.1)  # Pause briefly to ensure the plot window updates

    @staticmethod
    def get_smooth_tail(tail_path, swimbladder_point):
        number_of_tail_segments = 105
        extra_size = 6
        polyfit_power = 7
        interpolation_points = 10000

        if np.isnan(swimbladder_point[0]) or swimbladder_point[0] == 0:
            nan_array = np.empty((number_of_tail_segments, 2))
            nan_array[:] = np.NaN
            return np.nan, nan_array

        tail_dx_dy = tail_path[:-1, :] - tail_path[1:, :]
        tail_size = np.sqrt(np.sum(tail_dx_dy ** 2, axis=1))
        size_until_point = np.append([0], np.cumsum(tail_size))

        # swimbladder_idx = np.where(np.all(tail_path == swimbladder_point,axis=1))[0][0] # IL: replaced with next line for cases in which sb is not on tail path
        swimbladder_idx = np.argmin(np.linalg.norm(tail_path - swimbladder_point, axis=1))

        tail_xy = tail_path[:swimbladder_idx + 1, :].copy()
        size_until_point = size_until_point[:swimbladder_idx + 1]

        if len(tail_xy) < 10:
            nan_array = np.empty((number_of_tail_segments, 2))
            nan_array[:] = np.NaN
            return np.nan, nan_array

        xfit = np.polyfit(size_until_point, tail_xy[:, 0], polyfit_power)
        yfit = np.polyfit(size_until_point, tail_xy[:, 1], polyfit_power)

        tail_size_interp = np.linspace(0, max(size_until_point), num=interpolation_points)
        full_x_tail, full_y_tail = np.polyval(xfit, tail_size_interp), np.polyval(yfit, tail_size_interp)
        full_xy_tail = np.append(full_x_tail[:, np.newaxis], full_y_tail[:, np.newaxis], axis=1)
        full_tail_dxdy = full_xy_tail[:-1, :] - full_xy_tail[1:, :]
        tail_size = np.sqrt(np.sum(full_tail_dxdy ** 2, axis=1))
        size_until_point = np.cumsum(np.append([0], tail_size))

        xy_final = np.zeros((number_of_tail_segments, 2))
        segment_size = size_until_point[-1] / number_of_tail_segments

        for poly_idx in range(number_of_tail_segments):
            cur_idx = 0 if poly_idx == 0 else np.where(size_until_point < poly_idx * segment_size)[0].max()
            xy_final[poly_idx, :] = full_xy_tail[cur_idx, :]

        # notice that size_until_point[-1] is the size of tip of tail to the swimbladder, but the interpolated tail is 1 segment less than this size.
        return size_until_point[-1], xy_final


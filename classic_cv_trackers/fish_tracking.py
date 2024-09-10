import time

import networkx
import logging
from scipy.interpolate import interp1d

import cv2
from networkx.algorithms.components.connected import connected_components
import numpy as np
import math
from scipy.spatial import distance
from scipy.signal import find_peaks
from tqdm import tqdm

from classic_cv_trackers.abstract_and_common_trackers import ClassicCvAbstractTrackingAPI
from classic_cv_trackers import Colors
from classic_cv_trackers.common_tracking_features import EllipseData, absolute_angles_and_differences_in_deg
from utils_closed_loop.utils import get_angle_to_horizontal
from utils_closed_loop.noise_cleaning import clean_plate
from utils_closed_loop import machine_vision, numerical_analysis, svd_analyzer


class FrameAnalysisData:  # todo make common fish output? after tail merge
    """
    Hold tracking output. todo - simplify output to list? (remove inner classes?)
    """

    class TailData:
        tail_tip_point = None
        swimbladder_point = None

        def __init__(self, fish_tail_tip_point, tail_path, swimbladder_point):
            self.tail_tip_point = fish_tail_tip_point
            self.tail_path = tail_path
            self.swimbladder_point = swimbladder_point

    class EyesData:
        ellipses = None
        abs_angle_deg = None  # relative to X axis, 0-360 deg
        diff_from_fish_direction_deg = None
        contour_areas = None

        def __init__(self, eyes_dict, p_from, p_to):
            self.ellipses = [EllipseData(eye['center'], eye['major'], eye['minor'], eye['angle'])
                             for eye in eyes_dict]

            head_direction_angle = get_angle_to_horizontal(p_from, p_to)  # 0-360 relative to y axis
            eye_angles = [ContourBasedTracking.fix_ellipse_angle(eye['angle'], head_direction_angle)
                          for eye in eyes_dict]
            self.abs_angle_deg, self.diff_from_fish_direction_deg = \
                absolute_angles_and_differences_in_deg(eye_angles, head_direction_angle)
            self.contour_areas = [eye['area'] for eye in eyes_dict]

    is_ok = False
    is_prediction = False
    fish_contour = None
    fish_segment = None  # full fish pixels (for masking) - todo use contour instead (should change usage as well)
    eyes_contour = None
    eyes_data = None
    fish_head_origin_point = (np.nan, np.nan)
    fish_head_destination_point = (np.nan, np.nan)
    tail_data = None


class ContourBasedTracking(ClassicCvAbstractTrackingAPI):
    """ Use contours with surrounding ellipse around the fish for head direction and eyes position.
    This tracker functionality can be used by other trackers, to get additional metadata s.a. eyes and fish contours

    Usage:
    tracker = ContourBasedTracking()
    annotated_frame, output = tracker.analyse(frame, noise_frame, fps)  # lists are np.array of nX2 size for n points
    if output.is_ok: # false when can't find fish
        output.fish_contour will contain the fish external contour only
        output.fish_segment will contain the fish coordinates, in cv2 coordinate system (as returned from findNonZero)

    """
    minimum_midline_contour_area = 2
    small_fish_max_distance_from_origin = 50  # this value also allowing the bottom part of the swimbladder to be identify. used to be 50 in Yogev's version.
    small_fish_min_distance_from_origin = 20
    small_fish_size = 160
    clean_bnw_for_tail_threshold = 55

    def __init__(self, visualize_movie=False, input_video_has_plate=True, is_fast_run=False,
                 logger_filename='', is_debug=True, scale_area=1, is_twin_view=False, save_n_back=6,
                 max_invalid_n_back=2, is_predicting_missing_eyes=True):
        ClassicCvAbstractTrackingAPI.__init__(self, visualize_movie=visualize_movie)
        if logger_filename != '':
            if is_debug:
                logging.basicConfig(filename=logger_filename, level=logging.DEBUG)
            else:
                logging.basicConfig(filename=logger_filename, level=logging.INFO)
        elif is_debug:
            logging.basicConfig(level=logging.DEBUG)

        self.name = "fish_contour"
        self.should_remove_plate = input_video_has_plate
        self.is_twin_view = is_twin_view
        self.is_fast_run = is_fast_run
        self.is_predicting_missing_eyes = is_predicting_missing_eyes
        self.save_n_back = save_n_back
        self.min_valid_n_back = min(save_n_back, abs(self.save_n_back - max_invalid_n_back))

        # thresholds - todo parameters?
        self.hunt_threshold = 25
        self.tail_point_deviation_threshold = 24  # Determined using a histogram from 35000 data points
        self.head_origin_deviation_threshold = 30
        self.bnw_threshold = 10
        self.bout_movement_threshold = 3.1
        self.min_bout_frame_length = 10

        if scale_area is not None and isinstance(scale_area, (float, int)):
            self.scale = scale_area  # scale parameters relating to sizes
            logging.debug("Fish tracker initiated with scale=" + str(self.scale))

    def __reset_history(self):
        if self.save_n_back is not None:
            self.count_n_back = 0
            self.history_analysis_data_outputs.reset(frame_start=1, n_frames=self.save_n_back + 1)
            self.prev_head_tail_distance = []

    def _pre_process(self, dir_path, fish_name, event_id, noise_frame) -> None:
        pass  # nothing for now

    def detect_flipped_tail(self, analysis_data_outputs = None):
        """
        Sometimes we identify the tail in a flipped way (the fin as the tip and vise versa).
        This function check whether the tail is more similar to the tail in the previous frame when it is flipped.
        If so, it flips the tail and updates the tail tip. The swimbladder location is set to nan (we can't trust it anymore),
        and will be calculated as an outlier later on.
        """
        flipped_tail_threshold = 100  # magic number. It is more than 0 for cases where the tail is truncated (see comment below)
        frames_len = len(analysis_data_outputs.tail_path_list)

        for frame_idx in range(1, frames_len):
            if not (analysis_data_outputs.tail_tip_status_list[frame_idx] and
                    analysis_data_outputs.tail_tip_status_list[frame_idx - 1]):
                continue
            min_tail_shape = min(analysis_data_outputs.tail_path_list[frame_idx].shape[0],
                                 analysis_data_outputs.tail_path_list[frame_idx - 1].shape[0])
            regular_tail_distance = np.linalg.norm(analysis_data_outputs.tail_path_list[frame_idx][:min_tail_shape, :] -
                                                   analysis_data_outputs.tail_path_list[frame_idx - 1][:min_tail_shape,
                                                   :])
            flipped_tail_distance = np.linalg.norm(
                analysis_data_outputs.tail_path_list[frame_idx][min_tail_shape - 1::-1, :] -
                analysis_data_outputs.tail_path_list[frame_idx - 1][:min_tail_shape, :])
            # Can do better - there are cases where in one frame the tail is truncated from one size, and on the other frame it is truncated from the other size. It might cause the tail to flip.
            # maybe we should first shift the tails to get maximum overlap and then do the comparison of the flip and normal tail.
            if regular_tail_distance - flipped_tail_distance > flipped_tail_threshold:
                analysis_data_outputs.tail_path_list[frame_idx] = analysis_data_outputs.tail_path_list[frame_idx][::-1,
                                                                  :]
                analysis_data_outputs.tail_tip_point_list[frame_idx, :] = analysis_data_outputs.tail_path_list[
                                                                              frame_idx][0, :]
                # analysis_data_outputs.swimbladder_points_list[frame_idx,:] = np.nan #it would be the best to use self.get_swimbladder_from_midline_path() again, but we don't have the frame.
                # current_path_list = analysis_data_outputs.tail_path_list[frame_idx]
                # analysis_data_outputs.swimbladder_points_list[frame_idx,:] = analysis_data_outputs.tail_path_list[frame_idx][int(len(current_path_list)*2/3),:]

    def replace_sb_outlier(self, analysis_data_outputs, outlier_swimbladder_index, swimbladder_points_list,
                           median_swimbladder_distance_from_origin):
        '''
        This function recieves an index of a swimbladder outlier, and return a better swimbladder coordinate for this point.
        '''
        current_path_list = analysis_data_outputs.tail_path_list[outlier_swimbladder_index]
        last_third_of_path_list = int(len(current_path_list) * 2 / 3)
        distances_from_origin_on_path_list = np.linalg.norm(
            current_path_list - analysis_data_outputs.origin_head_points_list[outlier_swimbladder_index], axis=1)
        deviation_from_median = np.abs(distances_from_origin_on_path_list - median_swimbladder_distance_from_origin)

        # old code:
        # deviation_from_median_without_irrelevant_indices = deviation_from_median[last_third_of_path_list:median_swimbladder_index + index_deviation * 2]
        # approximated_swimbladder = np.min(deviation_from_median_without_irrelevant_indices)
        # approximated_swimbladder_index = np.where(deviation_from_median == approximated_swimbladder)[0][0]

        # new code:
        peak_indices = find_peaks(-deviation_from_median, height=-2, distance=5)[0]  # Magic number
        ## YR: maybe its better to write it like "did peak_indices has 0,1 or more matches"
        if peak_indices.shape == (0,):
            if deviation_from_median[
                -1] < 2:  # for a monotonically decreasing function, the last point will be the swimbladder
                peak_indices = np.array([len(deviation_from_median) - 1], dtype='int64')
            else:  ##YR: this is the next bug to fix.
                print(f'Failed to find swimbladder index for frame {outlier_swimbladder_index}.')
                print(
                    'This probably means the top part of the swimbladder is not on the tail contour. Consider changing the height parameter in find_peaks.')
                return []
        if np.isnan(analysis_data_outputs.swimbladder_points_list[outlier_swimbladder_index, 0]):
            ## Option 1: if there is no initial guess for the swimbladder, set it as the last point on the tail.
            ## In the next stage we choose the point closer to the initial guess, and so we will just peak the farthest point from the tip.
            # current_swimbladder_index = len(current_path_list) - 1

            # Option 2: if there is no initial guess for the swimbladder, take the point closest to the swimbladder location in the previous frame
            current_swimbladder_index = np.argmin(
                np.linalg.norm(swimbladder_points_list[outlier_swimbladder_index - 1] - current_path_list, axis=1))
        else:
            current_swimbladder_index = \
                np.where(np.all(current_path_list == swimbladder_points_list[outlier_swimbladder_index], axis=1))[0]
        approximated_swimbladder_index = peak_indices[
            np.argmin(abs(peak_indices - current_swimbladder_index))]  # might be better to do it for last third.

        return current_path_list[approximated_swimbladder_index]

    def swimbladder_post_process(self, analysis_data_outputs):
        swimbladder_points_list = np.array(analysis_data_outputs.swimbladder_points_list)

        swimbladder_indices = []
        problematic_point_indices = []
        nan_outliers = []
        for index, point in enumerate(swimbladder_points_list):
            if np.any(np.isnan(analysis_data_outputs.tail_path_list[index])) or len(
                    analysis_data_outputs.tail_path_list[index]) == 0:
                # for cases where there is no tail information to begin with
                problematic_point_indices.append(index)
                continue
            elif np.isnan(point[0]) or np.isnan(point[1]):
                # When only the swimbladder is missing
                # Explicitly mark the value as an outlier
                nan_outliers.append(index)
                continue

        swimbladder_distance_from_origin = np.linalg.norm(
            swimbladder_points_list - analysis_data_outputs.origin_head_points_list, axis=1)
        swimbladder_distance_from_origin_without_nan = swimbladder_distance_from_origin[
            ~np.isnan(swimbladder_distance_from_origin)]
        median_swimbladder_distance_from_origin = np.median(swimbladder_distance_from_origin_without_nan)
        swimbladder_distance_from_origin[
            nan_outliers] = self.small_fish_max_distance_from_origin * 3  # Just a big value to make sure it's an outlier
        outliers = np.where(np.abs(swimbladder_distance_from_origin - median_swimbladder_distance_from_origin) > np.std(
            swimbladder_distance_from_origin_without_nan) * 2)[0]
        is_swimbladder_point_approximated = [False] * len(
            swimbladder_points_list)  # YR: Shouldn't we set True to all non outliers?
        number_of_corrections_to_run = 5  # should be at least 1.
        # In the first iteration we correct the outliers identified so far. In the next iterations we would choose outliers based on the movement of the swimbladder between frames.
        for run_idx in range(number_of_corrections_to_run):
            for outlier_swimbladder_index in outliers:
                if outlier_swimbladder_index in problematic_point_indices:
                    continue

                approximated_swimbladder = self.replace_sb_outlier(analysis_data_outputs, outlier_swimbladder_index,
                                                                   swimbladder_points_list,
                                                                   median_swimbladder_distance_from_origin)

                if len(approximated_swimbladder) == 0:  # not good guess for swimbladder was found
                    swimbladder_points_list[outlier_swimbladder_index] = np.nan
                    continue

                swimbladder_points_list[outlier_swimbladder_index] = approximated_swimbladder
                is_swimbladder_point_approximated[outlier_swimbladder_index] = True
            distance_between_frames = np.linalg.norm(np.diff(swimbladder_points_list, axis=0), axis=1)
            head_distance_between_frames = np.linalg.norm(
                np.diff(analysis_data_outputs.origin_head_points_list, axis=0), axis=1)
            max_distance_between_frames = 10 * np.ones(distance_between_frames.shape)
            max_distance_between_frames[head_distance_between_frames < 2] = 4
            big_distance_idxs = np.where(np.logical_or(np.abs(distance_between_frames) > max_distance_between_frames,
                                                       np.isnan(distance_between_frames)))[
                                    0] + 1  # 10 is a magic number
            if big_distance_idxs.shape[0] == 0 or run_idx == number_of_corrections_to_run - 1:
                break
            # outliers = list(big_distance_idxs[np.concatenate([np.diff(big_distance_idxs)==1,np.array([False])])])
            outliers = big_distance_idxs  # YR: maybe only run the odd events?
            for outlier_idx in outliers:
                if len(analysis_data_outputs.tail_path_list[outlier_idx]) > 0:
                    swimbladder_points_list[outlier_idx] = analysis_data_outputs.tail_path_list[outlier_idx][int(len(
                        analysis_data_outputs.tail_path_list[outlier_idx]) * 2 / 3), :]
                else:
                    swimbladder_points_list[outlier_idx] = np.nan

        # TODO: make sure to put NaN if the field is not reliable.
        return swimbladder_points_list, is_swimbladder_point_approximated

    def _post_process(self, n_frames: int, analysis_data_outputs = None) -> dict:
        """todo add documentation for tail

        :param input_frames_list:
        :param analysis_data_outputs:
        :return:
        """
        self.detect_flipped_tail(analysis_data_outputs)
        tail_points = np.array(analysis_data_outputs.tail_tip_point_list)
        frames_len = len(tail_points)

        tail_point_differences = np.diff((tail_points[1:], tail_points[0:-1]), axis=0)[0]
        zeros_for_padding = np.zeros(shape=(1, 2), )
        tail_point_differences_padded = np.concatenate([zeros_for_padding, tail_point_differences])
        tail_point_differences_norm = np.linalg.norm(tail_point_differences_padded, axis=1)

        is_tail_point_diff_norm_below_threshold = tail_point_differences_norm < self.tail_point_deviation_threshold
        print("Frames of tail above threshold: ", np.where(is_tail_point_diff_norm_below_threshold == False))

        numerical_derivative = numerical_analysis.get_numerical_derivative(array=tail_points)
        velocity_norms = np.linalg.norm(x=numerical_derivative, axis=1)

        if not self.is_fast_run:
            swimbladder_points_list, is_swimbladder_point_approximated = self.swimbladder_post_process(
                analysis_data_outputs)
        else:
            swimbladder_points_list = np.nan * analysis_data_outputs.swimbladder_points_list
            is_swimbladder_point_approximated = [False] * len(swimbladder_points_list)

        analysis_data_outputs.swimbladder_points_list = swimbladder_points_list

        is_head_angle_diff_norm_below_threshold, is_head_origin_diff_norm_below_threshold = \
            self.__post_processing_find_head_errors(analysis_data_outputs)

        if len(np.where(is_head_origin_diff_norm_below_threshold == False)[0]) > 0:
            print("Frames of head above threshold: Origin ",
                  np.where(is_head_origin_diff_norm_below_threshold == False)[0] + 1)
        if len(np.where(is_head_angle_diff_norm_below_threshold == False)[0]) > 0:
            print("Frames of head above threshold: Angle ",
                  np.where(is_head_angle_diff_norm_below_threshold == False)[0] + 1)

        analysis_data_outputs.interpolated_tail_path = []
        analysis_data_outputs.tip_to_swimbladder_distance = []
        for frame_id in range(len(analysis_data_outputs.tail_path_list)):
            tip_to_swimbladder_distance_per_frame, interpolated_tail_path_per_frame = self.get_smooth_tail(
                analysis_data_outputs.tail_path_list[frame_id], swimbladder_points_list[frame_id])
            analysis_data_outputs.interpolated_tail_path.append(interpolated_tail_path_per_frame)
            analysis_data_outputs.tip_to_swimbladder_distance.append(tip_to_swimbladder_distance_per_frame)

        start_frames, end_frames, is_bout_frame_list = self.bout_detection(analysis_data_outputs.interpolated_tail_path)

        self.__reset_history()
        return {'is_bout_frame_list': is_bout_frame_list,
                'bout_start_frames': start_frames,
                'bout_end_frames': end_frames,
                'velocity_norms': np.array(velocity_norms),
                'swimbladder_points_list': swimbladder_points_list,
                'tip_to_swimbladder_distance': analysis_data_outputs.tip_to_swimbladder_distance,
                'is_swimbladder_point_approximated': np.array(is_swimbladder_point_approximated),
                'is_tail_point_diff_norm_below_threshold': np.array(is_tail_point_diff_norm_below_threshold),
                'is_head_origin_diff_norm_below_threshold': is_head_origin_diff_norm_below_threshold,
                'is_head_angle_diff_norm_below_threshold': is_head_angle_diff_norm_below_threshold,
                'interpolated_tail_path': analysis_data_outputs.interpolated_tail_path}

    @staticmethod
    def bout_detection(interpolated_tail_list, svd_comp=None, svd_thresh=None, svd_merge_thresh=None,
                       fixed_fish=False, friend_fish=False):
        """Detect start and end points of bouts in a given event, from the interpolated tail data, using the
        mean squared velocity of points along the end of the tail.

        :return: lists of start points and end points with matching indices of the bouts, a logical list of
         is_bout_frame_list
        """

        def smooth(a, wsz):
            ret = np.nancumsum(a, dtype=float)
            n = np.ones(len(a))
            n[np.isnan(a)] = 0
            n[0] = 1
            number_of_valid_elements = np.nancumsum(n)
            number_of_valid_elements[wsz:] = number_of_valid_elements[wsz:] - number_of_valid_elements[:-wsz]
            ret[wsz:] = ret[wsz:] - ret[:-wsz]
            start = np.nancumsum(a[:wsz - 1])[::2] / number_of_valid_elements[:wsz - 1:2]
            stop = (np.nancumsum(a[:-wsz:-1])[::2] / number_of_valid_elements[:wsz - 1:2])[::-1]
            middle = ret[wsz - 1:] / number_of_valid_elements[wsz - 1:]
            return np.concatenate((start, middle, stop))

        # thresholds - determined from examining different events.
        # bout duration, NaNs and std thresholds are used to determine false detections

        if friend_fish:
            is_bout_thresh = 100
            edge_thresh = 0.1
            bout_duration_thresh = 16  # [frames]
            std_noise_thresh = 10
            std_merge_thresh = 10000
            bout_nans_thresh = 0.25
            total_nans_thresh = 0.4
        else:
            is_bout_thresh = 100
            edge_thresh = 0.1
            bout_duration_thresh = 15  # [frames]
            std_noise_thresh = 10
            std_merge_thresh = 10000
            bout_nans_thresh = 0.25
            total_nans_thresh = 0.4

        interpolated_tail_3d = np.concatenate([frame[np.newaxis, :, :] for frame in interpolated_tail_list])
        x = interpolated_tail_3d[:, :, 0]
        y = interpolated_tail_3d[:, :, 1]
        dx = np.diff(x, n=1, axis=0)
        dy = np.diff(y, n=1, axis=0)
        tail_velocity = dx ** 2 + dy ** 2
        avg_tail_velocity = np.mean(tail_velocity[:, 1:15], 1)
        velocity_std = np.std(tail_velocity[:, 1:15], 1)
        smoothed_std = smooth(smooth(velocity_std, 9), 9)

        start_frames = []
        end_frames = []
        is_bout = False

        # if the fraction of NaN frames is larger than total_nans_thresh, the confidence in the detection is low,
        # so we return empty lists and list of zeros for is_bout_frame_list
        total_nans_percentage = sum(np.isnan(avg_tail_velocity)) / len(avg_tail_velocity)
        if total_nans_percentage > total_nans_thresh:
            is_bout_frame_vec = np.zeros(len(avg_tail_velocity) + 1, dtype=bool)
            return start_frames, end_frames, list(is_bout_frame_vec)

        # find the suspected bouts by the frames in which tail velocity exceeded
        # the threshold then find the start and end points in which the velocity is
        # below a lower threshold
        for frame in range(len(avg_tail_velocity)):
            if avg_tail_velocity[frame] > is_bout_thresh and not is_bout:
                is_bout = True
                start_frame_sus = np.intersect1d(np.argwhere(avg_tail_velocity[0:frame + 1] < edge_thresh),
                                                 np.argwhere(avg_tail_velocity[0:frame + 1] > 0))
                if len(start_frame_sus) > 0:
                    start_frame = np.max(start_frame_sus)
                    if start_frame in end_frames:
                        start_frames.append(start_frame + 1)
                    else:
                        start_frames.append(start_frame)
                else:
                    start_frames.append(frame)
            elif 0 < avg_tail_velocity[frame] < edge_thresh and is_bout:
                is_bout = False
                end_frames.append(frame)

        # if a start point was recognized without an end point, declare the last frame as an end point
        # 16.11.23 bug-fix: changed the addition to be the last frame that is not a nan instead of the last frame.
        # in case that the last valid frame turns out to be the same as the start frame, remove it
        if len(start_frames) > len(end_frames):
            # end_frames.append(len(avg_tail_velocity))
            last_valid_frame = np.argwhere(~np.isnan(avg_tail_velocity))[-1][0]
            if start_frames[-1] == last_valid_frame:
                start_frames = start_frames[0:-1]
            else:
                end_frames.append(last_valid_frame)
        # if an end point was recognized without a start point, declare the second frame as a start point
        if len(end_frames) > len(start_frames):
            start_frames.insert(0, 2)

        for repeat in range(2):
            # split suspicious bouts which appear to consist more than one bout - for long bouts(more than 70 frames),
            # look for more than one peak (with MinPeakProminence and MinPeakDistance).
            # if there is more than one, find the lowest point between them and add a new end and start frames,
            # effectively splitting the out
            init_bouts_num = len(end_frames)
            for bout in range(init_bouts_num):
                if end_frames[bout] - start_frames[bout] > 65:
                    if bout == 0:
                        max_peaks_frames, _ = find_peaks(smoothed_std[0:end_frames[bout]], prominence=2.5, distance=20)
                    else:
                        max_peaks_frames, _ = find_peaks(smoothed_std[start_frames[bout]:end_frames[bout]],
                                                         prominence=2.5, distance=20)
                    if len(max_peaks_frames) > 1:
                        max_peaks_frames = max_peaks_frames + start_frames[bout]
                        min_peaks_frames, _ = find_peaks(-smoothed_std[max_peaks_frames[0]:max_peaks_frames[-1]],
                                                         prominence=2.5)
                        min_peaks_frames = min_peaks_frames + max_peaks_frames[0]
                        for min_frame in min_peaks_frames:
                            if smoothed_std[min_frame] * 2.5 < np.min(smoothed_std[max_peaks_frames]):
                                end_frames.append(min_frame)
                                start_frames.append(min_frame + 1)
            start_frames.sort()
            end_frames.sort()

            # if there is an end point which is the same as the next start point, check if is real or due to an artifact
            # through the value of the std. large std means that we are inside a bout,
            # low std means we are between bouts. if real - keep both, if not - delete both.
            for bout in range(len(end_frames) - 1):
                if (end_frames[bout] + 1 == start_frames[bout + 1] and smoothed_std[end_frames[bout]]
                > std_merge_thresh) or start_frames[bout] >= end_frames[bout]:
                    end_frames[bout] = -1
                    start_frames[bout + 1] = -1
            start_frames = [frame for frame in start_frames if frame >= 0]
            end_frames = [frame for frame in end_frames if frame >= 0]

        # if after all there was no start or end points found, return empty lists - currently shouldn't happen
        if not start_frames or not end_frames:
            is_bout_frame_vec = np.zeros(len(avg_tail_velocity) + 1, dtype=bool)
            return start_frames, end_frames, list(is_bout_frame_vec)

        # deletion of wrong detections:
        # a. too many NaNs in the detected bout
        # b. bouts which are too short
        # c. bouts in which the variance (std) is too low
        # d. bouts for which the peak of the SVD is too short (3 frames)
        if svd_comp is not None:
            svd_envelope = np.sqrt(svd_comp[0, :] ** 2 + svd_comp[1, :] ** 2)
            svd_peaks, _ = find_peaks(smooth(svd_envelope, 9), distance=15, prominence=svd_thresh * 1.5,
                                      width=3)  # width and distance are in [frames], prominence


        else:
            svd_peaks = None
            svd_envelope = None

        for bout in tqdm(range(len(end_frames))):
            bout_nans_num = sum(np.isnan(avg_tail_velocity[start_frames[bout]:end_frames[bout]]))
            bout_nans_percentage = bout_nans_num / len(avg_tail_velocity[start_frames[bout]:end_frames[bout]])
            if bout_nans_percentage > bout_nans_thresh:
                start_frames[bout] = -1
                end_frames[bout] = -1
            if (end_frames[bout] - start_frames[bout]) < bout_duration_thresh:
                start_frames[bout] = -1
                end_frames[bout] = -1
            elif np.max(smoothed_std[start_frames[bout]:end_frames[bout]]) < std_noise_thresh:
                start_frames[bout] = -1
                end_frames[bout] = -1

            if svd_peaks is not None:
                if not any([frame for frame in svd_peaks if frame in np.arange(start_frames[bout], end_frames[bout])]):  # remove bouts in which the SVD peak doesn't meet the criterion
                    start_frames[bout] = -1
                    end_frames[bout] = -1

        start_frames = [frame for frame in start_frames if frame >= 0]
        end_frames = [frame for frame in end_frames if frame >= 0]

        # Merge bouts with overlapping end frames based on SVD values, then correct start and end frames
        if svd_envelope is not None:
            if svd_merge_thresh is not None:
                for bout in range(len(end_frames) - 1):
                    if start_frames[bout + 1] - end_frames[bout] <= 3 and svd_envelope[end_frames[
                        bout]] > svd_merge_thresh:  # merge bouts that were incorrectly split (up to suspected IBI of 5 frames)
                        end_frames[bout] = -1
                        start_frames[bout + 1] = -1
                start_frames = [frame for frame in start_frames if frame >= 0]
                end_frames = [frame for frame in end_frames if frame >= 0]

            backtrack_window = round(bout_duration_thresh / 2)  # frames, 8
            for bout in tqdm(range(len(end_frames))):
                # move start frames according to the svd envelope
                if bout == 0 and start_frames[bout] < round(backtrack_window / 2):  # first bout is a special case
                    start_min_val = np.nanmin(svd_envelope[0:(start_frames[bout] + backtrack_window)])
                    start_min_frame = np.nanargmin(svd_envelope[0:(start_frames[bout] + backtrack_window)])
                    if start_min_val < svd_envelope[start_frames[bout]]:
                        start_frames[bout] = start_frames[bout] + start_min_frame
                elif bout == 0 and start_frames[bout] >= round(backtrack_window / 2):
                    start_min_val = np.nanmin(svd_envelope[(start_frames[bout] - round(backtrack_window / 2)):(
                            start_frames[bout] + backtrack_window)])
                    start_min_frame = np.nanargmin(svd_envelope[(start_frames[bout] - round(backtrack_window / 2)):(
                            start_frames[bout] + backtrack_window)])
                    if start_min_val < svd_envelope[start_frames[bout]]:
                        start_frames[bout] = start_frames[bout] + start_min_frame - round(backtrack_window / 2)
                elif bout != 0:
                    if start_frames[bout] - round(backtrack_window / 2) > end_frames[bout - 1]:
                        start_min_val = np.nanmin(svd_envelope[(start_frames[bout] - round(backtrack_window / 2)):(
                                start_frames[bout] + backtrack_window)])
                        start_min_frame = np.nanargmin(svd_envelope[(start_frames[bout] - round(backtrack_window / 2)):(
                                start_frames[bout] + backtrack_window)])
                        if start_min_val < svd_envelope[start_frames[bout]]:
                            start_frames[bout] = start_frames[bout] + start_min_frame - round(backtrack_window / 2)
                    else:
                        start_min_val = np.nanmin(
                            svd_envelope[(end_frames[bout - 1] + 1):(start_frames[bout] + backtrack_window)])
                        start_min_frame = np.nanargmin(
                            svd_envelope[(end_frames[bout - 1] + 1):(start_frames[bout] + backtrack_window)])
                        if start_min_val < svd_envelope[start_frames[bout]]:
                            start_frames[bout] = start_frames[bout] + start_min_frame - (
                                    start_frames[bout] - end_frames[bout - 1]) + 1

                if bout < len(end_frames) - 1:
                    # move end frames according to the svd envelope
                    min_frame = np.nanargmin(
                        svd_envelope[(end_frames[bout] - backtrack_window):(start_frames[bout + 1])])
                    if min_frame < backtrack_window:  # cases where there is a minimum point of the SVD BEFORE the original detected end frame
                        end_frames[bout] = end_frames[
                                               bout] + min_frame - backtrack_window  # backtrack to a possibility of shortening the bout
                    elif any(np.where(svd_envelope[(end_frames[bout]):(start_frames[bout + 1])] <= svd_thresh)[
                                 0]):  # find the first point to cross the svd threshold and extend the bout
                        frames_to_add = np.nanargmax(
                            svd_envelope[end_frames[bout]:(start_frames[bout + 1])] < svd_thresh)
                        end_frames[bout] = end_frames[
                                               bout] + frames_to_add  # in cases where there are no points below the threshold, retain the original detection
                else:  # final bout, avoid exceeding the last frame
                    min_frame = np.nanargmin(svd_envelope[(end_frames[bout] - backtrack_window):])
                    if min_frame < backtrack_window:  # cases where there is a minimum point of the SVD BEFORE the original detected end frame
                        end_frames[bout] = end_frames[
                                               bout] + min_frame - backtrack_window  # backtrack to a possibility of shortening the bout
                    elif end_frames[bout] + np.nanargmax(svd_envelope[end_frames[bout]:] < svd_thresh) < len(
                            avg_tail_velocity):
                        end_frames[bout] = end_frames[bout] + np.nanargmax(svd_envelope[end_frames[bout]:] < svd_thresh)
                    else:
                        end_frames[bout] = len(avg_tail_velocity)  # final frame

        # correct to MatLab indexing
        start_frames = [frame + 1 for frame in start_frames]
        end_frames = [frame + 1 for frame in end_frames]

        # if an end point was recognized without a start point, declare the second frame as a start point
        if len(end_frames) > len(start_frames):
            start_frames.insert(0, 2)

        # bouts_num = len(start_points)
        # create is_bout_frame_list, we do this after we ensure we have the right bouts to avoid revising the vector
        # adding 1 to correct for the missing frame that comes from the velocity calculation (last frame always 0)
        is_bout = False
        is_bout_frame_vec = np.zeros(len(avg_tail_velocity) + 1, dtype=bool)
        for frame in range(len(avg_tail_velocity)):
            if frame in start_frames:
                is_bout = True
                is_bout_frame_vec[frame] = 1
            elif frame in end_frames:
                is_bout = False
            elif is_bout:
                is_bout_frame_vec[frame] = 1

        return start_frames, end_frames, list(is_bout_frame_vec)

    @staticmethod
    def calc_points_diff(points):
        point_differences = np.diff(points, axis=0)
        point_differences_norm = np.linalg.norm(np.concatenate([point_differences[0:1, :], point_differences]), axis=1)
        return point_differences_norm

    @staticmethod
    def fill_nan_2d_interpolate(data_2d, kind='previous'):
        if np.isnan(data_2d).any():
            data_2d[:, 0] = ContourBasedTracking.fill_nan_1d_interpolate(data_2d[:, 0], kind=kind)
            data_2d[:, 1] = ContourBasedTracking.fill_nan_1d_interpolate(data_2d[:, 1], kind=kind)
        return data_2d

    @staticmethod
    def fill_nan_1d_interpolate(data_1d, kind='previous'):
        nans, x = np.isnan(data_1d), lambda z: z.nonzero()[0]
        if sum(~nans) >= 1:  # minimum needed to use interp1d (more accurate interpolation)
            data_1d[nans] = np.interp(x(nans), x(~nans), data_1d[~nans])
        if sum(~nans) >= 2:  # minimum needed to use interp1d (more accurate interpolation)
            f = interp1d(x(~nans), data_1d[~nans], kind=kind, fill_value="extrapolate")
            data_1d[nans] = f(x(nans))
        return data_1d

    from scipy.interpolate import interp1d

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

    def _analyse(self, input_frame: np.array, noise_frame: np.array, fps: float, frame_number: int, additional=None):
        """

        :param input_frame:
        :param noise_frame:
        :param fps:
        :param frame_number:
        :param additional: not used here. List with inputs from other trackers etc
        :return: annotated frame (for debug) & output struct
        """
        output = FrameAnalysisData()

        if self.should_remove_plate:
            cleaned = self.clean_plate_noise(input_frame, frame_number, scale=self.scale)
            if cleaned is None:  # error
                if self.save_n_back is not None and self.save_n_back > 0:
                    self.save_history(output,
                                      frame_number=frame_number)  # append nan that will be filled in post-proc. todo- use history to find?
                return input_frame, output  # default is_ok = False
        else:
            cleaned = input_frame.copy()

        an_frame = cleaned.copy()  # output frame

        # Step 1- fish contour - return with error if incorrect
        fish_contour = self.get_fish_contour(cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.uint8),
                                             scale=self.scale)
        if fish_contour is None:
            logging.debug("Frame " + str(frame_number) + " didn't find fish")
            if self.save_n_back is not None and self.save_n_back > 0:
                self.save_history(output,
                                  frame_number=frame_number)  # append nan that will be filled in post-proc. todo- use history to find?
            return an_frame, output  # default is_ok = False
        output.fish_contour = fish_contour

        # Create fish mask from contour (both fish and background (non fish)
        cleaned_fish, cleaned_non_fish, segment, mask, expanded_mask = \
            self.get_fish_segment_and_masks(an_frame, cleaned, fish_contour)
        output.fish_segment = segment.transpose(0, 2, 1).reshape(-1,
                                                                 2)  # output shape (flatten): (#p, 2) - todo can remove?

        # Step 2- eyes - return with error if incorrect
        eyes_data, no_eyes_data = self.get_eyes_no_eyes_within_fish_segment(an_frame, cleaned_fish, frame_number,
                                                                            # scale = higher zoom with camera.
                                                                            # This increases fish area & also details
                                                                            # look better so we reduce eye color limit
                                                                            scale=self.scale,
                                                                            # todo this max_color is not good
                                                                            max_color=80 - abs(self.scale - 1) * 6)
        output.eyes_contour = [c['contour'] for c in eyes_data]

        if len(eyes_data) == 0 or len(eyes_data) == 1:
            logging.debug("Frame " + str(frame_number) + " didn't find eyes ({0} found)".format(len(eyes_data)))
            if not self.is_predicting_missing_eyes:  # error
                if self.save_n_back is not None and self.save_n_back > 0:
                    self.save_history(output, frame_number=frame_number)
                cv2.drawContours(an_frame, [output.fish_contour], -1, Colors.RED, 1)  # drawn here for debugging
                return an_frame, output  # default is_ok = False
            output.fish_head_origin_point = np.array([np.nan, np.nan])
            output.fish_head_destination_point = np.array([np.nan, np.nan])
        else:
            eyes_data, p_from, p_to, _, _, _ = \
                self.calc_fish_direction_from_eyes(cleaned_fish, eyes_data, frame_number, output, scale=self.scale)

            output.fish_head_origin_point = np.array(p_from)
            output.fish_head_destination_point = np.array(p_to)

            # Eyes difference and hunting
            output.eyes_data = FrameAnalysisData.EyesData(eyes_data, output.fish_head_origin_point,
                                                          output.fish_head_destination_point)
            # Eyes found - marked as ok for later
            output.is_ok = True

            cleaned_fish, is_paramecia_connected = \
                self.fix_paramecia_connect_to_fish_mouth(cleaned_fish, output.fish_head_origin_point,
                                                         output.fish_head_destination_point, eyes_data,
                                                         frame_number)
            if is_paramecia_connected:
                output.fish_contour = \
                    self.get_fish_contour(cv2.cvtColor(cleaned_fish, cv2.COLOR_BGR2GRAY).astype(np.uint8),
                                          scale=self.scale)
                # Create fish mask from contour (both fish and background (non fish)
                _, _, segment, _, _ = \
                    self.get_fish_segment_and_masks(an_frame, cleaned_fish, output.fish_contour)
                output.fish_segment = segment.transpose(0, 2, 1).reshape(-1,
                                                                         2)  # output shape (flatten): (#p, 2) - todo can remove?
                cv2.drawContours(an_frame, [output.fish_contour], -1, Colors.BLUE, 1)  # drawn here for debugging
            else:
                cv2.drawContours(an_frame, [output.fish_contour], -1, Colors.RED, 1)  # drawn here for debugging

        # If saving history- use to fix errors of jumping eyes
        if self.save_n_back is not None:
            origin_points = self.history_analysis_data_outputs.origin_head_points_list.copy()
            origin_points[self.history_analysis_data_outputs.is_head_prediction_list, :] = np.nan
            if 0 < self.min_valid_n_back <= sum(~np.isnan(origin_points)[:, 0]):
                self.validate_and_fix_head_data(self.history_analysis_data_outputs, output,
                                                frame_number)  # todo recalc eye contours
                output.is_ok = True

        if self.save_n_back is not None and self.save_n_back > 0:  # todo prediction?
            self.save_history(output, frame_number=frame_number)

        if not output.is_ok:
            return an_frame, output  # default is_ok = False

        # Tail segmentation and tip
        if not self.is_fast_run:
            midline_path = self.get_midline_as_list_of_connected_points(cleaned_fish, output)
            if midline_path is not None:
                tail_tip = np.array(midline_path[0])
                midline_path = np.array(midline_path)
                fish_size = np.median(self.prev_head_tail_distance) if len(
                    self.prev_head_tail_distance) > 15 else self.small_fish_size
                swimbladder_point_index = self.get_swimbladder_from_midline_path(cleaned_fish, midline_path, output,
                                                                                 fish_size)
                if swimbladder_point_index is not None:
                    swimbladder_point = (
                        midline_path[swimbladder_point_index][0], midline_path[swimbladder_point_index][1])
                else:
                    swimbladder_point = (np.nan, np.nan)
                output.tail_data = FrameAnalysisData.TailData(fish_tail_tip_point=tail_tip, tail_path=midline_path,
                                                              swimbladder_point=swimbladder_point)
                if len(self.prev_head_tail_distance) < 100:
                    self.prev_head_tail_distance.append(np.linalg.norm(tail_tip - output.fish_head_origin_point))
            else:
                logging.debug("Frame " + str(frame_number) + " failed to find midline")
                output.tail_data = None

        self.draw_output_on_annotated_frame(an_frame, output,
                                            hunt_threshold=self.hunt_threshold, redraw_fish_contours=False)

        if self.is_twin_view:  # show original and an side-by-side
            return np.hstack([input_frame, an_frame]), output
        else:
            return an_frame, output

    def fix_paramecia_connect_to_fish_mouth(self, cleaned_fish, fish_head_origin_point, fish_head_destination_point,
                                            eyes_data_dict, frame_number):
        is_paramecia_connected = False
        head_direction_angle = get_angle_to_horizontal(fish_head_origin_point, fish_head_destination_point)
        angle = 360 - (head_direction_angle + 90)  # clockwise from y axis

        # mask half circle around the eyes, in head direction (find mouth only)
        mmask_head = np.full((cleaned_fish.shape[0], cleaned_fish.shape[1]), 0, dtype=np.uint8)
        cv2.ellipse(mmask_head, self.point_to_int(fish_head_origin_point), (30, 30),
                    angle=angle, startAngle=0, endAngle=180, color=Colors.WHITE, thickness=cv2.FILLED)
        cleaned_fish_head = cv2.bitwise_and(cleaned_fish, cleaned_fish, mask=mmask_head)
        fish_head_contour = self.get_fish_contour(cleaned_fish_head, close_kernel=(5, 5), scale=self.scale)
        # self.show("head", cleaned_fish_head)
        if fish_head_contour is None or len(fish_head_contour) == 0:
            logging.error("frame {0}: can't check paramecia in mouth (empty contour for mask)".format(frame_number))
            return cleaned_fish, is_paramecia_connected

        el = cv2.fitEllipse(fish_head_contour)
        if cv2.contourArea(cv2.convexHull(fish_head_contour)) > 400 or min(el[1]) > 20:
            is_paramecia_connected = True
            mask_head = np.full((cleaned_fish.shape[0], cleaned_fish.shape[1]), 0, dtype=np.uint8)
            cv2.drawContours(mask_head, [fish_head_contour], -1, Colors.WHITE, cv2.FILLED)
            for e in eyes_data_dict:  # fill eye holes using ellipse fit
                cv2.ellipse(mask_head, e['ellipse'], color=Colors.WHITE, thickness=cv2.FILLED)
            mask_head = cv2.morphologyEx(mask_head, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))  # without paramecia
            cv2.ellipse(mask_head, self.point_to_int(fish_head_origin_point), (400, 400),
                        angle=angle, startAngle=180, endAngle=360, color=Colors.WHITE, thickness=cv2.FILLED)
            cleaned_fish = cv2.bitwise_and(cleaned_fish, cleaned_fish, mask=mask_head)

        return cleaned_fish, is_paramecia_connected

    def save_history(self, output, frame_number):
        is_nans = np.isnan(self.history_analysis_data_outputs.origin_head_points_list[:self.save_n_back, 0])
        if self.count_n_back >= self.save_n_back:  # cyclic rotation (constant length list)
            self.history_analysis_data_outputs.origin_head_points_list = \
                np.roll(self.history_analysis_data_outputs.origin_head_points_list, -1, axis=0)
            self.history_analysis_data_outputs.destination_head_points_list = \
                np.roll(self.history_analysis_data_outputs.destination_head_points_list, -1, axis=0)
            self.history_analysis_data_outputs.is_head_prediction_list = \
                np.roll(self.history_analysis_data_outputs.is_head_prediction_list, -1, axis=0)

            self.history_analysis_data_outputs.origin_head_points_list[-1, :] = np.nan
            self.history_analysis_data_outputs.destination_head_points_list[-1, :] = np.nan
            self.history_analysis_data_outputs.is_head_prediction_list[-1] = False
            ind = -2  # last one is saved as nan
        else:
            self.count_n_back += 1
            ind = np.where(is_nans)[0][0]
        self.history_analysis_data_outputs.origin_head_points_list[ind, :] = output.fish_head_origin_point
        self.history_analysis_data_outputs.destination_head_points_list[ind, :] = output.fish_head_destination_point
        self.history_analysis_data_outputs.is_head_prediction_list[ind] = output.is_prediction

    @classmethod
    def draw_output_on_annotated_frame(cls, an_frame, output: FrameAnalysisData,
                                       redraw_fish_contours=True, hunt_threshold=25,
                                       is_bout=None, velocity_norms=None, is_presentation=False,
                                       # font etc
                                       row_left_side_text=50, col_left_side_text=20, space_between_text_rows=25,
                                       col_right_side_text=680, row_right_side_text=20,
                                       text_font=cv2.FONT_HERSHEY_SIMPLEX, bold=2):
        """Add all elements on video (except for fish and eyes contours which are added before for debug

        :param is_presentation: when true, adjust to cleaner frame result (remove some metadata, make lines bolder)
        :param redraw_fish_contours: online, this is created outside of this function (for debug, in cade of errors)
        :param is_bout: True/False to annotate online, None to ignore
        :param velocity_norms: float to annotate online, None to ignore
        :param bold: for text
        :param text_font: for text
        :param hunt_threshold:
        :param an_frame: input frame
        :param output: fish analysis output struct for the annotations
        :param col_right_side_text, row_right_side_text: location of text beginning (right side)
        :param col_left_side_text, row_left_side_text: location of text beginning (left side)
        :param space_between_text_rows:
        :return: an_frame: frame with fish movie annotations
        """
        head_direction_angle = get_angle_to_horizontal(output.fish_head_origin_point,
                                                       output.fish_head_destination_point)
        if redraw_fish_contours:
            width = 2 if is_presentation else 1
            cv2.drawContours(an_frame, [output.fish_contour], -1, Colors.RED, width)
            cv2.drawContours(an_frame, output.eyes_contour, -1, color=Colors.CYAN)

        if output.eyes_data is not None:
            diff = output.eyes_data.diff_from_fish_direction_deg
            eye_angles = output.eyes_data.abs_angle_deg
            if not is_presentation:
                cv2.putText(an_frame, "|E-D|: ({0:.2f},{1:.2f})".format(diff[0], diff[1]),
                            (col_right_side_text, row_right_side_text), text_font, 0.6, Colors.GREEN, bold)
                row_right_side_text += space_between_text_rows
                cv2.putText(an_frame, "E: ({0:.2f},{1:.2f})".format(eye_angles[0], eye_angles[1]),
                            (col_right_side_text, row_right_side_text), text_font, 0.6, Colors.GREEN, bold)
                row_right_side_text += space_between_text_rows

        if not is_presentation:
            cv2.putText(an_frame, "Dir: {0:.2f}".format(head_direction_angle),
                        (col_right_side_text, row_right_side_text), text_font, 0.6, Colors.GREEN, bold)
            row_right_side_text += space_between_text_rows

        if output.eyes_data is not None:
            if not is_presentation:
                result, color = "No-hunt", Colors.GREEN
                if cls.is_hunting(hunt_threshold, output):
                    result, color = "Hunt", Colors.PINK
                cv2.putText(an_frame, result, (col_left_side_text, row_left_side_text), text_font, 0.6, color, bold)
                row_left_side_text += space_between_text_rows

            # draw ellipse majors
            ellipse: EllipseData
            for ellipse in output.eyes_data.ellipses:
                xc, yc = ellipse.ellipse_center
                angle, rmajor = ellipse.ellipse_direction, ellipse.ellipse_major
                angle = angle - 90 if angle > 90 else angle + 90
                xtop, ytop = xc + math.cos(math.radians(angle)) * rmajor, yc + math.sin(math.radians(angle)) * rmajor
                xbot, ybot = xc + math.cos(math.radians(angle + 180)) * rmajor, yc + math.sin(
                    math.radians(angle + 180)) * rmajor
                cv2.line(an_frame, (int(xtop), int(ytop)), (int(xbot), int(ybot)), Colors.CYAN)
            # draw ellipse centers
            # centers = [ellipse.ellipse_center for ellipse in output.eyes_data.ellipses]
            # cv2.circle(an_frame, cls.point_to_int(centers[0]), 1, Colors.CYAN, thickness=cv2.FILLED)
            # cv2.circle(an_frame, cls.point_to_int(centers[1]), 1, Colors.CYAN, thickness=cv2.FILLED)

        # Draw direction and points
        cv2.circle(an_frame, cls.point_to_int(output.fish_head_origin_point), 2, Colors.CYAN, thickness=cv2.FILLED)
        cv2.arrowedLine(an_frame, cls.point_to_int(output.fish_head_origin_point),
                        cls.point_to_int(output.fish_head_destination_point),
                        Colors.YELLOW if output.is_prediction else Colors.RED,
                        thickness=2 if is_presentation else 1)

        if output.tail_data is not None:
            tail: FrameAnalysisData.TailData = output.tail_data
            cv2.circle(an_frame, cls.point_to_int(tail.tail_tip_point), 2, Colors.YELLOW, thickness=cv2.FILLED)
            if not is_presentation:
                cv2.putText(an_frame, 'T: {0:.2f} {1:.2f}'.format(tail.tail_tip_point[0], tail.tail_tip_point[1]),
                            (col_right_side_text, row_right_side_text), text_font, 0.6, Colors.GREEN, bold)
                row_right_side_text += space_between_text_rows

            midline_path = tail.tail_path
            for index in range(1, len(midline_path)):
                first_point = tuple(midline_path[index - 1])
                second_point = tuple(midline_path[index])
                cv2.line(an_frame, first_point, second_point, Colors.GREEN, 1)

            current_swimbladder_point = tail.swimbladder_point
            if not np.isnan(current_swimbladder_point).any():
                cv2.circle(img=an_frame, center=cls.point_to_int(current_swimbladder_point), radius=2,
                           color=Colors.BLUE, thickness=cv2.FILLED)

        if is_bout is not None and not is_presentation:
            result, color = "No-Bout", Colors.GREEN
            if is_bout:
                result, color = "Bout", Colors.PINK
            cv2.putText(an_frame, result, (col_left_side_text, row_left_side_text), text_font, 0.6, color, bold)
            row_left_side_text += space_between_text_rows

        if velocity_norms is not None and not is_presentation:
            cv2.putText(an_frame, 'V: {0:.2f}'.format(velocity_norms),
                        (col_right_side_text, row_right_side_text), text_font, 0.6, Colors.GREEN, bold)
            row_right_side_text += space_between_text_rows

    @staticmethod
    def is_hunting(hunt_threshold, output):
        if output.eyes_data is None:
            return False
        return len([d for d in output.eyes_data.diff_from_fish_direction_deg if abs(d) >= 10]) == 2 \
            and np.mean(np.abs(output.eyes_data.diff_from_fish_direction_deg)) >= hunt_threshold \
            and sum(np.sign(output.eyes_data.diff_from_fish_direction_deg)) == 0

    @staticmethod
    def validate_and_fix_head_data(analysis_data_outputs, output: FrameAnalysisData,
                                   frame_number):
        check_origin = analysis_data_outputs.origin_head_points_list.copy()
        check_origin[analysis_data_outputs.is_head_prediction_list, :] = np.nan
        check_dest = analysis_data_outputs.destination_head_points_list.copy()
        check_dest[analysis_data_outputs.is_head_prediction_list, :] = np.nan
        origin_head_points = ContourBasedTracking.fill_nan_2d_interpolate(check_origin.copy(),
                                                                          kind='linear')  # quadratic?
        dest_head_points = ContourBasedTracking.fill_nan_2d_interpolate(check_dest.copy(), kind='linear')
        if not np.isnan(output.fish_head_origin_point).any():
            origin_head_points[-1, :] = output.fish_head_origin_point
        if not np.isnan(output.fish_head_destination_point).any():
            dest_head_points[-1, :] = output.fish_head_destination_point

        head_direction_angles = [get_angle_to_horizontal(orig, dest)
                                 for (orig, dest) in zip(origin_head_points, dest_head_points)]

        is_head_origin_diff_norm_below_threshold = \
            abs(np.diff(
                ContourBasedTracking.calc_points_diff(origin_head_points))) < 12  # self.head_origin_deviation_threshold

        # angle can jump to 360 if cross the edge, but head flip should be around 180 degrees
        is_head_angle_diff_norm_below_threshold = np.bitwise_or(abs(360 - abs(np.diff(head_direction_angles))) < 90,
                                                                abs(np.diff(head_direction_angles)) < 90)

        # quickfix: idenify both frame and after it as change: append zero and use diff equals 1 to identify location
        # (using int -1 marks false->true, and 1 marks true->false
        is_head_origin_diff_norm_below_threshold = \
            np.diff(np.concatenate([np.zeros(shape=(1,)), is_head_origin_diff_norm_below_threshold.astype(int)])) != -1
        is_head_angle_diff_norm_below_threshold = \
            np.diff(np.concatenate([np.zeros(shape=(1,)), is_head_angle_diff_norm_below_threshold.astype(int)])) != -1

        if not is_head_origin_diff_norm_below_threshold[-1] or not is_head_angle_diff_norm_below_threshold[-1] or \
                np.isnan(output.fish_head_origin_point[-1]) or np.isnan(output.fish_head_destination_point[-1]):
            reason = "empty eyes" if (np.isnan(output.fish_head_origin_point[-1]) or
                                      np.isnan(output.fish_head_destination_point[-1])) else "jump"
            logging.debug("Frame {0} using prediction due to {1}".format(frame_number, reason))

            if reason == "empty eyes" or (not is_head_origin_diff_norm_below_threshold[-1]):
                origin_head_points = \
                    ContourBasedTracking.fill_nan_2d_interpolate(analysis_data_outputs.origin_head_points_list.copy(),
                                                                 'linear')

            dest_head_points = \
                ContourBasedTracking.fill_nan_2d_interpolate(analysis_data_outputs.destination_head_points_list.copy(),
                                                             'linear')
            output.fish_head_origin_point = origin_head_points[-1, :].copy()
            output.fish_head_destination_point = dest_head_points[-1, :].copy()
            output.is_prediction = True

    @staticmethod
    def __post_processing_is_diff_below_threshold_fix(is_diff_below_threshold, origin_head_points, deviation_threshold,
                                                      name_for_debug="Origin", use_start_end_segmentation=False):
        """Fix issues with simple boolean mark of threshold:
        1. when calculating diff (change), it is returned as twice for single point that jumps
        2. when there is a segment of "jumps" (continuous or with "holes" of correct values), the frames within are not
        identified as false (the diff from previous value is small).

        :param is_diff_below_threshold: list of boolean values to fix
        :param origin_head_points: used to make sure the segment is correctly found
        :param is_diff_below_threshold:
        :param is_diff_below_threshold:
        :return:
        """
        # due to the way diff is calculated, we always have False for increase and False for one index after decrease
        start_end_indices = np.where(np.bitwise_not(is_diff_below_threshold))[0]
        if use_start_end_segmentation and (len(start_end_indices) > 0 and len(start_end_indices) % 2 == 0):
            start_indices, end_indices = start_end_indices[0:len(start_end_indices):2], \
                start_end_indices[1:len(start_end_indices):2] - 1
            false_indices = np.concatenate([np.arange(s, e + 1) for (s, e) in zip(start_indices, end_indices)])
            is_ok = np.concatenate(
                [abs(ContourBasedTracking.calc_points_diff(origin_head_points[s:e + 1, :])) < deviation_threshold
                 for (s, e) in zip(start_indices, end_indices)])
            if np.array(is_ok).all():  # all differences within start-end are indeed the same
                is_diff_below_threshold[:] = True
                is_diff_below_threshold[false_indices] = False
                logging.debug(name_for_debug +
                              " start-end indices {0} => false indices {1}".format(start_end_indices + 1,
                                                                                   false_indices + 1))
            else:
                logging.error(name_for_debug +
                              " start-end indices have jumps within segment {0}".format(false_indices + 1))
                is_diff_below_threshold = \
                    np.diff(np.concatenate([is_diff_below_threshold[0:1],
                                            is_diff_below_threshold.astype(int)])) != -1
        elif len(start_end_indices) > 0:
            logging.debug(name_for_debug + " start-end indices have odd number. {0}".format(start_end_indices + 1))
            is_diff_below_threshold = \
                np.diff(np.concatenate([is_diff_below_threshold[0:1],
                                        is_diff_below_threshold.astype(int)])) != -1
        return is_diff_below_threshold

    def __post_processing_find_head_errors(self, analysis_data_outputs, use_start_end_segmentation=False):
        """Return list of 2 boolean values, one check origin jump and one angle jump (head flip)

        :param analysis_data_outputs: tracker output struct of the whole movie
        :param use_start_end_segmentation: default false, until this logic find better the segments
        :return: is_head_angle_diff_norm_below_threshold, is_head_origin_diff_norm_below_threshold - lists of booleans
        """
        # origin points and destination points used for both point jump and angle jump check.
        # Make sure false status has nan value (prediction has valid value since we check the prediction is ok)
        check_origin = analysis_data_outputs.origin_head_points_list.copy()
        check_origin[analysis_data_outputs.fish_status_list == False, :] = np.nan
        check_dest = analysis_data_outputs.destination_head_points_list.copy()
        check_dest[analysis_data_outputs.fish_status_list == False, :] = np.nan

        # Fill nan values with interpolation
        origin_head_points = ContourBasedTracking.fill_nan_2d_interpolate(check_origin.copy())
        dest_head_points = ContourBasedTracking.fill_nan_2d_interpolate(check_dest.copy())

        head_direction_angles = [get_angle_to_horizontal(orig, dest)
                                 for (orig, dest) in zip(origin_head_points, dest_head_points)]

        # origin can jump up to some threshold
        differences_origin = abs(ContourBasedTracking.calc_points_diff(origin_head_points))
        is_head_origin_diff_norm_below_threshold = differences_origin < self.head_origin_deviation_threshold

        # angle can jump to 360 if cross the edge, but head flip should be around 180 degrees
        head_direction_angles = np.concatenate([head_direction_angles[0:1], head_direction_angles])
        is_head_angle_diff_norm_below_threshold = np.bitwise_or(abs(360 - np.diff(head_direction_angles)) < 90,
                                                                abs(np.diff(head_direction_angles)) < 90)

        # fine tune boolean marks by considering segments etc
        is_head_origin_diff_norm_below_threshold = \
            self.__post_processing_is_diff_below_threshold_fix(is_head_origin_diff_norm_below_threshold,
                                                               origin_head_points, self.head_origin_deviation_threshold,
                                                               name_for_debug="Origin",
                                                               use_start_end_segmentation=use_start_end_segmentation)
        is_head_angle_diff_norm_below_threshold = \
            self.__post_processing_is_diff_below_threshold_fix(is_head_angle_diff_norm_below_threshold,
                                                               head_direction_angles, 90, name_for_debug="Angle",
                                                               use_start_end_segmentation=use_start_end_segmentation)

        return is_head_angle_diff_norm_below_threshold, is_head_origin_diff_norm_below_threshold

    @classmethod
    def create_graph_from_points(cls, midline_points, height, width, ):
        """Creates an undirected weighted graph from a list of points on the fish's midline.

        Logic: For each point (representing a pixel) in midline_points, checks the adjacent
        8 points near it (using height and width to make sure we're not out of bounds). If an adjacent point exists on the midline points,
        creates an edge between both points.
        If two points are diagonal to one another, their relevant edge weight will be 2, if the adjacency is in a straight line, their weight will be 1.

        :param midline_points: List of points on the fish's midline
        :param height: Image's height in pixels
        :param width: Image's width in pixels
        :return: graph: A networkx undirected graph object, connecting all adjacent points on the midline (Those are the graph's vertice) by edges, with weight for diagonal or straight adjacencies.
        """
        graph = networkx.Graph()
        midline_points = [
            tuple(point) for point in midline_points.tolist()
        ]
        for curr_point in midline_points:
            curr_point = tuple(
                curr_point,
            )
            adjacent_points = [
                point for point in machine_vision.get_adjacent_points(
                    point=curr_point,
                    radius=1,
                    width=width,
                    height=height,
                ) if point in midline_points
            ]
            for point in adjacent_points:
                delta_x = abs(point[0] - curr_point[0])
                delta_y = abs(point[1] - curr_point[1])
                is_diagonal = delta_x + delta_y == 2
                if is_diagonal:
                    weight = 1
                else:
                    weight = 2

                graph.add_edge(
                    point,
                    curr_point,
                    weight=weight,
                )

        return graph

    @classmethod
    def get_midline_as_list_of_connected_points(cls, cleaned_fish, output, friend_fish=False):
        """
        Gets an image array of the fish and returns a list of points describing the fish midline (each point connected to the indices adjacent to it)

        :param cleaned_fish: image array only containing the fish.
        :param output: fish analysis struct used here in order to produce a clean midline image array.
        :return: A list of points (pixels) where each index on the list is connected to the previous and next indices.
        """
        fish_midline_clean = cls.get_clean_fish_midline(cleaned_fish, output, remove_head=False)
        if fish_midline_clean is None:
            return None

        # Get nonzero points (y, x)
        y = fish_midline_clean.nonzero()[0]
        x = fish_midline_clean.nonzero()[1]

        # Ensure that the points are sorted based on y-coordinates
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]

        # Perform 2D polynomial fit (fitting x as a function of y)
        poly_coeff = np.polyfit(y, x, 6)  # Fit polynomial to the x(y)
        poly_func = np.poly1d(poly_coeff)  # Create a polynomial function from the coefficients

        # Use the polynomial function to generate smoothed x values
        smoothed_x = poly_func(y)

        # Stack smoothed x and original y values to get the smoothed midline path
        smoothed_midline_path = np.stack([smoothed_x, y], axis=1)

        # Optional: Align the smoothed midline path to clean up any discrepancies
        midline_path = cls.align_midline_path(cleaned_fish, smoothed_midline_path, output, friend_fish)

        return midline_path

    @classmethod
    def align_midline_path(cls, cleaned_fish, midline_path, output, friend_fish=False):
        """
        Aligns the midline path so that the start point is the tip of the tail and the end point is near the head.

        :param cleaned_fish: image array only containing the fish.
        :param midline_path: list of connected points on the fish midline.
        :param output: fish analysis struct used here in order to produce a clean midline image array.
        :return: The midline path, aligned properly so that the starting point is the tip of the tail.
        """
        start_point = midline_path[0]
        end_point = midline_path[-1]
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(10, 10), )
        bnw_cleaned_for_tail = machine_vision.frame_to_bnw(frame=cleaned_fish, thresh=55)
        opened_blob = cv2.morphologyEx(bnw_cleaned_for_tail, cv2.MORPH_OPEN, kernel)

        if not friend_fish:
            return midline_path[::-1]

        end_point_in_opened_blob_status = machine_vision.check_point_in_blob_status(opened_blob, end_point)
        if end_point_in_opened_blob_status == 'inside':
            return midline_path

        start_point_in_opened_blob_status = machine_vision.check_point_in_blob_status(opened_blob, start_point)
        if start_point_in_opened_blob_status == 'inside':
            return midline_path[::-1]

        # Computes the euclidean distance from the origin point to determine which point is closer to the head.
        origin_point = output.fish_head_origin_point
        start_point_dist = ((origin_point[0] - start_point[0]) ** 2 + (origin_point[1] - start_point[1]) ** 2) ** 0.5
        end_point_dist = ((origin_point[0] - end_point[0]) ** 2 + (origin_point[1] - end_point[1]) ** 2) ** 0.5

        if start_point_dist > end_point_dist:
            return midline_path

        return midline_path[::-1]

    @classmethod
    def get_swimbladder_from_midline_path(cls, cleaned_fish, midline_path, output, fish_size=125):
        """
        """
        gray_fish = cv2.cvtColor(cleaned_fish, cv2.COLOR_BGR2GRAY)

        points_brightness = []
        for point in midline_path:
            brightness = gray_fish[point[1], point[0]]
            points_brightness.append(brightness)

        points_brightness = np.array(points_brightness) * -1
        tail_len_two_thirds = int(len(points_brightness) * 2 / 3)
        min_brightness = min(-175, np.median(points_brightness[tail_len_two_thirds:]) + 0.5 * np.median(
            np.abs(points_brightness[tail_len_two_thirds:] - np.median(points_brightness[tail_len_two_thirds:]))))
        peak_indices = find_peaks(points_brightness, height=min_brightness, distance=5)[0]  # Magic number
        peak_indices = sorted(peak_indices)
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE,
            ksize=(9, 9),
        )
        bnw_frame = machine_vision.frame_to_bnw(frame=cleaned_fish, thresh=55)
        filled_bnw_frame = machine_vision.fill_contour(bnw_frame)
        opened_fish = cv2.morphologyEx(filled_bnw_frame, cv2.MORPH_OPEN, kernel)

        height = gray_fish.shape[0]
        width = gray_fish.shape[1]
        last_third_of_tail = len(midline_path) * (2 / 3)
        index_and_brightness_list = []
        radius_of_adjacent_points = 2
        for peak_index in peak_indices:
            if peak_index < last_third_of_tail:
                continue
            peak_point = (midline_path[peak_index][0], midline_path[peak_index][1])
            peak_brightness = gray_fish[peak_point[1], peak_point[0]]
            adjacent_points = machine_vision.get_adjacent_points(point=peak_point, radius=radius_of_adjacent_points,
                                                                 width=width, height=height)
            is_all_adjacent_points_inside_fish = all([
                machine_vision.check_point_in_blob_status(opened_fish, point) == 'inside' for point in adjacent_points
            ])
            if is_all_adjacent_points_inside_fish:
                distance_from_origin = ((peak_point[0] - output.fish_head_origin_point[0]) ** 2 + (
                        peak_point[1] - output.fish_head_origin_point[1]) ** 2) ** 0.5
                index_and_brightness = {
                    'index': peak_index,
                    'brightness': peak_brightness,
                    'distance_from_origin': distance_from_origin,
                }
                index_and_brightness_list.append(index_and_brightness)

        max_distance_from_origin = (fish_size - cls.small_fish_size) / 4 + cls.small_fish_max_distance_from_origin
        min_distance_from_origin = (fish_size - cls.small_fish_size) / 4 + cls.small_fish_min_distance_from_origin
        index_and_brightness_list = [
            index_to_mean for index_to_mean in index_and_brightness_list if
            min_distance_from_origin < index_to_mean['distance_from_origin'] < max_distance_from_origin
        ]
        if index_and_brightness_list == []:
            return None

        # new code:
        if len(index_and_brightness_list) > 2:
            two_lowest_brightness = sorted([x['brightness'] for x in index_and_brightness_list])[1]
            two_lowest_brightness_list = [x for x in index_and_brightness_list if
                                          x['brightness'] <= two_lowest_brightness]
            swimbladder_point = min(two_lowest_brightness_list, key=lambda x: x['distance_from_origin'])
        else:
            swimbladder_point = min(index_and_brightness_list, key=lambda x: x['distance_from_origin'])

        # old code:
        # swimbladder_point = min(index_and_brightness_list, key=lambda x: x['brightness'])
        return swimbladder_point['index']

    @classmethod
    def tail_tip_from_fish_contour(cls, cleaned_fish, frame_number, output):
        """
        Finds the tail tip given the fish contour using BFS

        :param cleaned_fish: Image array only containing the fish.
        :param frame_number: Frame number used for logging purposes.
        :param output: fish analysis struct used here in order to produce a clean midline image array.
        :return: The point at the tip of the fish's tail.

        NOTE: This function is deprecated, it takes a very long time to find the tail tip and we have a better way of doing that.
        """
        tail_tip = None

        bnw_cleaned_for_tail = machine_vision.frame_to_bnw(frame=cleaned_fish, thresh=10)
        point_on_head = cls.point_to_int(output.fish_head_origin_point)

        skeleton = cv2.bitwise_or(machine_vision.skeleton(frame=cleaned_fish),
                                  cv2.ximgproc.thinning(src=bnw_cleaned_for_tail))
        skeleton_contours, _ = cv2.findContours(image=skeleton, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

        result = machine_vision.check_point_in_blob_status(bnw_blob=bnw_cleaned_for_tail,
                                                           point=point_on_head)  # todo needed?
        if result == 'outside':  # TODO: change to const?
            point_on_head = cls.get_point_close_to_head_for_fish_blob(frame=cleaned_fish,
                                                                      blur_intensity=11)  # TODO: Magic number
            result = machine_vision.check_point_in_blob_status(bnw_blob=bnw_cleaned_for_tail, point=point_on_head)
        if result == 'outside':
            logging.error('Failure finding point on head, point found outside of shape. Frame: {frame_number}'.format(
                frame_number=frame_number))
        else:
            tail_tip = cls.get_tail_tip(cleaned_fish_bnw_frame=bnw_cleaned_for_tail, point_on_head=point_on_head,
                                        skeleton_contours=skeleton_contours)
        return tail_tip

    @classmethod
    def get_clean_fish_midline(cls, cleaned_fish, output, remove_head=True):
        """
        Turns the isolated image of the fish into a black and white frame containing only the thinned out midline of the fish.

        :param cleaned_fish: image array only containing the fish.
        :param output: fish analysis struct used here in order to produce a clean midline image array.
        :return: A black and white frame containing only the thinned out midline of the fish.

        Assumptions:
        The image is large enough for the kernel  
        """
        bnw_cleaned_for_tail = machine_vision.frame_to_bnw(frame=cleaned_fish, thresh=cls.clean_bnw_for_tail_threshold)
        # bnw_cleaned_for_tail = machine_vision.fill_contour(bnw_cleaned_for_tail) ##YR added to fix detection of skeleton
        bnw_cleaned_for_tail = machine_vision.fill_contour_without_middle(
            bnw_cleaned_for_tail)  ##YR added to fix detection of skeleton

        # We use the logical "or" of two different thinning algorithms,
        # This proved to yield better results when trying to find a continuous midline path.
        skeleton = cv2.bitwise_or(machine_vision.skeleton(frame=cleaned_fish),
                                  cv2.ximgproc.thinning(src=bnw_cleaned_for_tail))

        cv2.drawContours(skeleton, output.eyes_contour, contourIdx=-1, color=Colors.WHITE, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(9, 9))

        if remove_head:
            isolated_head = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)
            isolated_head = cv2.morphologyEx(isolated_head, cv2.MORPH_OPEN, kernel)
            isolated_head_ordered_by_size_contours = machine_vision.find_contours_with_area_sorted(isolated_head)
            if len(isolated_head_ordered_by_size_contours) == 0:
                return None

            isolated_head_ordered_by_size_contours = isolated_head_ordered_by_size_contours[0]['contour']

            isolated_head = np.zeros(shape=isolated_head.shape, dtype=np.uint8)
            # I don't know if it's a bug, but if there is one contour, draw contours won't fill it unless we enclose contours in a list.
            cv2.drawContours(isolated_head, [isolated_head_ordered_by_size_contours], contourIdx=-1, color=Colors.WHITE,
                             thickness=cv2.FILLED)
            skeleton_midline = cv2.subtract(skeleton, isolated_head, )
        else:
            skeleton_midline = skeleton

        skeleton_contours, _ = cv2.findContours(image=skeleton_midline, mode=cv2.RETR_LIST,
                                                method=cv2.CHAIN_APPROX_NONE)
        skeleton_contours = [contour for contour in skeleton_contours
                             if cv2.contourArea(contour) > cls.minimum_midline_contour_area]

        skeleton_midline_clean = np.zeros(shape=skeleton.shape, dtype=np.uint8)

        # Draw the midline on a new image without the noise surrounding it
        cv2.drawContours(skeleton_midline_clean, skeleton_contours, contourIdx=-1, color=Colors.WHITE,
                         thickness=cv2.FILLED)

        # The midline is prone to have holes in it, this part fills in the holes.
        kern = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3), )
        closed = cv2.morphologyEx(skeleton_midline_clean, cv2.MORPH_CLOSE, kern)

        # Now that all of the holes are filled out, we can thin it one last time and get a continuous narrow line that is one pixel wide.
        closed_thinned = cv2.ximgproc.thinning(src=closed)

        return closed_thinned

    @staticmethod
    def fix_ellipse_angle(ellipse_angle, fish_dir_angle):  # todo move outside?
        """

        :param ellipse_angle: 0-180 relative to y axis (270 and 90 in x axis is 0 here)
        :param fish_dir_angle: 0-360 relative to x axis (counter-clockwise)
        :return: ellipse_angle is same direction relative to fish_direction_angle
        """
        result_ellipse_angle = 90 - ellipse_angle  # (-90) - 90 range - relative to x axis
        orig_result_ellipse_angle = result_ellipse_angle

        diff = result_ellipse_angle - fish_dir_angle
        if diff < 0:
            result_ellipse_angle += 180 * (round(abs(diff) / 180))
        elif diff > 180:
            result_ellipse_angle -= 180 * (round(abs(diff) / 180))

        if not (0 <= abs(result_ellipse_angle - fish_dir_angle) <= 90):
            logging.error('Error in ellipse angle. Bad result: {0}, fish-dir {1}, original ellipse angle {2]'.format(
                result_ellipse_angle, fish_dir_angle, orig_result_ellipse_angle))
        return result_ellipse_angle

    @staticmethod
    def get_fish_segment_and_masks(an_frame, cleaned, fish_contour):
        # mask contains fish only
        mask = np.full((an_frame.shape[0], an_frame.shape[1]), 0, dtype=np.uint8)
        cv2.drawContours(mask, [fish_contour], contourIdx=-1, color=Colors.WHITE, thickness=cv2.FILLED)
        segment = cv2.findNonZero(mask)  # shape: (# points, 1, 2)
        # Create cleaned figures only - expand mask with dilate + blurring
        d_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)  # expand a little the fish
        ret, d_mask = cv2.threshold(cv2.medianBlur(d_mask, 9), 30, 255, cv2.THRESH_BINARY)
        cleaned_fish = cv2.bitwise_and(cleaned, cleaned, mask=d_mask)  # search objects within fish only
        cleaned_non_fish = cv2.bitwise_and(cleaned, cleaned, mask=np.bitwise_not(d_mask))
        return cleaned_fish, cleaned_non_fish, segment, mask, d_mask

    @staticmethod
    def get_contours(gray, threshold1=30, threshold2=200, is_blur=False, is_close=True, ctype=cv2.RETR_TREE,
                     close_kernel=(5, 5), min_area_size=None):  # todo scale?
        gray = cv2.Canny(gray, threshold1, threshold2)
        if is_blur:  # smear edges to have full fish contour
            gray = cv2.blur(gray, (3, 3))
        elif is_close:  # use close instead
            kkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kkernel)

        contours, hierarchy = cv2.findContours(gray, ctype, cv2.CHAIN_APPROX_NONE)
        if min_area_size is None:
            return [c for c in contours if c.shape[0] >= 5], hierarchy  # ellipse fit requires min 5 points
        return [c for c in contours if c.shape[0] >= 5 and cv2.contourArea(c) >= min_area_size], hierarchy

    @staticmethod
    def two_closest_shape_contours(contours_data, max_distance=25, scale=1):  # todo move outside?
        min_area_diff = math.inf
        eyes_contours = None
        for c1_d in contours_data:
            c1 = c1_d['contour']
            (xc_1, yc_1), _, _ = c1_d['ellipse']
            for c2_d in contours_data:
                c2 = c2_d['contour']
                curr_area_diff = abs(cv2.contourArea(c1) - cv2.contourArea(c2))
                (xc_2, yc_2), _, _ = c2_d['ellipse']
                if curr_area_diff < min_area_diff and np.all(c2 != c1) and \
                        0 < distance.euclidean([xc_1, yc_1], [xc_2, yc_2]) <= max_distance * scale:
                    eyes_contours = [c1_d, c2_d]
                    min_area_diff = curr_area_diff
        return eyes_contours

    @classmethod
    def get_eyes_no_eyes_within_fish_segment(cls, an_frame, frame, frame_number, scale=1,
                                             min_eyes_area=50, max_color=80, max_distance=25):  # todo refactor
        """Logic: find dark (<=max_color) contours within [min_eyes_size-min_eyes_size] area size (eyes are ~60).
        If more than 2 are found (for example, if fins create another dark contour), search for 2 closest to each other.

        min_eyes_area=50, max_eyes_area=200 - needed?

        :param is_new_movie:
        :param frame_number:
        :param scale: scale distance & area (based on zoom)
        :param max_distance:
        :param min_eyes_area:
        :param an_frame: frame to annotate eyes
        :param frame: input frame, masked by fish segment
        :param max_color: define how dark the eyes should be. ~70
        :return:
        """

        def search_possible_kernels(curr_frame=frame):  # start with normal/most common. If not working try higher
            inner_cnts = search(curr_frame=curr_frame)
            found = False
            if 2 <= len(inner_cnts):  # no eyes were found
                eyes_sus = [c for c in inner_cnts if c[1] <= max_color]
                if len(eyes_sus) >= 2:
                    found = True
            if not found:
                inner_cnts2 = search(curr_frame=curr_frame, close_kernel=(3, 3))
                if 2 <= len(inner_cnts2):  # no eyes were found
                    eyes_sus = [c for c in inner_cnts2 if c[1] <= max_color]
                    if len(eyes_sus) >= 2:
                        found = True
                        inner_cnts = inner_cnts2
            if not found:
                inner_cnts2 = search(curr_frame=curr_frame, close_kernel=(7, 7))
                if 2 <= len(inner_cnts2):  # no eyes were found
                    eyes_sus = [c for c in inner_cnts2 if c[1] <= max_color]
                    if len(eyes_sus) >= 2:
                        found = True
                        inner_cnts = inner_cnts2
            return inner_cnts, found

        def search(curr_frame=frame, close_kernel=(5, 5)):
            contours, hier = cls.get_contours(curr_frame, close_kernel=close_kernel, ctype=cv2.RETR_CCOMP)
            inner_cnts = []
            for i, cont in enumerate(contours):
                if len(cont) >= 5:
                    mask_c = np.full((curr_frame.shape[0], curr_frame.shape[1]), 0, dtype=np.uint8)
                    cv2.drawContours(mask_c, [cont], -1, color=Colors.WHITE, thickness=cv2.FILLED)
                    mean_intensity = cv2.mean(curr_frame, mask=mask_c)[0]
                    inner_cnts.append([i, mean_intensity, cont, mask_c, cv2.fitEllipse(cont), cv2.contourArea(cont)])
            inner_cnts = sorted(inner_cnts, key=lambda r: -r[1])  # sort by mean intensity small to big
            inner_cnts = [c for c in inner_cnts if c[-1] >= min_eyes_area * scale and c[1] <= max_color]
            inds = set()
            for cont_1 in inner_cnts:
                (xc_1, yc_1), _, _ = cont_1[4]
                for cont_2 in inner_cnts:
                    (xc_2, yc_2), _, _ = cont_2[4]
                    if 0 < distance.euclidean([xc_1, yc_1], [xc_2, yc_2]) <= max_distance * scale:
                        inds.add(cont_1[0])
                        inds.add(cont_2[0])
            return [c for c in inner_cnts if c[0] in inds]

        eyes_data = []
        no_eyes_data = []

        inner_cnts, found = search_possible_kernels()
        if not found:
            frame_2 = frame.copy()
            # This is an corner case where eye is separated from body- try to use large contour to fix frame
            fish = cls.get_fish_contour(frame_2, scale=scale)
            if fish is None:
                return [], []  # this shouldn't happen
            cv2.drawContours(frame_2, [fish], -1, color=Colors.WHITE)
            inner_cnts2, found = search_possible_kernels(frame_2)
            if found:
                inner_cnts = inner_cnts2

        for i, mean_intensity, c, mask_c, ellipse, area in inner_cnts:
            # todo didn't find more efficient way to calculate mean within contour
            if mean_intensity <= max_color:  # Should be very dark
                eyes_data.append({'contour': c, 'mask': mask_c, 'area': cv2.contourArea(c),
                                  'mean_intensity': mean_intensity, 'ellipse': ellipse})
            else:
                no_eyes_data.append({'contour': c, 'mask': mask_c, 'area': cv2.contourArea(c),
                                     'mean_intensity': mean_intensity, 'ellipse': ellipse})
        if 0 < len(eyes_data) <= 2:  # one/two eyes were found
            cv2.drawContours(an_frame, [c['contour'] for c in eyes_data], -1, color=Colors.CYAN)
            return eyes_data, no_eyes_data
        elif len(eyes_data) > 2:
            eyes_data2 = cls.two_closest_shape_contours(eyes_data, max_distance=max_distance, scale=scale)
            if eyes_data2 is None:
                return [], []
            eyes_ellipses = [c['ellipse'] for c in eyes_data2]
            no_eyes_data.extend([c_d for c_d in eyes_data if c_d['ellipse'] not in eyes_ellipses])
            cv2.drawContours(an_frame, [c['contour'] for c in eyes_data2], -1, color=Colors.CYAN)
            return eyes_data2, no_eyes_data
        elif len(eyes_data) == 0:
            return [], []  # this shouldn't happen

    @classmethod
    def clean_plate_noise(cls, input_frame, frame_number, scale=1,
                          min_fish_size=2000, max_fish_size=10000, max_size_ratio=3, is_blur=True):
        """Remove contours which creates a plate.
        Use fish suspects & paramecia (based on size) estimates to remove plate (which can be "broken" to several
        pieces). is_blur=True adds few more pixels to make sureall is removed.
        :return: 'cleaned' frame
        """
        contours, _ = cls.get_contours(input_frame, ctype=cv2.RETR_EXTERNAL, close_kernel=(9, 9))
        hulls = [(cv2.convexHull(cnt), cnt) for cnt in contours]

        # Fish is limited both in its area and the ratio between this and the hull (small pieces of the plate have
        # larger ratio between the 2 even if the area is similar)
        fish_suspects = [(h, c) for (h, c) in hulls if
                         min_fish_size * scale <= cv2.contourArea(h) <= max_fish_size and
                         1 <= cv2.contourArea(h) / cv2.contourArea(c) <= max_size_ratio * scale]

        if len(fish_suspects) == 1:  # found exactly the fish - remove plate!
            fish_h, fish_c = fish_suspects[0]
            non_fish_contours = [c for (h, c) in hulls if 500 <= cv2.contourArea(h) and  # don't remove paramecia
                                 # Validate not fish
                                 cv2.contourArea(fish_h) != cv2.contourArea(h) and
                                 cv2.contourArea(fish_c) != cv2.contourArea(c)]
            result = input_frame.copy()
            mask = np.zeros(result.shape[:2], np.uint8)
            cv2.drawContours(mask, non_fish_contours, -1, Colors.WHITE, thickness=cv2.FILLED)
            if is_blur:  # smear edges to have full fish contour
                _, mask = cv2.threshold(cv2.blur(mask, (11, 11)), 10, 255, cv2.THRESH_BINARY)
            return cv2.bitwise_and(input_frame, input_frame, mask=cv2.bitwise_not(mask))

        # This should happen if fish is attached to plate (try to estimate plate as circle)
        result = clean_plate(input_frame)
        if result is not None:  # succeed -> check fish is found
            fish_contour = cls.get_fish_contour(result.astype(input_frame.dtype), scale=scale)  # get external only
            if fish_contour is not None:
                return result
            else:
                logging.error("Frame " + str(frame_number) + " find plate but didn't find fish.")
        else:
            logging.error("Frame " + str(frame_number) + " didn't find plate.")

        return None

    @classmethod
    def get_fish_contour(cls, gray, close_kernel=(5, 5), min_fish_size=50, max_fish_size=50000, scale=1,
                         threshold1=30, threshold2=200):
        contours, _ = cls.get_contours(gray, ctype=cv2.RETR_EXTERNAL, close_kernel=close_kernel,
                                       threshold1=threshold1, threshold2=threshold2)  # get external only
        # remove paramecia
        contours = [c for c in contours if min_fish_size * scale <= cv2.contourArea(c) <= max_fish_size * scale]
        if len(contours) > 0:
            return max(contours, key=cv2.contourArea)  # if cleaned, fish is largest
        return None

    @staticmethod
    def get_fish_points(fish_contour):
        fish_ellipse = cv2.fitEllipse(fish_contour)
        (xc, yc), (d1, d2), angle = fish_ellipse
        rmajor = max(d1, d2) / 2
        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90
        xtop = xc + math.cos(math.radians(angle)) * rmajor
        ytop = yc + math.sin(math.radians(angle)) * rmajor
        xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
        ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
        return (xtop, ytop), (xbot, ybot)

    def calc_fish_direction_from_eyes(self, cleaned_fish_frame, eyes_data, frame_number, output, scale=1,
                                      head_size=30):
        """

        :param cleaned_fish_frame: input for current algorithm - original pixels of fish only (after mask)
        :param eyes_data: as calculated by searching for their contours and ellipses.
        :param frame_number: for debug use (you can break-point on specific frame)
        :param output: set to False if can't calculate direction properly
        :return: eyes_data, p_from, p_to - where p_to and p_from are dest and origin of fish direction
        """
        # Head contour around eyes - via bounding rectangle around both eyes
        min_x, min_y, max_x, max_y = [np.inf, np.inf, 0, 0]
        cleaned_fish_head = None
        fish_head_contour = None
        for c in output.eyes_contour:
            (x, y, w, h) = cv2.boundingRect(c)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)

        # If didn't find this rectangle - proceed with fish contour
        if not (max_x - min_x > 0 and max_y - min_y > 0):
            p_to, p_from, eyes_data = \
                self.calc_0_approx_fish_direction_based_on_eyes(output.fish_contour, output.eyes_contour, eyes_data)
        else:
            # mask head - expand rectangle with add_head pixels
            add_head = head_size * max(1, int(scale / 2))
            top_left = (min_x - add_head, min_y - add_head)
            bottom_right = (max_x + add_head, max_y + add_head)
            # cv2.rectangle(an_frame, top_left, bottom_right, Colors.PINK)  # visualize rect

            # mask
            mask_head = np.full((cleaned_fish_frame.shape[0], cleaned_fish_frame.shape[1]), 0, dtype=np.uint8)
            cv2.rectangle(mask_head, top_left, bottom_right, Colors.WHITE, cv2.FILLED)
            cleaned_fish_head = cv2.bitwise_and(cleaned_fish_frame, cleaned_fish_frame,
                                                mask=mask_head)  # search objects within fish only
            # show("head", cleaned_fish_head)
            # Search fish again: At this point the eyes were found= dont expand fish too much (smaller kernel)
            fish_head_contour = self.get_fish_contour(cleaned_fish_head, close_kernel=(5, 5), scale=scale)

            # Approx.0 on head only
            p_to, p_from, eyes_data = \
                self.calc_0_approx_fish_direction_based_on_eyes(fish_head_contour, output.eyes_contour, eyes_data)

            # cv2.circle(an_frame, self.point_to_int(p_to), 5, Colors.BLUE, -1)

            # Override p_to with line through fish head
            # vx, vy, x0, y0 = self.get_head_line(fish_head_contour)  # this is an alternative algo.
            vx, vy, x0, y0 = self.get_line_perpendicular_to_eyes(eyes_data, p_from)

            # Fix vx vy based on 'stronger' direction (fix near 0 errors): the fix is since can be 180deg error
            length = add_head * 2
            if np.abs(vy) >= np.abs(vx) and vy != 0:
                t1 = length * np.sign((p_to[1] - y0) / vy)
            elif np.abs(vx) >= np.abs(vy) and vx != 0:
                t1 = length * np.sign((p_to[0] - x0) / vx)
            else:
                output.is_ok = False
                logging.error("Error: frame #{0} has 0 vx and vy!".format(frame_number))
                t1 = 0

            p_to = (float(x0 + t1 * vx), float(y0 + t1 * vy))
        return eyes_data, p_from, p_to, cleaned_fish_head, mask_head, fish_head_contour

    @staticmethod
    def calc_0_approx_fish_direction_based_on_eyes(head_contour, eyes_contours, eyes_data):
        """ Used to calc direction, by using shape fit around eyes as p_to, and mid-eyes point as p_from.
        This is 0th approx, since using fish direction it is fixed.

        :param head_contour:
        :param eyes_contours:
        :param eyes_data:
        :return:
        """
        p_to = ContourBasedTracking.closest_point_of_bound_shape_to_eyes(eyes_contours, head_contour)

        middle = [0, 0]
        for i in range(len(eyes_data)):
            (xc, yc), (d1, d2), angle = eyes_data[i]['ellipse']
            major = max(d1, d2) / 2
            minor = min(d1, d2) / 2
            eyes_data[i]['center'] = (xc, yc)
            eyes_data[i]['angle'] = angle
            eyes_data[i]['major'] = major
            eyes_data[i]['minor'] = minor
            middle[0] += xc
            middle[1] += yc
        p_from = (middle[0] / 2, middle[1] / 2)
        return p_to, p_from, eyes_data

    @staticmethod
    def closest_point_of_bound_shape_to_eyes(eyes_contours, head_contour):
        pts = ContourBasedTracking.get_fish_points(head_contour)
        distances = []
        for p in pts:
            distance_to_contours_top = [distance.euclidean((c[0][0][0], c[0][0][1]), (p[0], p[1])) for c in
                                        eyes_contours]
            distances.append({'dist': sum(distance_to_contours_top), 'point': (p[0], p[1])})
        distances = sorted(distances, key=lambda d: d['dist'])
        p_to = (distances[0]['point'][0], distances[0]['point'][1])  # 0 is closest
        return p_to

    @staticmethod
    def get_head_line(head_contour):
        [vx, vy, x, y] = cv2.fitLine(head_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        return vx[0], vy[0], x[0], y[0]  # todo cut with contour and not image

    @staticmethod
    def get_line_perpendicular_to_eyes(eyes_data, p_from):
        centers = [eye['center'] for eye in eyes_data]
        vx = centers[0][0] - centers[1][0]
        vy = centers[0][1] - centers[1][1]
        # Normal to line
        mag = math.sqrt(vx ** 2 + vy ** 2)
        temp = vx / mag
        vx = -(vy / mag)
        vy = temp
        return vx, vy, p_from[0], p_from[1]

    @staticmethod
    def get_point_close_to_head_for_fish_blob(frame, blur_intensity=11, open_intensity=1):
        bnw_frame = machine_vision.blur_image_to_bnw(frame=frame, blur_intensity=blur_intensity)

        default_center_of_mass = machine_vision.get_center_of_mass_from_single_blob(bnw_frame=bnw_frame)
        prev_center_of_mass = default_center_of_mass

        for open_intensity in range(1, 50):  # TODO: Magic number
            bnw_opened_frame = machine_vision.open_bnw_blob(frame=bnw_frame, open_intensity=open_intensity)

            center_of_mass = machine_vision.get_center_of_mass_from_single_blob(bnw_frame=bnw_opened_frame)
            if center_of_mass is None:
                return prev_center_of_mass

            prev_center_of_mass = center_of_mass

        return center_of_mass

    @staticmethod
    def get_tail_tip(cleaned_fish_bnw_frame, point_on_head, skeleton_contours):
        frame_shape = (cleaned_fish_bnw_frame.shape[0], cleaned_fish_bnw_frame.shape[1], 3)
        thinned_out_fish_blob = np.zeros(shape=frame_shape, dtype=np.uint8)

        head_circle_radius = 15  # TODO: Magic number
        cv2.circle(img=thinned_out_fish_blob, center=point_on_head, radius=head_circle_radius, color=Colors.WHITE,
                   thickness=cv2.FILLED)
        cv2.drawContours(
            image=thinned_out_fish_blob,
            contours=skeleton_contours,
            contourIdx=-1,  # Add const
            color=Colors.WHITE,
            thickness=3,  # TODO: Magic number
        )
        thinned_out_fish_blob = machine_vision.frame_to_bnw(
            frame=thinned_out_fish_blob,
            thresh=10,  # TODO: Magic number
        )

        point_on_head_chebyshev_dict = machine_vision.create_chebyshev_dict_from_blob(
            bnw_blob_frame=thinned_out_fish_blob, point=point_on_head)

        flattened_skeleton_contour_points = machine_vision.flatten_contour_list(contour_list=skeleton_contours)
        skeleton_tuple_points = machine_vision.np_points_to_tuple_points(np_points=flattened_skeleton_contour_points)

        skel_points_and_distances_on_larvafish = {}
        for point in skeleton_tuple_points:
            point_distance = point_on_head_chebyshev_dict.get(point, -1)
            if point_distance > 0:
                skel_points_and_distances_on_larvafish[point] = point_distance

        point_at_tail_tip = max(skel_points_and_distances_on_larvafish, key=skel_points_and_distances_on_larvafish.get)
        return np.array(point_at_tail_tip).astype(np.float)

    def all_events_post_process(self, fish):
        start_time = time.time()

        svd_struct = svd_analyzer.fish_svd_data(fish)
        fish_events = svd_struct.events_with_svd()
        fish_power = np.sqrt(np.sum(svd_struct.new_axes[:2, :] ** 2, axis=0))
        svd_thresh = np.percentile(fish_power, 60)
        svd_merge_thresh = np.percentile(fish_power, 70)
        for event_id in range(len(fish.events)):
            (fish.events[event_id].tail.bout_start_frames, fish.events[event_id].tail.bout_end_frames,
             fish.events[event_id].tail.is_bout_frame_list) = \
                (self.bout_detection(fish.events[event_id].tail.interpolated_tail_path, fish_events[event_id],
                                     svd_thresh, svd_merge_thresh))
        svd_struct = svd_analyzer.fish_svd_data(
            fish)  # run the events again, this time with the right bouts detection mechanism

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(f"The main took {elapsed_time / 60:.4f} minutes to run.")
        return fish, svd_struct

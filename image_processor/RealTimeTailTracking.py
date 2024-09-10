import copy
import logging
import numpy as np
import cv2

from preprocess_config import *
from classic_cv_trackers.fish_tracking import ContourBasedTracking, FrameAnalysisData
from utils_closed_loop.utils import get_angle_to_horizontal
from utils_closed_loop import numerical_analysis
from classic_cv_trackers import Colors


class RealTimeTailTracking(ContourBasedTracking):
    """For closed-set loop. Inherit ContourBasedTracking for common functionality in fish detection.
    This tracker receives a different frame resolution,
    with fish in predefined location and orientation,
    and should output very fast the minimal data needed to detect tail movement direction.
    """

    def __init__(self, stateless=False, fixed_fish=False, friend_fish=False, head_origin=None, head_destination=None,
                 **kwargs):  # f3: x -= 165 y-= 65
        """ x_lim and y_lim are empty list if no mask needed on input.
        Otherwise - these are borders for mask.
        :param x_lim: default match social data
        :param y_lim:
        :param kwargs: ContourBasedTracking args
        """
        # IL: Imri's cropping for fixed fish.
        if fixed_fish:
            self.x_lim = [head_origin[0] - FOCAL_LIM_X_MINUS, head_origin[0] + FOCAL_LIM_X_PLUS]
            self.y_lim = [head_destination[1] - FOCAL_LIM_Y_MINUS, head_origin[1] + FOCAL_LIM_Y_PLUS]
        else:
            self.x_lim = [0, 0]
            self.y_lim = [0, 0]

        self.stateless = stateless  # important! when this is true, we can't trust analysis order!
        kwargs["input_video_has_plate"] = False
        ContourBasedTracking.__init__(self, **kwargs)
        self.save_n_back = 3
        self.thr_binary = 55  # todo
        self.prev_sb = None
        self.head_origin = head_origin
        self.head_destination = head_destination
        self.fixed_fish = fixed_fish
        self.friend_fish = friend_fish


    def _analyse(self, input_frame: np.array, noise_frame: np.array, fps: float, frame_number: int, additional=None):
        """

        :param input_frame:
        :param noise_frame:
        :param fps:
        :param frame_number:
        :param additional: not used here. List with inputs from other trackers etc
        :return: annotated frame (for debug) & output struct
        """
        if not self.friend_fish:
            if self.fixed_fish:
                input = copy.deepcopy(input_frame[self.y_lim[0]:self.y_lim[1], self.x_lim[0]:self.x_lim[1], :])
            else:
                input = input_frame
            input_frame_masked = self.__get_masked_input(input)
        else:
            input_frame[self.head_origin[1] - HEAD_DARKENING_PERIMETER:input_frame.shape[0],
            self.head_origin[0] - HEAD_DARKENING_PERIMETER:self.head_origin[0] + HEAD_DARKENING_PERIMETER, :] = 0
            input = copy.deepcopy(input_frame)
            input_frame_masked = input

        return self.__analyse_tail_only(input, input_frame_masked, fps, frame_number)

    def __get_masked_input(self, input_frame):
        input_frame_masked = input_frame
        input_frame_masked_no_head = input_frame_masked[15:, :, :]  # todo 15 is magic number from past
        gray = cv2.cvtColor(input_frame_masked_no_head, cv2.COLOR_BGR2GRAY)
        input_frame_masked_no_head, _ = self.__mask_image(gray, input_frame_masked_no_head, self.thr_binary)
        input_frame_masked[15:, :, :] = input_frame_masked_no_head
        input_frame_masked[0, :, :] = Colors.BLACK
        return input_frame_masked

    @staticmethod
    def __mask_image(gray, input_frame_masked_no_head, thr_binary, kernel=(5, 5)):
        mask = cv2.morphologyEx(cv2.threshold(gray, thr_binary, 255, cv2.THRESH_BINARY)[1],  # todo otsu?
                                cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))
        result = cv2.bitwise_and(input_frame_masked_no_head, input_frame_masked_no_head,
                                 mask=(mask.astype(np.uint8)))
        return result, mask

    @staticmethod
    def __fast_tail_edge_point(cleaned_fish, mask, head_point):
        """
        :param cleaned_fish: Original image, masked
        :param mask: mask marking where fish is
        :param mask_head:
        :param head_point:
        :return:
        """
        corners = cv2.goodFeaturesToTrack(mask, 4, 0.7, 50)  # todo quality
        if corners is None:
            return None
        corners = np.array([corner.ravel() for corner in corners])
        # a_frame = cleaned_fish.copy()
        # for p in corners:
        #     cv2.circle(a_frame, ContourBasedTracking.point_to_int(p), 4, Colors.YELLOW, thickness=cv2.FILLED)
        # ContourBasedTracking.show("a_frame", a_frame), cv2.waitKey(1)
        corner_dists = [(corner.ravel(), np.linalg.norm([corner.ravel()] - np.array(head_point))) for corner in corners]
        return max(corner_dists, key=lambda c_d: c_d[1])[0]

    def __analyse_tail_only(self, input, cleaned, fps, frame_number, min_fish_size=200):
        output = FrameAnalysisData()
        self.reason = ""
        output.is_ok = False
        output.fish_contour = None
        an_frame = input  # output frame

        _, mask = cv2.threshold(cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY), 70, 255, cv2.THRESH_BINARY)  # 95

        cleaned_fish = cv2.bitwise_and(cv2.cvtColor(input, cv2.COLOR_BGR2GRAY),
                                       cv2.cvtColor(input, cv2.COLOR_BGR2GRAY),
                                       mask=(mask.astype(np.uint8))).astype(cleaned.dtype)

        # Step 1- fish contour - return with error if incorrect
        fish_contour = self.get_fish_contour(cleaned_fish.astype(np.uint8),
                                             scale=self.scale, min_fish_size=min_fish_size)
        if fish_contour is None:
            logging.debug("Frame " + str(frame_number) + " didn't find fish")
            return an_frame, output  # default is_ok = False

        cleaned_fish, cleaned_non_fish, segment, mask, expanded_mask = \
            self.get_fish_segment_and_masks(an_frame, cleaned, fish_contour)


        if not self.is_fast_run:
            midline_path = self.get_midline_as_list_of_connected_points(cleaned_fish, output, self.friend_fish)
            if midline_path is not None:
                midline_path = np.array(midline_path)
                output.tail_data = FrameAnalysisData.TailData(fish_tail_tip_point=midline_path[0],
                                                              swimbladder_point=None,
                                                              tail_path=midline_path)
                output.is_ok = True


        return an_frame, output

    @classmethod
    def draw_output_on_annotated_frame(cls, an_frame, output: FrameAnalysisData,
                                       redraw_fish_contours=True, hunt_threshold=25, reason="",
                                       is_bout=None, velocity_norms=None,
                                       # font etc
                                       row_left_side_text=50, col_left_side_text=20, space_between_text_rows=25,
                                       col_right_side_text=680, row_right_side_text=20,
                                       text_font=cv2.FONT_HERSHEY_SIMPLEX, bold=2):
        if redraw_fish_contours:
            cv2.drawContours(an_frame, [output.fish_contour], -1, Colors.RED, 2)
            cv2.drawContours(an_frame, output.eyes_contour, -1, color=Colors.CYAN)

        if output.tail_data is not None:
            tail: FrameAnalysisData.TailData = output.tail_data
            cv2.circle(an_frame, cls.point_to_int(tail.tail_tip_point), 4, Colors.YELLOW, thickness=cv2.FILLED)
            tail_direction_angle = get_angle_to_horizontal(output.fish_head_origin_point, tail.tail_tip_point)
            tail_direction_angle = 360 - (tail_direction_angle + 90)  # clockwise from y axis
            # cv2.putText(an_frame, 'T: {0:.2f}'.format(tail_direction_angle),
            #             (col_left_side_text, row_left_side_text), text_font, 0.6, Colors.GREEN, bold)
            row_left_side_text += space_between_text_rows
            cv2.putText(an_frame, reason,
                        (col_left_side_text, row_left_side_text), text_font, 0.6, Colors.GREEN, bold)

            if output.tail_data.tail_path is not None:
                midline_path = output.tail_data.tail_path
                for index in range(1, len(midline_path)):
                    first_point = tuple(midline_path[index - 1])
                    second_point = tuple(midline_path[index])
                    cv2.line(an_frame, first_point, second_point, Colors.GREEN, 1)

    @classmethod
    def get_sb_point(cls, input, cleaned, add_head=20):
        # very high brightness is eyes + sb (big large ellipse)
        _, mask = cv2.threshold(cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)

        # bounding rectangle around head
        mask_head = np.full((cleaned.shape[0], cleaned.shape[1]), 0, dtype=cleaned.dtype)

        contours, _ = cls.get_contours(mask, ctype=cv2.RETR_EXTERNAL)  # get external only
        c = max(contours, key=cv2.contourArea)

        min_x, min_y, max_x, max_y = [np.inf, np.inf, 0, 0]
        (x, y, w, h) = cv2.boundingRect(c)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        top_left = (min_x - add_head, min_y)
        bottom_right = (max_x + add_head, max_y)
        cv2.rectangle(mask_head, top_left, bottom_right, Colors.WHITE, cv2.FILLED)

        cent_x, cent_y = np.average([min_x, max_x]), np.average([min_y, max_y])  # for sb point

        # cleaned_fish_tail_only = cv2.bitwise_and(cleaned, cleaned,
        #                                          mask=(np.bitwise_not(mask_head))).astype(cleaned.dtype)
        # cls.show("sb", cleaned_fish_tail_only)

        return [cent_x, cent_y], [cent_x, max_y]

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


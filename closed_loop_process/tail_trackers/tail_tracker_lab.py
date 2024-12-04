from closed_loop_process.tail_trackers.abstract_tail_tracker import AbstractTailTracker
import numpy as np
import cv2
import logging
from closed_loop_process.utils_closed_loop import machine_vision
from closed_loop_process.utils_closed_loop import Colors
import networkx

class LabTailTracker(AbstractTailTracker):
    '''
    Same logic as in the fixed fish tracker, too slow for realtime
    '''
    def __init__(self, image_processor, head_origin):
        super().__init__(image_processor)
        self.head_origin = head_origin

    def tail_tracking(self, raw_frame):
        super().tail_tracking(raw_frame)
        self.image_processor.load_mat(raw_frame)
        binary_img, _ = self.image_processor.preprocess_binary()
        tail_points = LabTailTracker.standalone_tail_tracking_func(binary_img, self.head_origin)
        tail_angles = LabTailTracker.calc_theta_matrix(tail_points)
        return tail_angles, tail_points

    @staticmethod
    def calc_theta_matrix(tail_data):
        """
        Receives a mx105x2 matrix of tail points (or a single 105x2 array for a single sample)
        and returns a mx98 matrix of angles cutting the last points.
        :return: angles features matrix
        """
        extra_size = 6  # Trust issues with end-of-fit points
        # Determine if input is 2D or 3D
        if tail_data.ndim == 3:
            # 3D case: multiple samples
            tail_diff = np.diff(tail_data, axis=1)  # Difference along the points axis
            tail_line = tail_diff[:, :, 0] + 1j * tail_diff[:, :, 1]  # Complex representation of vector differences
            d_theta = np.angle(tail_line)  # Calculate angles
            theta_matrix = np.unwrap(d_theta, axis=1)[:, :-extra_size + 1]  # Unwrap and remove unreliable end points
            theta_matrix = theta_matrix[:, 1:]  # Remove the first point to match desired output size
        elif tail_data.ndim == 2:
            if np.all(np.isnan(tail_data)):
                return np.zeros((1,98))
            # 2D case: single sample
            tail_diff = np.diff(tail_data, axis=0)  # Difference along the points axis
            tail_line = tail_diff[:, 0] + 1j * tail_diff[:, 1]  # Complex representation of vector differences
            d_theta = np.angle(tail_line)  # Calculate angles
            theta_matrix = np.unwrap(d_theta)[:-extra_size + 1]  # Unwrap and remove unreliable end points
            theta_matrix = theta_matrix[1:]  # Remove the first point to match desired output size
            theta_matrix = theta_matrix[np.newaxis, :]  # Add new axis to make it 2D compatible
        return theta_matrix

    @staticmethod
    def standalone_tail_tracking_func(binary_matrix, head_origin):
        height, width = binary_matrix.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[binary_matrix == 0] = [0, 0, 0]  # For black
        rgb_image[binary_matrix == 255] = [255, 255, 255]  # For white
        annotated_frame, fish_analysis_output = LabTailTracker.analyse(rgb_image, 0)
        if fish_analysis_output['tail_data'] is None:
            return np.full((105, 2), None, dtype=object)
        tail_points = fish_analysis_output['tail_data']['tail_path']
        interpolated_tail = LabTailTracker.get_smooth_tail(tail_points, head_origin)
        tail_points = interpolated_tail[1]
        return tail_points

    @staticmethod
    def analyse(binary_matrix, frame_num):
        input_frame_masked = LabTailTracker.get_masked_input(binary_matrix)
        return LabTailTracker.analyse_tail_only(binary_matrix, input_frame_masked, frame_num)

    @staticmethod
    def analyse_tail_only(input, cleaned, frame_number, min_fish_size=200):
        output = {}
        output['is_ok'] = False
        output['fish_contour'] = None
        output['tail_data'] = None
        an_frame = input  # output frame

        _, mask = cv2.threshold(cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY), 70, 255, cv2.THRESH_BINARY)  # 95

        cleaned_fish = cv2.bitwise_and(cv2.cvtColor(input, cv2.COLOR_BGR2GRAY),
                                       cv2.cvtColor(input, cv2.COLOR_BGR2GRAY),
                                       mask=(mask.astype(np.uint8))).astype(cleaned.dtype)

        # Step 1- fish contour - return with error if incorrect
        scale = 1
        fish_contour = LabTailTracker.get_fish_contour(cleaned_fish.astype(np.uint8),
                                                       scale=scale, min_fish_size=min_fish_size)
        if fish_contour is None:
            logging.debug("Frame " + str(frame_number) + " didn't find fish")
            return an_frame, output  # default is_ok = False

        cleaned_fish, cleaned_non_fish, segment, mask, expanded_mask = \
            LabTailTracker.get_fish_segment_and_masks(an_frame, cleaned, fish_contour)

        midline_path = LabTailTracker.get_midline_as_list_of_connected_points(cleaned_fish, output)
        if midline_path is not None:
            midline_path = np.array(midline_path)
            output['tail_data'] = dict(fish_tail_tip_point=midline_path[0],
                                       swimbladder_point=None,
                                       tail_path=midline_path)
            output['is_ok'] = True
        return an_frame, output

    @staticmethod
    def get_midline_as_list_of_connected_points(cleaned_fish, output, friend_fish=False):
        """
        Gets an image array of the fish and returns a list of points describing the fish midline (each point connected to the indices adjacent to it)

        :param cleaned_fish: image array only containing the fish.
        :param output: fish analysis struct used here in order to produce a clean midline image array.
        :return: A list of points (pixels) where each index on the list is connected to the previous and next indices.
        """
        fish_midline_clean = LabTailTracker.get_clean_fish_midline(cleaned_fish, output, remove_head=False)
        if fish_midline_clean is None:
            return None

        y = fish_midline_clean.nonzero()[0]
        x = fish_midline_clean.nonzero()[1]
        nonzero_midline_points = np.stack([x, y], axis=1)

        height = fish_midline_clean.shape[0]
        width = fish_midline_clean.shape[1]
        midline_graph = LabTailTracker.create_graph_from_points(midline_points=nonzero_midline_points, height=height,
                                                                width=width, )

        connected_components_subgraph = [
            midline_graph.subgraph(component).copy() for component in networkx.connected_components(midline_graph)
        ]
        biggest_subgraph = max(connected_components_subgraph, key=len)
        periphery = networkx.periphery(biggest_subgraph)

        periphery_paths = []
        for point in periphery:
            # points that are diagonals to one another weight less
            periphery_paths += [networkx.shortest_path(midline_graph, point, other_point, weight='weight') for
                                other_point in periphery]

        midline_path = max(periphery_paths, key=len)
        midline_path = LabTailTracker.align_midline_path(cleaned_fish, midline_path, output, friend_fish)
        return midline_path

    @staticmethod
    def align_midline_path(cleaned_fish, midline_path, output, friend_fish=False):
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

    @staticmethod
    def create_graph_from_points(midline_points, height, width, ):
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

    @staticmethod
    def get_clean_fish_midline(cleaned_fish, output, remove_head=True):
        """
        Turns the isolated image of the fish into a black and white frame containing only the thinned out midline of the fish.

        :param cleaned_fish: image array only containing the fish.
        :param output: fish analysis struct used here in order to produce a clean midline image array.
        :return: A black and white frame containing only the thinned out midline of the fish.

        Assumptions:
        The image is large enough for the kernel
        """
        clean_bnw_for_tail_threshold = 55
        bnw_cleaned_for_tail = machine_vision.frame_to_bnw(frame=cleaned_fish, thresh=clean_bnw_for_tail_threshold)
        # bnw_cleaned_for_tail = machine_vision.fill_contour(bnw_cleaned_for_tail) ##YR added to fix detection of skeleton
        bnw_cleaned_for_tail = machine_vision.fill_contour_without_middle(
            bnw_cleaned_for_tail)  ##YR added to fix detection of skeleton

        # We use the logical "or" of two different thinning algorithms,
        # This proved to yield better results when trying to find a continuous midline path.
        skeleton = cv2.bitwise_or(machine_vision.skeleton(frame=cleaned_fish),
                                  cv2.ximgproc.thinning(src=bnw_cleaned_for_tail))

        cv2.drawContours(skeleton, None, contourIdx=-1, color=Colors.WHITE, thickness=cv2.FILLED)
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
        minimum_midline_contour_area = 2
        skeleton_contours = [contour for contour in skeleton_contours
                             if cv2.contourArea(contour) > minimum_midline_contour_area]

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
    def get_fish_contour(gray, close_kernel=(5, 5), min_fish_size=50, max_fish_size=50000, scale=1,
                         threshold1=30, threshold2=200):
        contours, _ = LabTailTracker.get_contours(gray, ctype=cv2.RETR_EXTERNAL, close_kernel=close_kernel,
                                                  threshold1=threshold1, threshold2=threshold2)  # get external only
        # remove paramecia
        contours = [c for c in contours if min_fish_size * scale <= cv2.contourArea(c) <= max_fish_size * scale]
        if len(contours) > 0:
            return max(contours, key=cv2.contourArea)  # if cleaned, fish is largest
        return None

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
    def get_masked_input(input_frame):
        thr_binary = 55
        input_frame_masked = input_frame
        input_frame_masked_no_head = input_frame_masked[15:, :, :]  # todo 15 is magic number from past
        gray = cv2.cvtColor(input_frame_masked_no_head, cv2.COLOR_BGR2GRAY)
        input_frame_masked_no_head, _ = LabTailTracker.mask_image(gray, input_frame_masked_no_head, thr_binary)
        input_frame_masked[15:, :, :] = input_frame_masked_no_head
        input_frame_masked[0, :, :] = Colors.BLACK
        return input_frame_masked

    @staticmethod
    def mask_image(gray, input_frame_masked_no_head, thr_binary, kernel=(5, 5)):
        mask = cv2.morphologyEx(cv2.threshold(gray, thr_binary, 255, cv2.THRESH_BINARY)[1],  # todo otsu?
                                cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))
        result = cv2.bitwise_and(input_frame_masked_no_head, input_frame_masked_no_head,
                                 mask=(mask.astype(np.uint8)))
        return result, mask

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

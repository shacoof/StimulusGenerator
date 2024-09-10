import itertools
import numpy as np
import cv2

from classic_cv_trackers import Colors as Colors

GRAYSCALE_WHITE = 255
POINT_POLYGON_VALUES = {
    1: 'inside',
    -1: 'outside',
    0: 'edge',
}


def frame_to_bnw(
    frame,
    thresh,
    thresh_maxval=255,
):
    gray_frame = cv2.cvtColor(
        src=frame,
        code=cv2.COLOR_BGR2GRAY,
    )

    ret, bnw_frame = cv2.threshold(
        src=gray_frame,
        thresh=thresh,
        maxval=thresh_maxval,
        type=cv2.THRESH_BINARY,
    )

    return bnw_frame


def extract_background(
    path,
):
    cap = cv2.VideoCapture(path)

    try:
        ret, background = cap.read()
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is not None:
                kernel_size = (11, 11,)
                frame = cv2.GaussianBlur(
                    src=frame,
                    ksize=kernel_size,
                    sigmaX=0,
                )
                background = cv2.min(
                    frame,
                    background,
                )
            else:
                break
    except Exception as exc:
        raise exc
    finally:
        cap.release()

    return background


def show_image(
    image,
    delay=0,
    destroy_at_end=True,
    magnify=False,
    magnify_by=10,
):
    window = cv2.namedWindow(
        winname='image',
        flags=cv2.WINDOW_NORMAL,
    )
    cv2.imshow(
        winname='image',
        mat=image,
    )
    if magnify:
        width = image.shape[0]
        height = image.shape[1]

        cv2.resizeWindow(
            winname='image',
            width=width * magnify_by,
            height=height * magnify_by,
        )
        cv2.waitKey(
            delay=1,
        )

    cv2.waitKey(
        delay=delay,
    )

    if destroy_at_end:
        cv2.destroyAllWindows()


def blur_image_to_bnw(
    frame,
    blur_intensity,
    bnw_thresh=80,
):
    if blur_intensity > 1:
        blur_intensity = (blur_intensity * 2) - 1

    gray_frame = cv2.cvtColor(
        src=frame,
        code=cv2.COLOR_BGR2GRAY,
    )
    blur_kernel = (
        blur_intensity,
        blur_intensity,
    )

    gray_blurred_frame = cv2.GaussianBlur(
        src=gray_frame,
        ksize=blur_kernel,
        sigmaX=0,
    )
    ret, bnw_frame = cv2.threshold(
        src=gray_blurred_frame,
        thresh=bnw_thresh,
        maxval=255,
        type=cv2.THRESH_BINARY,
    )

    return bnw_frame


def open_bnw_blob(
    frame,
    open_intensity,
):
    open_kernel_shape = (
        open_intensity,
        open_intensity,
    )
    kernel = np.ones(
        shape=open_kernel_shape,
        dtype=np.uint8,
    )
    bnw_opened_frame = cv2.morphologyEx(
        src=frame,
        op=cv2.MORPH_OPEN,
        kernel=kernel,
    )

    return bnw_opened_frame


def skeleton(
    frame,
):
    gray_frame = cv2.cvtColor(
        src=frame,
        code=cv2.COLOR_BGR2GRAY,
    )
    ret, bnw_frame = cv2.threshold(
        src=gray_frame,
        thresh=10,
        maxval=255,
        type=cv2.THRESH_BINARY,
    )

    size = np.size(
        a=bnw_frame,
    )
    skel = np.zeros(
        shape=bnw_frame.shape,
        dtype=np.uint8,
    )

    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_CROSS,
        ksize=(3, 3),
    )

    while True:
        opened_image = cv2.morphologyEx(
            src=bnw_frame,
            op=cv2.MORPH_OPEN,
            kernel=kernel,
        )

        subtracted = cv2.subtract(
            src1=bnw_frame,
            src2=opened_image,
        )
        eroded = cv2.erode(
            src=bnw_frame,
            kernel=kernel,
        )

        skel = cv2.bitwise_or(
            src1=skel,
            src2=subtracted,
        )
        bnw_frame = eroded.copy()

        zero_pixels_count = cv2.countNonZero(
            src=bnw_frame,
        )
        if zero_pixels_count == 0:
            break

    return skel

def get_mask_out_of_bnw_blob(
    bnw_frame,
    inverted_mask=False,
    mask_type='BGR',
):
    all_contours, hierarchy = cv2.findContours(
        image=bnw_frame,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE,
    )

    if all_contours == []:
        return None

    contour = all_contours[0]

    frame_shape = (
        bnw_frame.shape[0],
        bnw_frame.shape[1],
        3,
    )
    mask = np.zeros(
        shape=frame_shape,
        dtype=np.uint8,
    )

    cv2.drawContours(
        image=mask,
        contours=all_contours,
        contourIdx=0,
        color=Colors.WHITE,
        thickness=-1,
    )

    if inverted_mask:
        mask = cv2.bitwise_not(
            src=mask,
        )

    if mask_type == 'BGR':
        return mask

    gray_mask = cv2.cvtColor(
        src=mask,
        code=cv2.COLOR_BGR2GRAY,
    )

    if mask_type == 'gray':
        return gray_mask

    elif mask_type == 'bnw':
        ret, bnw_mask = cv2.threshold(
            src=gray_mask,
            thresh=20,
            maxval=255,
            type=cv2.THRESH_BINARY,
        )
        return bnw_mask

    raise TypeError('Invalid mask_type passed as argument')

def find_closest_white_pixel_to_point_on_average(
    bnw_blob,
    point,
    width,
    height,
):
    # Potentially buggy
    distance_matrix = cv2.distanceTransform(
        src=bnw_blob,
        distanceType=cv2.DIST_C,
        maskSize=None,
        dst=None,
        dstType=None,
    )

    distance_from_pixel = distance_matrix[
        point[0],
        point[1],
    ]
    radius = int(
        distance_matrix[
            point[0],
            point[1],
        ]
    )
    if radius > 3:
        radius = int(
            radius / 2,
        )

    frame_shape = (
        bnw_blob.shape[0],
        bnw_blob.shape[1],
    )
    mask = np.zeros(
        shape=frame_shape,
        dtype=np.uint8,
    )
    cv2.circle(
        img=mask,
        center=point,
        radius=radius,
        color=Colors.WHITE,
        thickness=-1,
    )

    masked_blob = cv2.bitwise_and(
        src1=mask,
        src2=bnw_blob,
    )
    nonzero_values = cv2.findNonZero(
        src=masked_blob,
    )
    nonzero_values_amount = len(
        nonzero_values,
    )
    if nonzero_values_amount > 1:
        closest_point_on_average = get_center_of_mass_from_single_blob(
            masked_blob,
        )
    else:
        closest_point_on_average = np_point_to_tuple_point(
            np_point=nonzero_values[0],
        )

    return closest_point_on_average


def check_point_in_blob_status(
    bnw_blob,
    point,
):
    blob_contours, hierarchy = cv2.findContours(
        image=bnw_blob,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_NONE,
    )

    result = None
    for contour in blob_contours:
        result = cv2.pointPolygonTest(contour=contour, pt=point, measureDist=False)
        if result >= 0:
            return POINT_POLYGON_VALUES[result]

    if result is None:  # error
        return None
    return POINT_POLYGON_VALUES[result]


def get_single_frame_from_video(
    path,
    frame_number=0,
):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()

    frame_counter = 0
    try:
        while(cap.isOpened()):
            ret, frame = cap.read()

            if frame is not None:
                if frame_counter == frame_number:
                    break
            else:
                break

            frame_counter += 1
    except Exception as exc:
        raise exc
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return frame


def flatten_contour_list(
    contour_list,
):
    return [
        point
        for contour in contour_list
        for point in contour
    ]


def get_angle_between_vectors(
    vec_1,
    vec_2,
):
    vec_1_angle = np.arctan2(
        vec_1[1],
        vec_1[0],
    )
    vec_2_angle = np.arctan2(
        vec_2[1],
        vec_2[0],
    )

    angle = vec_2_angle - vec_1_angle
    normalized_angle = angle
    if angle > np.pi:
        normalized_angle = angle - 2 * np.pi
    elif angle <= -np.pi:
        normalized_angle = angle + 2 * np.pi

    return float(normalized_angle)


def get_chebyshev_distance_for_point_in_points(
    points,
    point,
):
    points_with_distance_from_point = []
    for p in points:
        x = point[0] - p[0]
        y = point[1] - p[1]

        chebyshev_distance = max(
            np.abs(x),
            np.abs(y),
        )

        points_with_distance_from_point.append(
            {
                'point': p,
                'distance': chebyshev_distance,
            },
        )

    return points_with_distance_from_point


def find_gaps_between_edges(
    edges_list,
    width,
    height,
):
    adjacent_edges_points_list = []
    for edges in edges_list:
        edges = np_points_to_tuple_points(
            np_points=edges,
        )

        adjacent_edges_points_list.append(
            set(
                get_adjacent_points_for_points(
                    points=edges,
                    radius=1,
                    width=width,
                    height=height,
                ),
            ),
        )

    intersections = set()
    for adjacent_edges_points in adjacent_edges_points_list:
        adjacent_edges_points_list_copy = adjacent_edges_points_list[:]
        adjacent_edges_points_list_copy.remove(
            adjacent_edges_points,
        )
        all_other_edges_adjacent_points = set.union(
            *adjacent_edges_points_list_copy,
        )

        intersection = adjacent_edges_points.intersection(
            all_other_edges_adjacent_points,
        )

        intersections = intersections.union(
            intersection,
        )
    intersections = list(
        intersections,
    )

    intersections = tuple_points_to_np_points_for_cv2(
        tuple_points=intersections,
    )
    return intersections


def get_center_of_mass_from_single_blob(
    bnw_frame,
):
    all_contours_sorted = find_contours_with_area_sorted(
        bnw_frame=bnw_frame,
    )
    if all_contours_sorted == []:
        return None

    contour = all_contours_sorted[0]['contour']

    moment_mat = cv2.moments(contour)
    cx = int(
        moment_mat['m10'] / moment_mat['m00']
    )
    cy = int(
        moment_mat['m01'] / moment_mat['m00']
    )
    return (cx, cy,)


def create_chebyshev_dict_from_blob(
    bnw_blob_frame,
    point,
):
    width = bnw_blob_frame.shape[0]
    height = bnw_blob_frame.shape[1]

    blob_points = turn_blob_frame_to_points(
        blob_frame=bnw_blob_frame,
    )

    chebyshev_dict = create_chebyshev_dict_from_blob_points(
        blob_points=blob_points,
        start_point=point,
        height=height,
        width=width,
    )

    return chebyshev_dict


def turn_blob_frame_to_points(
    blob_frame,
):
    np_points = np.argwhere(
        a=blob_frame == GRAYSCALE_WHITE,
    )
    np_points = np.flip(
        m=np_points,
    )

    cv2_np_points = [
        [np_point] for np_point in np_points
    ]
    tuple_points = np_points_to_tuple_points(
        np_points=cv2_np_points,
    )

    return tuple_points


def create_chebyshev_dict_from_blob_points(
    blob_points,
    start_point,
    height,
    width,
):
    if start_point in blob_points:
        blob_points.remove(
            start_point,
        )
    else:
        raise ValueError(
            'Start point was not found in the blob. (Centroid probably outside of blob)',
        )

    chebyshev_dict = _create_distance_dict_from_points(
        blob_points=blob_points,
        points=[start_point],
        height=height,
        width=width,
        distance_from_original_point=1,
    )

    chebyshev_dict[start_point] = 0

    return chebyshev_dict


def _create_distance_dict_from_points(
    blob_points,
    points,
    height,
    width,
    distance_from_original_point,
):
    adjacent_points = get_adjacent_points_for_points(
        points=points,
        radius=1,
        width=width,
        height=height,
    )
    adjacent_points_in_blob = [
        point for point in adjacent_points if point in blob_points
    ]

    chebyshev_dict_for_adjacent_points = {}
    for point in adjacent_points_in_blob:
        chebyshev_dict_for_adjacent_points[point] = distance_from_original_point

    blob_points_without_adjacent = [
        blob_point for blob_point in blob_points
        if blob_point not in adjacent_points
    ]

    no_adjacent_points_found = blob_points_without_adjacent == blob_points
    if no_adjacent_points_found:
        return chebyshev_dict_for_adjacent_points

    distance_from_original_point += 1
    chebyshev_dict = _create_distance_dict_from_points(
        blob_points=blob_points_without_adjacent,
        points=adjacent_points_in_blob,
        height=height,
        width=width,
        distance_from_original_point=distance_from_original_point,
    )

    chebyshev_dict.update(
        chebyshev_dict_for_adjacent_points,
    )

    return chebyshev_dict


def get_edges_from_points_list(
    points,
    width,
    height,
):
    edge_points = []
    points = np_points_to_tuple_points(
        np_points=points,
    )

    for point in points:
        adjacent_points = get_adjacent_points(
            point=point,
            radius=1,
            width=width,
            height=height,
        )
        adjacent_points_found = [
            adj_point for adj_point in adjacent_points if adj_point in points
        ]
        num_of_adjacent_points = len(
            adjacent_points_found,
        )

        if num_of_adjacent_points > 3:
            continue
        elif num_of_adjacent_points == 1:
            edge_points.append(
                point,
            )
        else:
            for adjacent_point in adjacent_points_found:
                max_distances_from_adjacent_point = [
                    max_index_distance(
                        adjacent_point,
                        other_adjacent_point
                    ) for other_adjacent_point in adjacent_points_found
                ]

                all_adjacent_points_are_adjacent_to_one_another = all(
                    distance <= 1 for distance in max_distances_from_adjacent_point
                )

                if not all_adjacent_points_are_adjacent_to_one_another:
                    break
            else:
                edge_points.append(
                    point,
                )

    edge_points = tuple_points_to_np_points_for_cv2(
        tuple_points=edge_points,
    )
    return edge_points


def max_index_distance(
    point_1,
    point_2,
):
    x_dist = np.abs(
        point_1[0] - point_2[0],
    )
    y_dist = np.abs(
        point_1[1] - point_2[1],
    )
    return max(
        x_dist,
        y_dist,
    )

def group_contiguous_points(
    points,
    radius,
    width,
    height,
    convert_to_tuple_points=True,
    return_np_points=True,
):
    grouped_contiguous_points_list = []
    if convert_to_tuple_points:
        points = np_points_to_tuple_points(
            np_points=points,
        )

    while points != []:
        contiguous_points = get_contiguous_points_for_points(
            contiguous_points=[points[0]],
            points=points,
            radius=radius,
            width=width,
            height=height,
        )
        grouped_contiguous_points_list.append(
            contiguous_points,
        )

        points = list(
            set(points) - set(contiguous_points),
        )

    if return_np_points:
        grouped_contiguous_points_list = [
            tuple_points_to_np_points_for_cv2(
                tuple_points=grouped_contiguous_points,
            ) for grouped_contiguous_points in grouped_contiguous_points_list
        ]

    return grouped_contiguous_points_list

def get_contiguous_points_for_points(
    points,
    contiguous_points,
    radius,
    width,
    height,
):
    points_without_found_points = []

    adjacent_points = get_adjacent_points_for_points(
        points=contiguous_points,
        radius=radius,
        width=width,
        height=height,
    )
    existing_adjacent_points = [
        adjacent_point for adjacent_point in adjacent_points
        if adjacent_point in points
    ]

    points_without_found_points = list(
        set(points) - set(contiguous_points) - set(existing_adjacent_points),
    )

    remaining_contiguous_points = []
    if existing_adjacent_points != []:
        remaining_contiguous_points = get_contiguous_points_for_points(
            points=points_without_found_points,
            contiguous_points=existing_adjacent_points,
            radius=radius,
            width=width,
            height=height,
        )

    all_contiguous_points = list(
        set.union(
            set(remaining_contiguous_points),
            set(existing_adjacent_points),
            set(contiguous_points),
        )
    )

    return all_contiguous_points


def get_adjacent_points_for_points(
    points,
    radius,
    width,
    height,
):
    all_adjacent = []

    for point in points:
        adjacent_points = get_adjacent_points(
            point=point,
            radius=radius,
            width=width,
            height=height,
        )

        all_adjacent += adjacent_points

    adjacent_points_without_starting_points = list(
        set(all_adjacent) - set(points),
    )

    return adjacent_points_without_starting_points


def get_adjacent_points(
    point,
    radius,
    width,
    height,
):
    cartesian_product = list(
        itertools.product(
            range(
                point[0] - radius,
                point[0] + radius + 1,
            ),
            range(
                point[1] - radius,
                point[1] + radius + 1,
            ),
        )
    )
    cartesian_product.remove(
        point,
    )

    return [
        tup for tup in cartesian_product if
        tup[0] > 0 and tup[1] > 0 and
        tup[0] < width and tup[1] < height
    ]


def get_adjacent_points_for_points_in_points(
    points_source,
    points,
    radius,
    width,
    height,
):
    adjacent_points = get_adjacent_points_for_points(
        points=points,
        radius=radius,
        width=width,
        height=height,
    )

    points_in_source = [
        adjacent_point for adjacent_point in adjacent_points
        if adjacent_point in points_source
    ]

    return points_in_source


def fit_vector_to_points_segment(
    points_segment,
    main_point_on_segment,
):
    points_segment_np = tuple_points_to_np_points_for_cv2(
        tuple_points=points_segment,
    )
    direction_vector_quadrants = get_vector_quadrant(
        points_segment=points_segment,
        main_point_on_segment=main_point_on_segment,
    )

    radius_epsilon = 0.01
    angle_epsilon = 0.01
    line = cv2.fitLine(
        points=points_segment_np,
        distType=cv2.DIST_L2,
        param=0,
        reps=radius_epsilon,
        aeps=angle_epsilon,
    )

    vx, vy, x, y = line
    if direction_vector_quadrants[0] * vx < 0:
        vx = vx * -1
    if direction_vector_quadrants[1] * vy < 0:
        vy = vy * -1

    direction_vector = {
        'x': vx,
        'y': vy,
        'x_quadrant': direction_vector_quadrants[0],
        'y_quadrant': direction_vector_quadrants[1],
    }

    return direction_vector


def get_vector_quadrant(
    points_segment,
    main_point_on_segment,
):
    chebyshev_distance = get_chebyshev_distance_for_point_in_points(
        points=points_segment,
        point=main_point_on_segment,
    )
    furthest_point = max(
        chebyshev_distance,
        key=lambda item: item['distance'],
    )
    furthest_point_from_main_point_on_segment = furthest_point['point']

    x_direction = furthest_point_from_main_point_on_segment[0] - \
        main_point_on_segment[0]
    y_direction = furthest_point_from_main_point_on_segment[1] - \
        main_point_on_segment[1]

    tail_vector_quadrants = np.sign(
        (
            x_direction,
            y_direction,
        )
    )

    return tail_vector_quadrants


def np_int32_tuple_to_int(
    np_tuple,
):
    return (
        int(np_tuple[0]),
        int(np_tuple[1]),
    )


def np_points_to_tuple_points(
    np_points,
):
    tuple_points = [
        np_point_to_tuple_point(
            np_point=np_point,
        ) for np_point in np_points
    ]

    duplicates_removed_tuple_points = list(
        set(tuple_points),
    )

    return duplicates_removed_tuple_points


def np_point_to_tuple_point(
    np_point,
):
    tuple_point = (
        np_point[0][0],
        np_point[0][1],
    )

    return tuple_point


def tuple_points_to_np_points_for_cv2(
    tuple_points,
):
    wrapped_tuple_points = [
        [point] for point in tuple_points
    ]

    np_points = np.array(
        object=wrapped_tuple_points,
        dtype=np.int32,
    )

    return np_points


def len_to_color(
    length,
):
    if length <= 5:
        return (0, 0, 255)
    elif length <= 25:
        return (0, 0, 127)
    elif length <= 50:
        return (0, 255, 0)
    elif length <= 75:
        return (0, 127, 0)
    elif length <= 100:
        return (0, 255, 255)
    elif length <= 125:
        return (255, 0, 0)
    elif length <= 150:
        return (127, 0, 0)
    elif length <= 200:
        return (255, 0, 255)


def is_similar(
    image1,
    image2,
):
    same_shape = image1.shape == image2.shape
    if not same_shape:
        return False

    is_same_bit_values = not(
        np.bitwise_xor(
            image1,
            image2,
        ).any()
    )

    return is_same_bit_values


def save_image(
    dest_path,
    frame,
):
    cv2.imwrite(
        filename=dest_path,
        img=frame,
    )


def load_image(
    source_path,
    grayscale=False,
):
    flags = cv2.IMREAD_UNCHANGED
    if grayscale:
        flags = cv2.IMREAD_GRAYSCALE

    image = cv2.imread(
        filename=source_path,
        flags=flags,
    )

    return image


def find_contours_with_area_sorted(
    bnw_frame
):
    all_contours, hierarchy = cv2.findContours(
        image=bnw_frame,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE,
    )

    contours_with_area = []
    for contour in all_contours:
        area = cv2.contourArea(
            contour=contour,
        )

        contours_with_area.append(
            {
                'contour': contour,
                'area': area,
            },
        )

    def sort_by_area(contour): return contour['area']

    return sorted(
        contours_with_area,
        key=sort_by_area,
        reverse=True,
    )

def find_contours_with_length_sorted(
    bnw_frame
):
    all_contours, hierarchy = cv2.findContours(
        image=bnw_frame,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE,
    )

    contours_with_area = []
    for contour in all_contours:
        length = len(contour)

        contours_with_area.append(
            {
                'contour': contour,
                'length': length,
            },
        )

    def sort_by_length(contour): return contour['length']

    return sorted(
        contours_with_area,
        key=sort_by_length,
        reverse=True,
    )


def get_farthest_points_distance(
    contour,
):
    flattened_contour = flatten_contour_list(contour_list=[contour])

    top_left_pixel = np.min(
        flattened_contour,
        axis=(1, 0)
    )
    bottom_right_pixel = np.max(
        flattened_contour,
        axis=(1, 0)
    )
    return np.abs(top_left_pixel - bottom_right_pixel)

def fill_contour(
    orig_image,
):
    im_floodfill = orig_image.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = orig_image | im_floodfill_inv
    
    return im_out

def fill_contour_without_middle(
    orig_image,
):
    center_of_mass = get_center_of_mass_from_single_blob(orig_image)
    im_floodfill = orig_image.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, center_of_mass, 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = orig_image | im_floodfill_inv

    im_out = cv2.bitwise_and(im_out,fill_contour(orig_image))

    return im_out

def put_frame_number_on_frame(
    frame,
    frame_number,
    color=Colors.WHITE,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '{frame_number}'.format(
        frame_number=frame_number,
    )
    upper_left_corner = (50, 50)

    cv2.putText(
        img=frame,
        text=text,
        org=upper_left_corner,
        fontFace=font,
        fontScale=1,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return frame


def put_text_on_frame(
    frame,
    text,
    position,
    color=Colors.WHITE,
    font_size_scale=1,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '{text}'.format(
        text=text,
    )
    upper_left_corner = position

    cv2.putText(
        img=frame,
        text=text,
        org=upper_left_corner,
        fontFace=font,
        fontScale=font_size_scale,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return frame

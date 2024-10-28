import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def get_tail_angles(im, tail_start, tail_end, n_segments = 25, tail_filter_width = 0, n_output_segments = 98
                    , window_size = 10):
    """Finds the tail for an embedded fish, given the starting point and
    the direction of the tail. Alternative to the sequential circular arches.

    Parameters
    ----------
    im :
        image to process
    tail_start :
        starting point (x, y) in pixels
    tail_end :
        tail tip (x, y) in pixels
    n_segments :
        number of desired segments (Default value = 25)
    window_size :
        window size in pixel for center-of-mass calculation (Default value = 10)
    tail_filter_width :
        for filtering sharp angles (probably)
    n_output_segments:
        length of angles after interpolation
    Returns
    -------
    type
        list of angles, points representing the tail
    """
    start_x, start_y = tail_start
    tail_end_x, tail_end_y = tail_end
    tail_length_x = tail_end_x - start_x
    tail_length_y = tail_end_y - start_y

    # Calculate tail length:
    length_tail = np.sqrt(tail_length_x ** 2 + tail_length_y ** 2)

    # Segment length from tail length and n of segments:
    seg_length = length_tail / n_segments

    n_segments += 1

    # Initial displacements in x and y:
    disp_x = tail_length_x / n_segments
    disp_y = tail_length_y / n_segments
    first_angle = np.arctan2(disp_x, disp_y)
    angles = np.full(n_segments - 1, np.nan)
    points = np.zeros((n_segments - 1, 2))
    points[0, :] = start_x, start_y
    halfwin = window_size / 2
    for i in range(1, n_segments):
        # Use next segment function for find next point
        # with center-of-mass displacement:
        start_x, start_y, disp_x, disp_y, acc = _next_segment(
            im, start_x, start_y, disp_x, disp_y, halfwin, seg_length
        )
        if start_x < 0:
            print("W:segment {} not detected".format(i))
            break

        abs_angle = np.arctan2(disp_x, disp_y)
        points[i - 1, :] = start_x + disp_x, start_y + disp_y
        angles[i - 1] = abs_angle

    # we want angles to be continuous, this removes potential 2pi discontinuities
    angles = np.unwrap(angles)
    # we do not need to record a large amount of angles
    if tail_filter_width > 0:
        angles = gaussian_filter1d(angles, tail_filter_width, mode="nearest")

    # angles[:4] = np.clip(angles[:4], -0.1, 0.1)
    # Generate the x-values for the original data points
    x_original = np.linspace(0, 1, n_segments - 1)
    # Step 1: Perform a degree-7 polynomial fit
    poly_coeffs = np.polyfit(x_original, angles, deg=7)
    # Step 2: Create a polynomial from the coefficients
    poly = np.poly1d(poly_coeffs)
    # Evaluate the polynomial at the original x-values (smooth the angles)
    angles_smooth = poly(x_original)
    # Step 3: Perform interpolation to get n_output_segments points
    angles_interpolated = np.interp(
        np.linspace(0, 1, n_output_segments),  # New points to interpolate to
        np.linspace(0, 1, n_segments - 1),  # Old points (from original angles)
        angles_smooth  # Polynomial-smoothed angles
    )
    new_seg_length = length_tail / n_output_segments
    points = np.round(points[::-1])
    return -angles_interpolated[::-1], points, new_seg_length


def _next_segment(fc, xm, ym, dx, dy, halfwin, next_point_dist):
    """Find the endpoint of the next tail segment
    by calculating the moments in a look-ahead area

    Parameters
    ----------
    fc :
        image to find tail
    xm :
        starting point x
    ym :
        starting point y
    dx :
        initial displacement x
    dy :
        initial displacement y
    wind_size :
        size of the window to estimate next tail point
    next_point_dist :
        distance to the next tail point
    halfwin :


    Returns
    -------

    """

    # Generate square window for center of mass
    halfwin2 = halfwin**2
    y_max, x_max = fc.shape
    xs = min(max(int(round(xm + dx - halfwin)), 0), x_max)
    xe = min(max(int(round(xm + dx + halfwin)), 0), x_max)
    ys = min(max(int(round(ym + dy - halfwin)), 0), y_max)
    ye = min(max(int(round(ym + dy + halfwin)), 0), y_max)

    # at the edge returns invalid data
    if xs == xe and ys == ye:
        return -1, -1, 0, 0, 0

    # accumulators
    acc = 0.0
    acc_x = 0.0
    acc_y = 0.0
    for x in range(xs, xe):
        for y in range(ys, ye):
            lx = (xs + halfwin - x) ** 2
            ly = (ys + halfwin - y) ** 2
            if lx + ly <= halfwin2:
                acc_x += x * fc[y, x]
                acc_y += y * fc[y, x]
                acc += fc[y, x]

    if acc == 0:
        return -1, -1, 0, 0, 0

    # center of mass relative to the starting points
    mn_y = acc_y / acc - ym
    mn_x = acc_x / acc - xm

    # normalise to segment length
    a = np.sqrt(mn_y**2 + mn_x**2) / next_point_dist

    # check center of mass validity
    if a == 0:
        return -1, -1, 0, 0, 0

    # Use normalization factor
    dx = mn_x / a
    dy = mn_y / a

    return xm + dx, ym + dy, dx, dy, acc


def reproduce_tail_from_angles(start_point, angles, seg_length):
    """
    Reproduce the tail trace from given angles, starting point, and segment length.

    Parameters
    ----------
    start_point : tuple
        Starting point of the tail (x, y).
    angles : list or numpy array
        List of angles for each tail segment.
    seg_length : float
        The length of each tail segment.

    Returns
    -------
    tail_points : list of tuples
        List of (x, y) points representing the tail segments.
    """
    angles = -angles[::-1]
    # Initialize the starting point
    x, y = start_point
    tail_points = [(x, y)]

    # Iterate through each angle and compute the next point
    for angle in angles:
        # Calculate the next segment's displacement based on angle
        dx = seg_length * np.sin(angle)
        dy = seg_length * np.cos(angle)

        # Update the current point
        x += dx
        y += dy

        # Append the new point to the list
        tail_points.append((x, y))

    return tail_points
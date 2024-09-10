import logging

import cv2
import os
import numpy as np

from classic_cv_trackers import Colors

# todo can read from txt?
FRAME_ROWS = 896
FRAME_COLS = 900


def get_plate(gray, threshold=0.01, min_area=100000, visualize_movie=False):  # full plate ~500k, fish ~4k
    # note: can use threshold to reduce widening the plate, but it causes half circle shapes, req. rewrite code below
    contours0, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [(cv2.convexHull(cnt), cv2.contourArea(cnt), cnt) for cnt in contours0]
    hulls = [h for h in hulls if h[1] >= min_area]
    hulls = sorted(hulls, key=lambda r: -r[1])  # external and internal of plate

    if visualize_movie:
        r = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(r,contours0, -1, Colors.GREEN, cv2.FILLED)
        cv2.drawContours(r, [cnt for cnt in contours0 if cv2.contourArea(cnt) > min_area/2], -1, Colors.CYAN, cv2.FILLED)
        cv2.drawContours(r, [h[0] for h in hulls], -1, Colors.RED)
        cv2.imshow("Error", resize(r))
        cv2.waitKey(120)

    if len(hulls) >= 1:
        answers = []
        for plate, area, plate_contour in hulls:
            if len(plate) >= 5:  # fitEllipse req this
                ellipse = cv2.fitEllipse(plate)
                poly_ellipse = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2),
                                                                                           int(ellipse[1][1] / 2)),
                                                int(ellipse[2]), 0, 360, 5)
                ans = cv2.matchShapes(poly_ellipse, plate, 1, 0.0)  # lower = better match
                if ans <= threshold:
                    answers.append((ans, plate, area, plate_contour, poly_ellipse))
            # new videos - allow external image frame with inner ellipse
            elif len(plate) == 4 and [0, 0] in plate and [gray.shape[1] - 1, gray.shape[0] - 1] in plate:
                answers.append((1, plate, area, plate_contour, plate))
        answers = sorted(answers, key=lambda r: -r[2])  # external and internal of plate are ordered by area
        if len(answers) >= 2:
            return answers[0], answers[1]
        elif len(answers) >= 1:
            return answers[0], None
    return None


def resize(f):
    return cv2.resize(f, (round(f.shape[0] / 1.5), round(f.shape[1] / 1.5)))


def threshold_to_emphasize_plate(gray, block_size=21, open_kernel_size=(5, 5)):
    # Use binarization with assigning 255 to threshold, k=-0.3
    gray = cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY, block_size, -0.3,
                                         binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)

    # remove small white noise in result image
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size))
    return gray


def clean_plate(input_frame, threshold=0.05, additional_thickness_remove=2):
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY).astype(input_frame.dtype)
    black_n_white = threshold_to_emphasize_plate(gray)
    # cv2.imshow('black_n_white', resize(black_n_white))

    results = get_plate(black_n_white, threshold=threshold)
    if results is None:
        results = get_plate(gray, threshold=threshold)
        if results is None:
            results = get_plate(cv2.equalizeHist(gray), threshold=threshold)
            if results is None:
                logging.info("Didnt find plate")
                return None

    ((ans_o, plate_o, area_o, plate_contour_o, poly_ellipse_o), r2) = results

    if r2 is None:  # only outer plate - search inner using these results
        # use contour to search inner hull - this is done to remove artifact when fish is near the plate and
        # plate_contour_o is connected to the fish (the inner hull will catch only ellipse shape)
        mask = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(mask, [poly_ellipse_o], -1, Colors.WHITE, thickness=cv2.FILLED)
        cv2.drawContours(mask, [plate_contour_o], -1, Colors.BLACK, thickness=cv2.FILLED)
        cv2.drawContours(mask, [poly_ellipse_o], -1, Colors.WHITE, thickness=additional_thickness_remove)
        results2 = get_plate(mask, threshold=0.2)
        if results2 is not None and results2[1] is None:  # found inner only
            (r2, _) = results2
        elif results2 is not None:  # take most inner
            (_, r2) = results2

    mask = np.zeros(gray.shape[:2], np.uint8)
    if r2 is not None:
        (ans_i, plate_i, area_i, plate_contour_i, poly_ellipse_i) = r2
        cv2.drawContours(mask, [plate_i], -1, Colors.WHITE, thickness=cv2.FILLED)
        cv2.drawContours(mask, [plate_i], -1, Colors.BLACK, thickness=additional_thickness_remove)

    clean = cv2.bitwise_and(input_frame, input_frame, mask=mask).astype(input_frame.dtype)

    # cv2.imshow('mask o-i', resize(mask))
    # res = input_frame.copy()
    # if r2 is None:  # only outer plate
    #     cv2.drawContours(res, [plate_o], -1, Colors.CYAN)
    #     cv2.drawContours(res, [plate_contour_o], -1, Colors.BLUE)
    # else:
    #     cv2.drawContours(res, [plate_i], -1, Colors.RED, thickness=additional_thickness_remove)
    #     cv2.drawContours(res, [plate_i], -1, Colors.PINK)
    #     cv2.drawContours(res, [plate_o], -1, Colors.CYAN)
    #     cv2.drawContours(res, [plate_contour_o], -1, Colors.BLUE)
    # cv2.imshow('result of circle detect', resize(res))
    # cv2.waitKey(120)

    if r2 is not None:
        return clean
    return None  # didn't find plate


def static_noise_frame_from_full_event(frames_for_noise_dir):
    """ create a 'static noise' frame represents the mean+3*std shade of each pixel, sampled from input noise folder.

    :param frames_for_noise_dir: directory full path, contains the random frames of the fish video for noise calculation
    :return frame (in open-cv coordinates) of static noise for this fish
    """
    noise_frame_path = os.path.join(frames_for_noise_dir, 'noise_frame.npy')
    if os.path.isfile(noise_frame_path):
        return np.load(noise_frame_path)

    # Else - create and save for next run (slow)
    frames_list = [f for f in os.listdir(frames_for_noise_dir) if f.lower().endswith('.raw')]
    noise_frames = np.zeros([len(frames_list), FRAME_COLS, FRAME_ROWS])
    for i, file_frame in enumerate(frames_list):
        if i % 100 == 0:
            print(i)

        frame = np.fromfile(os.path.join(frames_for_noise_dir, file_frame), dtype=np.uint8)
        # support multiple frames in RAW - take 1st
        noise_frames[i, :, :] = frame.reshape([FRAME_COLS, FRAME_ROWS, -1])[:, :, 0]

    # Coordinates are transposed to match cv2 frame
    mean_frame = np.mean(noise_frames, axis=0)
    std_frame = np.std(noise_frames, axis=0)
    median_frame = np.median(noise_frames, axis=0)
    assert (mean_frame.shape == (FRAME_COLS, FRAME_ROWS))
    assert (std_frame.shape == (FRAME_COLS, FRAME_ROWS))
    assert (median_frame.shape == (FRAME_COLS, FRAME_ROWS))

    # add for noise debug - save more than final noise frame
    cv2.imwrite(os.path.join(frames_for_noise_dir, 'mean_frame.jpg'), mean_frame)
    np.save(os.path.join(frames_for_noise_dir, 'mean_frame.npy'), mean_frame)
    cv2.imwrite(os.path.join(frames_for_noise_dir, 'std_frame.jpg'), std_frame)
    np.save(os.path.join(frames_for_noise_dir, 'std_frame.npy'), std_frame)
    cv2.imwrite(os.path.join(frames_for_noise_dir, 'median_frame.jpg'), median_frame)
    np.save(os.path.join(frames_for_noise_dir, 'median_frame.npy'), median_frame)

    # temporary - for investigation add noise frames for segments of the whole movie
    step = 30 * 5
    for i in range(0, noise_frames.shape[0], step):  # frame each 2s => jump 30 * 4 to have 5m
        postfix = "-{0:n}-{1:n}-min_".format(i / 30, (i + step) / 30)
        for what in ['mean', 'median', 'std']:
            if what == 'mean':
                part = np.mean(noise_frames[i:(i + step), :, :], axis=0)
            elif what == 'median':
                part = np.median(noise_frames[i:(i + step), :, :], axis=0)
            else:
                part = np.std(noise_frames[i:(i + step), :, :], axis=0)
            cv2.imwrite(os.path.join(frames_for_noise_dir, what + '_frame' + postfix + '.jpg'), part)
            np.save(os.path.join(frames_for_noise_dir, what + '_frame' + postfix + '.npy'), part)

    noise_frame = np.clip(mean_frame + (3 * std_frame), 0, 255)
    np.save(noise_frame_path, noise_frame)
    cv2.imwrite(os.path.join(frames_for_noise_dir, 'noise_frame.jpg'), noise_frame)
    return noise_frame


def clean(input_frame, statistic_noise_frame):
    '''
    clean the noise of sample frames of the whole video, from the frame we want to 'clean'
    the input frames have the same shape.
    :param input_frame: frame as numpy array
    :param statistic_noise_frame: frame as numpy array
    :return: the clean frame as numpy array
    '''
    if len(input_frame.shape) == 3: # color
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY).astype(input_frame.dtype)

    input_frame = input_frame.astype(statistic_noise_frame.dtype)
    clean_frame = input_frame - statistic_noise_frame
    clean_frame = np.clip(clean_frame, 0, 255)
    return clean_frame


def clean2binary(clean, option, fixed_th=0):
    '''
    make it binary with different threshold
    option 0: no additional threshold
    option 1: looking for the next pick in the histogram (after the median)
    option 2: looking for the next pick in the histogram (after 0)
    option 3: a fixed threshold from a given input
    :param clean: a single frame without static noise
    :param option: which threshold to calculate
    :param fixed_th: used only in option 3
    :return: binary frame
    '''
    threshold = 0

    if option == 0:
        threshold = 0

    if option == 1:
        hist = np.histogram(clean, bins=257)[0]
        threshold = int(round(float(np.median(clean))))
        while hist[threshold] > hist[threshold + 1] or hist[threshold] > hist[threshold + 2]:
            threshold += 1
            if threshold == 256:
                break

    if option == 2:
        hist = np.histogram(clean, bins=257)[0]
        threshold = 0
        while hist[threshold] > hist[threshold + 1] or hist[threshold] > hist[threshold + 2]:
            threshold += 1
            if threshold == 256:
                break

    if option == 3:
        threshold = fixed_th

    if option == 4:
        return cv2.threshold(clean.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bin_frame = (clean > threshold).astype(int)
    return threshold, bin_frame

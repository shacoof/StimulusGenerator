from recognize_bout_start.RecognizeBout import RecognizeBout
from calibration.calibrate import Calibrator

import numpy as np
import os

def main_closed_loop():
    #user input
    debug_bout_detector = False
    debug_tail = False
    debug_PCA = False
    use_camera = True
    number_of_frames_calibration = 500
    camera_frame_rate = 500
    frames_from_bout = 35

    # directory settings
    images_path = f"Z:\Lab-Shared\Data\ClosedLoop\\20231204-f2\\raw_data"
    number_of_frames = 500

    import time



    if not use_camera:
        images_paths = []
        sorted_filenames = sorted(
            [filename for filename in os.listdir(images_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
        )
        for filename in sorted_filenames:
            images_paths.append(os.path.join(images_path, filename))
        number_of_frames = min(number_of_frames_calibration, len(images_paths))
        calibrator = Calibrator(calculate_PCA=True, live_camera=use_camera, images_path=images_path,
                                num_frames=number_of_frames_calibration, plot_bout_detector=debug_bout_detector)
        [pca_and_predict, image_processor, tail_tracker, bout_recognizer] = calibrator.start_calibrating()
        bout_frames = np.zeros((frames_from_bout, 105, 2))
        i = 0
        while i < number_of_frames:
            path = images_paths[i]
            image_processor.load_image(path)
            bout_recognizer.update(image_processor.get_image_matrix())
            verdict, diff = bout_recognizer.is_start_of_bout(i)

            if verdict and (i < number_of_frames - frames_from_bout):
                print("bout detected")
                bout_frames = np.zeros((frames_from_bout, 105, 2))

                # Analyze bout
                for j in range(frames_from_bout):
                    start_time = time.time()

                    binary_image = image_processor.preprocess_binary()
                    tail_tracker.load_binary_image(binary_image)
                    tail_points = tail_tracker.get_tail_points(i + j)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    #print(f"Elapsed time: {elapsed_time} seconds")
                    if debug_tail:
                        tail_tracker.plot_points(i+j)

                    bout_frames[j, :, :] = tail_points
                    path = images_paths[i + j]
                    image_processor.load_image(path)
                    bout_recognizer.update(image_processor.get_image_matrix())

                angle, distance = pca_and_predict.reduce_dimensionality_and_predict(bout_frames, to_plot=debug_PCA)
                print(
                    f"frame {i} predicted angle {round(angle[0][0], 2)}, predicted distance {round(distance[0][0], 2)}")

                # Skip the frames for the bout that was just detected
                i += frames_from_bout
            else:
                i += 1  # Proceed to the next frame if no bout is detected
    else: # use live camera

        calibrator = Calibrator(calculate_PCA=True, live_camera=True,
                                num_frames=number_of_frames_calibration, plot_bout_detector=debug_bout_detector)
        [pca_and_predict, image_processor, tail_tracker, bout_recognizer] = calibrator.start_calibrating()
        bout_frames = np.zeros((frames_from_bout, 105, 2))
        i = 0
        while i < number_of_frames:
            image_processor.load_image()
            bout_recognizer.update(image_processor.get_image_matrix())
            verdict, diff = bout_recognizer.is_start_of_bout(i)

            if verdict and (i < number_of_frames - frames_from_bout):
                print("bout detected")
                bout_frames = np.zeros((frames_from_bout, 105, 2))

                # Analyze bout
                for j in range(frames_from_bout):
                    binary_image = image_processor.preprocess_binary()
                    tail_tracker.load_binary_image(binary_image)
                    tail_points = tail_tracker.get_tail_points(i + j)

                    if debug_tail:
                        tail_tracker.plot_points(i + j)

                    bout_frames[j, :, :] = tail_points
                    image_processor.load_image()
                    bout_recognizer.update(image_processor.get_image_matrix())

                angle, distance = pca_and_predict.reduce_dimensionality_and_predict(bout_frames, to_plot=debug_PCA)
                print(
                    f"frame {i} predicted angle {round(angle[0][0], 2)}, predicted distance {round(distance[0][0], 2)}")

                # Skip the frames for the bout that was just detected
                i += frames_from_bout
            else:
                i += 1  # Proceed to the next frame if no bout is detected





#time to get tail points 0.02 sec
#time to tell if it is a start of a bout 0 sec


#Timer


# Run mode 1 - camera, debug bout detector
# Run mode 2 - camera, debug tail and PCA
# Run mode 3 - camera, no debug
# Run mode 4 - directory, debug bout detector
# Run mode 5 - directory, debug tail and PCA
# Run mode 6 - directory, no debug
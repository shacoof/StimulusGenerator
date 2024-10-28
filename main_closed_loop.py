from calibration.calibrate import Calibrator
from closed_loop_config import *
import numpy as np
import os
import time
import multiprocessing
from image_processor.tail_tracker import standalone_tail_tracking_func
from image_processor.stytra_tail_tracking import get_tail_angles, reproduce_tail_from_angles
from PIL import Image
import matplotlib.pyplot as plt

def worker_target(bout_frames_queue, tail_data_queue, head_origin, tail_tip):
    """Worker method that processes matrices from the input queue."""
    while True:
        tuple_val = bout_frames_queue.get()
        if tuple_val is None:
            # Terminate worker on receiving None
            break
        start_time = time.time()
        idx, image = tuple_val[0], tuple_val[1]
        tail_data, _, __ = get_tail_angles(image,head_origin,tail_tip)
        tail_data_queue.put((idx,tail_data))
        end_time = time.time()
        print(f"tail analysis time {end_time - start_time}")


class ClosedLoop:
    def __init__(self,pca_and_predict, image_processor, head_origin, tail_tip, bout_recognizer,
                 multiprocess_prediction_queue,
                 num_workers=3, num_bout_frames = frames_from_bout):
        """
        Preforms closed loop
        """
        self.pca_and_predict = pca_and_predict
        self.image_processor = image_processor
        self.bout_recognizer = bout_recognizer
        self.is_bout = False
        self.bout_index = 0
        self.bout_frames = np.zeros((frames_from_bout, 98))
        self.current_frame = 0
        self.multiprocess_prediction_queue = multiprocess_prediction_queue
        self.bout_start_time = 0
        self.num_bout_frames = num_bout_frames
        self.bout_frames_queue = multiprocessing.Queue(maxsize=num_bout_frames)
        self.tail_data_queue = multiprocessing.Queue(maxsize=num_bout_frames)
        self.workers = []
        self.head_origin = head_origin
        self.tail_tip = tail_tip
        self.num_workers = num_workers
        if use_multi_processing:
            self.start_workers()

    def start_workers(self):
        """Starts worker processes."""
        for _ in range(self.num_workers):
            frame_processing_worker = multiprocessing.Process(
                        target=worker_target,
                         args=(self.bout_frames_queue, self.tail_data_queue, self.head_origin, self.tail_tip))
            frame_processing_worker.start()
            self.workers.append(frame_processing_worker)

    def stop_workers(self):
        """Stops all worker processes by sending 'None' to the input queue."""
        for _ in range(self.num_workers):
            self.bout_frames_queue.put(None)
        for p in self.workers:
            p.join()

    def end_of_bout(self):
        bout_frames = np.zeros((frames_from_bout, 98))
        processed_count = 0
        while processed_count < self.num_bout_frames:
            # Check if the output queue is at capacity
            if self.tail_data_queue.qsize() == self.num_bout_frames:
                print(f"Output queue is full with {self.num_bout_frames} matrices.")
            # Get and store the results in the list by their original index
            if not self.tail_data_queue.empty():
                bout_index, tail_data = self.tail_data_queue.get()
                bout_frames[bout_index - 1, :] = tail_data
                processed_count += 1
        return bout_frames

    import matplotlib.pyplot as plt
    import numpy as np

    def debug_plot(self, start_point, angles, image, seg_length):
        """
        Reproduce the tail trace from given angles, starting point, and segment length.

        Parameters
        ----------
        start_point : tuple
            Starting point of the tail (x, y).
        angles : list or numpy array
            List of angles for each tail segment.
        image : numpy array
            The image on which the tail is being traced.
        seg_length : float
            The length of each tail segment.

        Returns
        -------
        tail_points : list of tuples
            List of (x, y) points representing the tail segments.
        """
        is_bout = self.is_bout  # Check if a bout is detected
        angles = -angles[::-1]
        x, y = start_point
        tail_points = [(x, y)]

        # Compute the tail trace points
        for angle in angles:
            dx = seg_length * np.sin(angle)
            dy = seg_length * np.cos(angle)
            x += dx
            y += dy
            tail_points.append((x, y))

        # Extract x and y points for plotting
        tail_x, tail_y = zip(*tail_points)

        # Clear the current figure
        plt.clf()

        # Create subplot (or get existing one if already created)
        ax = plt.gca()

        # Plot the image
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.plot(tail_x, tail_y, 'red', marker='o', markersize=1, label="Tail Trace")
        ax.scatter(start_point[0], start_point[1], color='red', label="Start Point")

        # Update the title based on the is_bout flag
        if is_bout:
            ax.set_title(f"Bout Detected frame {self.bout_index}", color='red')
        else:
            ax.set_title("Reproduced Tail from Angles")

        ax.axis('off')

        # Update the plot
        plt.draw()
        plt.pause(0.01)  # Pause briefly to ensure the plot window updates
        return tail_points

    def process_frame(self, frame):
        # time_now = time.time()
        # self.bout_start_time = time_now
        #
        # time.sleep(10)
        # if self.current_frame % 2 == 0:
        #     angle = -20
        #     distance = 0.5
        # else:
        #     angle = 20
        #     distance = 0.5
        # print(f"angle {angle} distance {distance}")
        # self.multiprocess_prediction_queue.put((angle, distance))
        # self.current_frame += 1

        if frame is None:
            return

        self.current_frame += 1
        self.image_processor.load_mat(frame)
        self.bout_recognizer.update(self.image_processor.get_image_matrix())
        binary_image, subtracted = self.image_processor.preprocess_binary()
        tail_angles, points, seg_length = get_tail_angles(subtracted, self.head_origin, self.tail_tip)
        if debug_mode:
            self.debug_plot(self.head_origin, tail_angles, subtracted, seg_length)

        # if this is a bout frame
        if self.is_bout:
            self.bout_index += 1
            self.bout_frames[self.bout_index, :] = tail_angles
            #last bout frame
            if self.bout_index == frames_from_bout - 1:
                self.is_bout = False
                angle, distance = self.pca_and_predict.reduce_dimensionality_and_predict(self.bout_frames, to_plot=debug_PCA)
                self.bout_frames = np.zeros((frames_from_bout, 98))
                self.multiprocess_prediction_queue.put((angle, distance))
                print(f"time to process bout {time.time() - self.bout_start_time}")
                print(
                    f"frame {self.current_frame} predicted angle {angle}, predicted distance {distance}")
        else:
            verdict, diff = self.bout_recognizer.is_start_of_bout(self.current_frame)
            if verdict:
                self.bout_start_time = time.time()
                self.bout_index = 0
                self.is_bout = True
                self.bout_frames[self.bout_index, :] = tail_angles

    def process_frame_multi_processing(self, frame):
        # program stop
        if frame is None:
            self.stop_workers()
            return

        self.current_frame += 1
        self.image_processor.load_mat(frame)
        self.bout_recognizer.update(self.image_processor.get_image_matrix())
        # if this is a bout frame
        if self.is_bout:
            self.bout_index += 1
            binary_image, subtracted = self.image_processor.preprocess_binary()
            # Put in multiprocess queue
            self.bout_frames_queue.put((self.bout_index, subtracted))

            #last bout frame
            if self.bout_index == frames_from_bout:
                self.is_bout = False
                self.bout_index = 0
                bout_frames = self.end_of_bout()
                angle, distance = self.pca_and_predict.reduce_dimensionality_and_predict(bout_frames, to_plot=debug_PCA)
                self.multiprocess_prediction_queue.put((angle, distance))
                bout_end_time = time.time()
                print(f"time to process bout {bout_end_time - self.bout_start_time}")
                print(
                    f"frame {self.current_frame} predicted angle {angle}, predicted distance {distance}")
        else:
            verdict, diff = self.bout_recognizer.is_start_of_bout(self.current_frame)
            if verdict:
                self.bout_start_time = time.time()
                self.is_bout = True
                self.bout_index += 1
                binary_image, subtracted = self.image_processor.preprocess_binary()
                # Put in multiprocess queue
                self.bout_frames_queue.put((self.bout_index, subtracted))



if __name__ == '__main__':

    # every 1/500 sec call function with frame
    queue_closed_loop_prediction = multiprocessing.Queue()
    # directory settings
    start_frame = 197751
    end_frame = 198450
    images_path = f"Z:\Lab-Shared\Data\ClosedLoop\\20231204-f2\\raw_data"
    calibrator = Calibrator(calculate_PCA=False, live_camera=False,
                            plot_bout_detector=False,start_frame=start_frame,
                            end_frame=start_frame + 500,
                            debug_PCA=False,images_path=images_path)
    [pca_and_predict, image_processor, bout_recognizer, head_origin, tail_tip] = calibrator.start_calibrating()
    closed_loop_class = ClosedLoop(pca_and_predict, image_processor, head_origin, tail_tip, bout_recognizer
                                   , queue_closed_loop_prediction)
    # load frames
    all_frame_mats = []

    for i in range(start_frame,end_frame+1):
        # Format the image filename based on the numbering pattern
        img_filename = f"img{str(i).zfill(12)}.jpg"
        img_path = os.path.join(images_path, img_filename)
        try:
            with Image.open(img_path) as img:
                image_matrix = np.array(img)
                all_frame_mats.append(image_matrix)
        except Exception as e:
            print(f"Error loading image: {e}")


    for i in range(len(all_frame_mats)):
        start_time = time.time()
        if use_multi_processing:
            closed_loop_class.process_frame_multi_processing(all_frame_mats[i])
        else:
            closed_loop_class.process_frame(all_frame_mats[i])
        end_time = time.time()
        print(f"time to process frame {end_time-start_time}")

        #time.sleep(0.006024)
        #time.sleep(0.002)

    closed_loop_class.process_frame_multi_processing(None)


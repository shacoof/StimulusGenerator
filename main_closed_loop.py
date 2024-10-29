from calibration.calibrate import Calibrator
from closed_loop_config import *
import numpy as np
import os
import time
import multiprocessing
from image_processor.stytra_tail_tracking import get_tail_angles, reproduce_tail_from_angles
from PIL import Image
import matplotlib.pyplot as plt

def plot_worker(shared_data, lock):
    """
    Worker function that listens to the plot_queue and handles the plotting
    of tail traces to avoid delays in the main process.
    """
    plt.ion()
    fig, ax = plt.subplots()
    img_plot = None
    tail_line, = ax.plot([], [], 'red', marker='o', markersize=1, label="Tail Trace")
    ax.axis('off')

    while True:
        # Read the shared values within the lock
        start_point = shared_data["start_point"]
        angles = shared_data["angles"]
        image = shared_data["image"]
        seg_length = shared_data["seg_length"]
        is_bout = shared_data["is_bout"]
        bout_index = shared_data["bout_index"]

        # Check if termination condition
        if start_point is None:
            break


        # Update image plot (only if needed)
        if img_plot is None:
            img_plot = ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        else:
            img_plot.set_data(image)

        # Calculate tail points
        x, y = start_point
        angles = -angles[::-1]
        tail_points = [(x, y)]
        for angle in angles:
            dx = seg_length * np.sin(angle)
            dy = seg_length * np.cos(angle)
            x += dx
            y += dy
            tail_points.append((x, y))

        tail_x, tail_y = zip(*tail_points)
        tail_line.set_data(tail_x, tail_y)

        # Set title
        ax.set_title(f"Bout Detected frame {bout_index}" if is_bout else "Reproduced Tail from Angles",
                     color='red' if is_bout else 'black')

        # Draw efficiently with blit
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.close()


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
        self.shared_data = multiprocessing.Manager().dict()
        self.lock = multiprocessing.Lock()
        self.plot_process = multiprocessing.Process(target=plot_worker, args=(self.shared_data, self.lock))
        self.shared_data["start_point"] = (0, 0)
        self.shared_data["angles"] = np.array([])
        self.shared_data["image"] = np.zeros((100, 100))
        self.shared_data["seg_length"] = 1.0
        self.shared_data["is_bout"] = False
        self.shared_data["bout_index"] = 0

        if debug_mode:
            self.plot_process.start()

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

    def update_shared_data(self, start_point, angles, image, seg_length):
        """
        Prepare the data for plotting and send it to the plot queue.
        """
        self.shared_data['start_point'] = start_point
        self.shared_data['angles'] = angles
        self.shared_data['image'] = image
        self.shared_data['seg_length'] = seg_length
        self.shared_data['is_bout'] = self.is_bout
        self.shared_data['bout_index'] = self.bout_index



    def stop_plotting(self):
        """Terminate the plot worker."""
        with self.lock:
            self.shared_data["start_point"] = None
        self.plot_process.join()

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
            if debug_mode:
                self.stop_plotting()
            return

        self.current_frame += 1
        self.image_processor.load_mat(frame)
        self.bout_recognizer.update(self.image_processor.get_image_matrix())
        binary_image, subtracted = self.image_processor.preprocess_binary()
        tail_angles, points, seg_length = get_tail_angles(subtracted, self.head_origin, self.tail_tip)
        if self.current_frame % 14 == 0:
            self.update_shared_data(self.head_origin, tail_angles, subtracted, seg_length)

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
    images_path = "\\\ems.elsc.huji.ac.il\\avitan-lab\Lab-Shared\Data\ClosedLoop\\20231204-f2\\raw_data"
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

    start_total = time.time()
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
    print(f"total time = {time.time() - start_total}")
    closed_loop_class.process_frame(None)


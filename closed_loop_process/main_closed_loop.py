from config_files.closed_loop_config import *
import numpy as np
import os
import time
import multiprocessing
from PIL import Image
import matplotlib.pyplot as plt
import psutil

def plot_worker(shared_data, lock):
    """
    Worker function that listens to the plot_queue and handles the plotting
    of tail traces to avoid delays in the main process.
    """
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('off')

    while True:
        with lock:
            tail_points = shared_data["tail_points"]
            image = shared_data["image"]
            is_bout = shared_data["is_bout"]
            bout_index = shared_data["frame_number"]

        if is_bout is None:
            break

        ax.clear()  # Clear the previous frame
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)

        tail_x, tail_y = zip(*tail_points)
        ax.plot(tail_x, tail_y, 'red', marker='o', markersize=1)

        ax.set_title(
            f"Bout Detected frame {bout_index}" if is_bout else f"Reproduced Tail from Angles frame {bout_index}",
            color='red' if is_bout else 'black')

        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.close()



class ClosedLoop:
    def __init__(self,pca_and_predict, image_processor, tail_tracker, bout_recognizer,
                 multiprocess_prediction_queue):
        """
        Preforms closed loop
        """
        self.pca_and_predict = pca_and_predict
        self.image_processor = image_processor
        self.bout_recognizer = bout_recognizer
        self.tail_tracker = tail_tracker
        self.is_bout = False
        self.bout_index = 0
        self.bout_frames = np.zeros((frames_from_bout, 98))
        self.current_frame = 0
        self.multiprocess_prediction_queue = multiprocess_prediction_queue
        self.bout_start_time = 0
        self.frame_time = time.time()
        self.shared_data = multiprocessing.Manager().dict()
        self.lock = multiprocessing.Lock()
        self.plot_process = multiprocessing.Process(target=plot_worker, args=(self.shared_data, self.lock))
        self.shared_data["tail_points"] = np.zeros((105, 2))
        self.shared_data["image"] = np.zeros((100, 100))
        self.shared_data["is_bout"] = False
        self.shared_data["frame_number"] = 0

        if emulator_with_camera:
            start_frame = 197751
            end_frame = 198450
            self.all_frame_mats = []
            images_path = "\\\ems.elsc.huji.ac.il\\avitan-lab\Lab-Shared\Data\ClosedLoop\\20231204-f2\\raw_data"
            for i in range(start_frame, end_frame + 1):
                # Format the image filename based on the numbering pattern
                img_filename = f"img{str(i).zfill(12)}.jpg"
                img_path = os.path.join(images_path, img_filename)
                try:
                    with Image.open(img_path) as img:
                        image_matrix = np.array(img)
                        self.all_frame_mats.append(image_matrix)
                except Exception as e:
                    print(f"Error loading image: {e}")
            self.number_of_frames = len(self.all_frame_mats)
        if debug_mode:
            self.plot_process.start()
            process5_psutil = psutil.Process(self.plot_process.pid)
            process5_psutil.cpu_affinity([5])



    def update_shared_data(self, tail_points, image):
        """
        Prepare the data for plotting and send it to the plot queue.
        """
        with self.lock:
            self.shared_data['tail_points'] = tail_points
            self.shared_data['image'] = image
            self.shared_data['is_bout'] = self.is_bout
            self.shared_data['frame_number'] = self.current_frame


    def stop_plotting(self):
        """Terminate the plot worker."""
        with self.lock:
            self.shared_data["is_bout"] = None
        self.plot_process.join()

    def process_frame(self, frame):
        """
        Processes a single frame and updates the multiprocess_prediction_queue accordingly to communicate to outer
        processes
        Args:
            frame: a np.array 2D matrice of the camera acquired image
        Returns:
        """
        time_now = time.time()
        self.bout_start_time = time_now

        time.sleep(10)
        if self.current_frame % 2 == 0:
            angle = -40
            distance = 0.5
        else:
            angle = 40
            distance = 1
        print(f"angle {angle} distance {distance}")
        self.multiprocess_prediction_queue.put((angle, distance))
        self.current_frame += 1

        print(time.time() - self.frame_time)
        self.frame_time = time.time()

        # if frame is None:
        #     if debug_mode:
        #         self.stop_plotting()
        #     return
        #
        # if emulator_with_camera:
        #     frame = self.all_frame_mats[self.current_frame%len(self.all_frame_mats)]
        #
        # print_time('before tail_tracking')
        # tail_angles, tail_points = self.tail_tracker.tail_tracking(frame)
        # print_time('after tail_tracking')
        #
        # if self.current_frame % 10 == 0 and debug_mode:
        #     self.update_shared_data(tail_points, frame)
        # self.current_frame +=1
        #
        # self.bout_recognizer.update(frame)
        # print_time('after bout_recognizer')
        # # if this is a bout frame
        # if self.is_bout:
        #     self.bout_index += 1
        #     self.bout_frames[self.bout_index, :] = tail_angles
        #     #last bout frame
        #     if self.bout_index == frames_from_bout - 1:
        #         self.is_bout = False
        #         angle, distance = self.pca_and_predict.reduce_dimensionality_and_predict(self.bout_frames, to_plot=False)
        #         self.bout_frames = np.zeros((frames_from_bout, 98))
        #         self.multiprocess_prediction_queue.put((angle, distance))
        #         print(f"time to process bout {time.time() - self.bout_start_time}")
        #         print(
        #             f"frame {self.current_frame} predicted angle {angle}, predicted distance {distance}")
        # else:
        #     verdict, diff = self.bout_recognizer.is_start_of_bout(self.current_frame)
        #     print_time('after is_start_of_bout')
        #     if verdict:
        #         self.bout_start_time = time.time()
        #         self.bout_index = 0
        #         self.is_bout = True
        #         self.bout_frames[self.bout_index, :] = tail_angles





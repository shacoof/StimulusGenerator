from recognize_bout_start.RecognizeBout import RecognizeBout
from calibration.calibrate import Calibrator
from closed_loop_config import *
from renderer.Renderer import Renderer
import numpy as np
import os
import time
import multiprocessing
from image_processor.tail_tracker import standalone_tail_tracking_func
from PIL import Image


def worker_target(bout_frames_queue, tail_data_queue, head_origin):
    """Worker method that processes matrices from the input queue."""
    while True:
        tuple_val = bout_frames_queue.get()
        if tuple_val is None:
            # Terminate worker on receiving None
            break
        start_time = time.time()
        idx, binary_image = tuple_val[0], tuple_val[1]
        tail_data = standalone_tail_tracking_func(binary_image, head_origin, 0, False)
        tail_data_queue.put((idx,tail_data))
        end_time = time.time()
        #print(f"tail analysis time {end_time - start_time}")


class ClosedLoop:
    def __init__(self,pca_and_predict, image_processor, head_origin, bout_recognizer, multiprocess_prediction_queue,
                 num_workers=3, num_bout_frames = 35):
        """
        Preforms closed loop
        """
        self.pca_and_predict = pca_and_predict
        self.image_processor = image_processor
        self.bout_recognizer = bout_recognizer
        self.is_bout = False
        self.bout_index = 0
        self.bout_frames = np.zeros((frames_from_bout, 105, 2))
        self.current_frame = 0
        self.renderer = Renderer()
        self.multiprocess_prediction_queue = multiprocess_prediction_queue
        self.bout_start_time = 0
        self.num_bout_frames = num_bout_frames
        self.bout_frames_queue = multiprocessing.Queue(maxsize=num_bout_frames)
        self.tail_data_queue = multiprocessing.Queue(maxsize=num_bout_frames)
        self.workers = []
        self.head_origin = head_origin
        self.num_workers = num_workers
        self.start_workers()

    def start_workers(self):
        """Starts worker processes."""
        for _ in range(self.num_workers):
            frame_processing_worker = multiprocessing.Process(
                        target=worker_target,
                         args=(self.bout_frames_queue, self.tail_data_queue, self.head_origin))
            frame_processing_worker.start()
            self.workers.append(frame_processing_worker)

    def stop_workers(self):
        """Stops all worker processes by sending 'None' to the input queue."""
        for _ in range(self.num_workers):
            self.bout_frames_queue.put(None)
        for p in self.workers:
            p.join()

    def end_of_bout(self):
        bout_frames = np.zeros((frames_from_bout, 105, 2))
        processed_count = 0
        while processed_count < self.num_bout_frames:
            # Check if the output queue is at capacity
            if self.tail_data_queue.qsize() == self.num_bout_frames:
                print(f"Output queue is full with {self.num_bout_frames} matrices.")
            # Get and store the results in the list by their original index
            if not self.tail_data_queue.empty():
                bout_index, tail_data = self.tail_data_queue.get()
                bout_frames[bout_index - 1, :, :] = tail_data  # Place the result in the correct position based on the index
                processed_count += 1
        return bout_frames


    def process_frame(self, frame):
        # time_now = time.time()
        # print(f"time {time_now - self.bout_start_time}")
        # self.bout_start_time = time_now
        #
        # time.sleep(1)
        # if self.current_frame % 2 ==0:
        #     new_angle, new_distance = self.renderer.calc_new_angle_and_size(40, 1)
        # else:
        #     new_angle, new_distance = self.renderer.calc_new_angle_and_size(-40, 1)
        # print(f"angle {new_angle} distance {new_distance}")
        # self.multiprocess_prediction_queue.put((new_angle, new_distance))
        # self.current_frame += 1

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
            binary_image = self.image_processor.preprocess_binary()
            # Put in multiprocess queue
            self.bout_frames_queue.put((self.bout_index,binary_image))

            #last bout frame
            if self.bout_index == frames_from_bout:
                self.is_bout = False
                self.bout_index = 0
                cur_time = time.time()
                print(f"time from start {cur_time - self.bout_start_time}")
                bout_frames = self.end_of_bout()
                angle, distance = self.pca_and_predict.reduce_dimensionality_and_predict(bout_frames, to_plot=debug_PCA)
                new_angle, new_distance = self.renderer.calc_new_angle_and_size(angle, distance)
                self.multiprocess_prediction_queue.put((new_angle, new_distance))
                bout_end_time = time.time()
                print(f"time to process bout {bout_end_time - self.bout_start_time}")
                # print(
                #     f"frame {self.current_frame} predicted angle {angle}, predicted distance {distance}")
                # print(
                #     f"frame {self.current_frame} new angle {new_angle}, new size {new_distance}")
        else:
            verdict, diff = self.bout_recognizer.is_start_of_bout(self.current_frame)
            if verdict:
                self.bout_start_time = time.time()
                self.is_bout = True
                self.bout_index += 1
                binary_image = self.image_processor.preprocess_binary()
                # Put in multiprocess queue
                self.bout_frames_queue.put((self.bout_index,binary_image))


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
    [pca_and_predict, image_processor, bout_recognizer, head_origin] = calibrator.start_calibrating()
    closed_loop_class = ClosedLoop(pca_and_predict, image_processor, head_origin, bout_recognizer, queue_closed_loop_prediction)
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
        # here
        start_time = time.time()
        closed_loop_class.process_frame(all_frame_mats[i])
        end_time = time.time()
        time.sleep(0.002)

        #print(f"time to process frame {end_time-start_time}")
    closed_loop_class.process_frame(None)


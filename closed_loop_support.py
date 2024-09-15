import logging
from image_processor.ImageProcessor import ImageProcessor
from main_closed_loop import ClosedLoop
from Stimulus import Stimulus
from constants import *

class StimuliGeneratorClosedLoop:
    def __init__(self, canvas, app, closed_loop_pred_queue, output_device=None, camera_queue=None):
        self.stimulus_struct = {
            "exitCriteria": "Time",
            "startX": '90',
            "startY": '450',
            "endX": '90',
            "endY": '450',
            "repetitions": '1',
            "fastSpeed": '0',
            "slowSpeed": '0',
            "startShapeRadius": '4',
            "endShapeRadius": '4',
            "fastDuration": '500',
            "slowDuration": '500',
            "startMode": "AFTER",
            "delay": '0',
            "duration": '500',  # ms
            "xType": "degrees"
        }
        self.camera_queue = camera_queue
        self.closed_loop_pred_queue = closed_loop_pred_queue
        self.output_device = output_device
        self.batchNo = 0
        self.canvas = canvas
        self.app = app
        self.current_stimulus = Stimulus(self.stimulus_struct, canvas, app, 0)
        self.stim_id = 1


    def stop_stimulus(self):
        if self.current_stimulus:
            self.current_stimulus.terminate_run()
            self.current_stimulus.status = DONE  # Set the status to DONE

    def run_stimulus(self):
        # Ensure a stimulus is set and running
        if self.current_stimulus and self.current_stimulus.status == RUNNING:
            self.current_stimulus.move()
            if not self.current_stimulus.trigger_out_sent:
                if self.output_device is not None:
                    self.output_device.give_pulse()
                    self.app.setDebugText("Sent pulse for i={0}".format(i, self.current_stimulus))
                self.current_stimulus.trigger_out_sent = True

    def run_stimuli_closed_loop(self):
        self.run_stimulus()
        try:
            # Check if there is a new stimulus from the queue
            if not self.closed_loop_pred_queue.empty():
                angle, distance = self.closed_loop_pred_queue.get()
                self.modify_stimulus_dict(angle,distance)
                self.stop_stimulus()
                self.current_stimulus = Stimulus(self.stimulus_struct, self.canvas, self.app, self.stim_id)
                self.current_stimulus.init_shape(0)
                self.stim_id += 1
                print(f"New stimulus added with ID: {self.stim_id}")
        except Exception as e:
            print(f"Error processing queue: {e}")


    def modify_stimulus_dict(self, angle, distance):
        self.stimulus_struct["startX"] = self.stimulus_struct["endX"]
        self.stimulus_struct["endX"] = str(angle)
        self.stimulus_struct["startShapeRadius"] = self.stimulus_struct["endShapeRadius"]
        self.stimulus_struct["endShapeRadius"] = str(distance)


def start_closed_loop_background(queue_writer, state, pca_and_predict, bout_recognizer,tail_tracker,min_frame,
                                 mean_frame, head_origin, queue_predictions):
    # Target function for real-time image processing
    logging.info("Closed loop started")
    image_processor = ImageProcessor(False)
    image_processor.calc_masks(min_frame, mean_frame, head_origin)
    closed_loop_class = ClosedLoop(pca_and_predict, image_processor, tail_tracker, bout_recognizer,queue_predictions)
    while state.value == 1:
        if not queue_writer.empty():
            i, image_result = queue_writer.get(timeout=1)  # Fetch from the queue
            closed_loop_class.process_frame(image_result)  # Process the frame
        else:
            pass
            #logging.warning("Queue is empty, no image to process.")

    # Clean-up logic for closed-loop background when state is not RUN
    logging.info("Closed loop background finished")
import logging
import time
from os import times
from multiprocessing import queues
from angle_dist_translater.AngleDistTranslator import AngleDistTranslator
from closed_loop_process.print_time import reset_time, print_time, print_statistics, start_time_logger
from config_files.closed_loop_config import *
from closed_loop_process.main_closed_loop import ClosedLoop
from Stimulus.Stimulus import Stimulus
from constants import *
import copy
import pandas as pd


class StimuliGeneratorClosedLoop:
    def __init__(self, canvas, app, closed_loop_pred_queue, output_device=None, camera_queue=None, calib_mode=False):
        self.start_trial_from_left = False
        self.calib_mode = calib_mode
        self.camera_queue = camera_queue
        self.closed_loop_pred_queue = closed_loop_pred_queue
        self.output_device = output_device
        self.canvas = canvas
        self.app = app
        self.batchNo = 0
        self.stim_id = 1
        self.stimuli_type = "floating"
        self.stimuli_log = pd.DataFrame(
            columns=['TS', 'Event Type', 'End/Start', 'Predicted Angle', 'Predicted Distance',
                     'Current Angle', 'Current Size'])
        self._init_stimulus_structs()
        self.current_stim_struct = copy.deepcopy(self.basic_stimulus_struct) if calib_mode else copy.deepcopy(
            self.stimulus_struct_start_left)
        self.current_stimulus = Stimulus(self.current_stim_struct, canvas, app, 0)
        self.renderer = AngleDistTranslator(int(self.current_stim_struct["startX"]),
                                            int(self.current_stim_struct["startShapeRadius"]))

    def _init_stimulus_structs(self):
        float_duration = self.calc_duration(start_angle, end_angle, stimuli_floating_speed)
        self.basic_stimulus_struct = {
            "exitCriteria": "Time", "startX": str(start_angle), "startY": '450', "endX": '30', "endY": '450',
            "repetitions": '1', "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": '4', "endShapeRadius": '4',
            "fastDuration": '100', "slowDuration": '100', "startMode": "AFTER", "delay": '0', "duration": '100',
            "xType": "degrees"
        }
        self.stimulus_struct_start_left = copy.deepcopy(self.basic_stimulus_struct)
        self.stimulus_struct_start_left.update(
            {"startX": str(start_angle), "endX": str(end_angle), "duration": float_duration})

        self.stimulus_struct_start_right = copy.deepcopy(self.basic_stimulus_struct)
        self.stimulus_struct_start_right.update(
            {"startX": str(end_angle), "endX": str(start_angle), "duration": float_duration})

        self.stimulus_struct_spacer = {
            "exitCriteria": "Spacer", "startX": '0', "startY": '0', "endX": '0', "endY": '0', "repetitions": '1',
            "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": '4', "endShapeRadius": '4', "fastDuration": '0',
            "slowDuration": '0', "startMode": "AFTER", "delay": '0', "duration": str(spacer_duration),
            "xType": "degrees"
        }

    @staticmethod
    def calc_duration(start_angle, end_angle, angular_velocity):
        duration = str(round(abs(end_angle - start_angle) / angular_velocity * 1000))
        return duration

    def stop_stimulus(self):
        if self.current_stimulus:
            self.current_stimulus.terminate_run()
            self.current_stimulus.status = DONE  # Set the status to DONE

    def start_stimulus(self):
        self.current_stimulus.init_shape(0)
        print(f"New {self.stimuli_type} stimulus added with ID: {self.stim_id}")
        self.stim_id += 1

    def _send_pulse_and_write_log(self, event_type, start_end, predicted_angle, predicted_distance):
        """Send a pulse if required for the current stimulus."""
        if not self.current_stimulus.trigger_out_sent and self.output_device:
            self.output_device.give_pulse()
            self.app.setDebugText(f"Sent pulse for i={self.stim_id}, stimulus={self.current_stimulus}")
            self.stimuli_log = self.stimuli_log.append({'TS': pd.Timestamp.now(), 'Event Type': event_type,
                                                        'End/Start': start_end, 'Predicted Angle': predicted_angle,
                                                        'Predicted Distance': predicted_distance,
                                                        'Current Angle': self.renderer.current_angle,
                                                        'Current Size': self.renderer.current_size})
            self.current_stimulus.trigger_out_sent = True

    def run_stimulus(self):
        # Ensure a stimulus is set and running
        if self.current_stimulus and self.current_stimulus.status == RUNNING:
            self.current_stimulus.move()
            # update the angle_dist_translater's current angle and size
            self.renderer.reset_food(int(self.current_stimulus.current_degree), int(self.current_stimulus.currRadius))
        elif not self.calib_mode:  # stimuli is done - need to initiate new stimuli
            self.stop_stimulus()
            if self.stimuli_type == "floating":  # change to spacer
                self._send_pulse_and_write_log("trial","end", "NA","NA")
                self.stimuli_type = "spacer"
                self.start_trial_from_left = not self.start_trial_from_left
                self.current_stim_struct = copy.deepcopy(self.stimulus_struct_spacer)
            elif self.stimuli_type == "moving":  # change to floating from current point in the same direction
                self._send_pulse_and_write_log("movement", "end", "NA", "NA")
                self.stimuli_type = "floating"
                if self.start_trial_from_left:
                    self.current_stim_struct = copy.deepcopy(self.stimulus_struct_start_left)
                else:
                    self.current_stim_struct = copy.deepcopy(self.stimulus_struct_start_right)
                self.current_stim_struct["startX"] = self.renderer.current_angle
                self.current_stim_struct["duration"] = StimuliGeneratorClosedLoop.calc_duration(
                    int(self.current_stim_struct["startX"]), int(self.current_stim_struct["endX"]),
                    stimuli_floating_speed)
            elif self.stimuli_type == "spacer":  # change to floating
                self._send_pulse_and_write_log("trial", "start", "NA", "NA")
                self.stimuli_type = "floating"
                if self.start_trial_from_left:
                    self.current_stim_struct = copy.deepcopy(self.stimulus_struct_start_left)
                else:
                    self.current_stim_struct = copy.deepcopy(self.stimulus_struct_start_right)
            self.current_stimulus = Stimulus(self.current_stim_struct, self.canvas, self.app, self.stim_id)
            self.start_stimulus()

    def run_stimuli_closed_loop(self):
        self.run_stimulus()
        # Check if there is a new stimulus from the queue
        if not self.closed_loop_pred_queue.empty():
            angle, distance = self.closed_loop_pred_queue.get()
            if self.calib_mode:
                self.modify_stimulus_dict(angle, 4)
                self.stop_stimulus()
                self.current_stimulus = Stimulus(self.current_stim_struct, self.canvas, self.app, self.stim_id)
                self.start_stimulus()

            elif self.stimuli_type != "spacer":
                self.stop_stimulus()
                old_angle = self.renderer.current_angle
                old_size = self.renderer.current_size
                res = self.renderer.calc_new_angle_and_size(angle, distance)
                if res is None:  # end of trial - the stimuli is out of range or the hunt is finished
                    # run spacer
                    self.stimuli_type = "spacer"
                    self._send_pulse_and_write_log("trial", "end", str(angle), str(distance))
                    self.current_stimulus = Stimulus(self.stimulus_struct_spacer, self.canvas, self.app, self.stim_id)
                    self.start_trial_from_left = not self.start_trial_from_left
                else:  # update with new movement angle and distance
                    self.stimuli_type = "moving"
                    self._send_pulse_and_write_log("moving", "start", str(angle), str(distance))
                    new_angle, new_size = res
                    self.modify_stimulus_dict(new_angle, new_size, old_angle, old_size)
                    self.current_stimulus = Stimulus(self.current_stim_struct, self.canvas, self.app, self.stim_id)
                self.start_stimulus()

    def modify_stimulus_dict(self, new_angle, new_distance, old_angle=None, old_size=None):
        new_angle = round(new_angle)
        new_distance = round(new_distance)
        if self.calib_mode:

            self.current_stim_struct.update({"startX": self.current_stim_struct["endX"], "endX": str(new_angle),
                                             "startShapeRadius": self.current_stim_struct["endShapeRadius"],
                                             "endShapeRadius": str(new_distance)})
        else:
            self.current_stim_struct.update({"exitCriteria": "Time", "startX": old_angle, "endX": str(new_angle),
                                             "startShapeRadius": old_size, "endShapeRadius": str(new_distance),
                                             "duration": self.calc_duration(int(old_angle), int(new_angle),
                                                                            stimuli_moving_speed)})
def empty_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()  # Non-blocking get
        except Exception as e:
            break


def start_closed_loop_background(queue_writer, state, pca_and_predict, bout_recognizer,tail_tracker,image_processor, queue_predictions):
    import psutil
    p = psutil.Process()  # Get current process
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    closed_loop_class = ClosedLoop(pca_and_predict, image_processor, tail_tracker, bout_recognizer,
                                   queue_predictions)
    print("Closed loop started")
    empty_queue(queue_writer)
    start_time_logger('CLOSED LOOP')
    prev_time = time.perf_counter()
    while state.value == 1:
        reset_time()
        print('after reset')
        images_queue_size = queue_writer.qsize()
        print(images_queue_size)
        if images_queue_size > 1:
            print(f"Warning: images queue size is of size {images_queue_size}")

        print_time('before queue')
        try:
            i, image_result = queue_writer.get_nowait()   # Attempt to get an item without waiting
            print(f"time to image frame = {time.perf_counter() - prev_time}")
            prev_time= time.perf_counter()
        except queues.Empty:
            print("Queue is empty, no item to retrieve.")
            continue
        print_time('after queue')
        closed_loop_class.process_frame(image_result)  # Process the frame
        print_time('after process')

    print_statistics()
    closed_loop_class.process_frame(None)
    # Clean-up logic for closed-loop background when state is not RUN
    logging.info("Closed loop background finished")

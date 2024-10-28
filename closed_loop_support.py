import logging
from renderer.Renderer import Renderer
from closed_loop_config import *
from image_processor.ImageProcessor import ImageProcessor
from main_closed_loop import ClosedLoop
from Stimulus import Stimulus
from constants import *
import copy

class StimuliGeneratorClosedLoop:
    def __init__(self, canvas, app, closed_loop_pred_queue, output_device=None, camera_queue=None, calib_mode = False):
        self.start_trial_from_left = False
        self.calib_mode = calib_mode
        float_duration = StimuliGeneratorClosedLoop.calc_duration(start_angle, end_angle, stimuli_floating_speed)
        self.basic_stimulus_struct = {
            "exitCriteria": "Time",
            "startX": str(start_angle),
            "startY": '450',
            "endX": '30',
            "endY": '450',
            "repetitions": '1',
            "fastSpeed": '0',
            "slowSpeed": '0',
            "startShapeRadius": '4',
            "endShapeRadius": '4',
            "fastDuration": '100',
            "slowDuration": '100',
            "startMode": "AFTER",
            "delay": '0',
            "duration": '100',  # ms
            "xType": "degrees"
        }
        self.stimulus_struct_start_left = {
            "exitCriteria": "Time",
            "startX": str(start_angle),
            "startY": '450',
            "endX": str(end_angle),
            "endY": '450',
            "repetitions": '1',
            "fastSpeed": '0',
            "slowSpeed": '0',
            "startShapeRadius": '4',
            "endShapeRadius": '4',
            "fastDuration": '100',
            "slowDuration": '100',
            "startMode": "AFTER",
            "delay": '0',
            "duration": float_duration,  # ms
            "xType": "degrees"
        }
        self.stimulus_struct_start_right = {
            "exitCriteria": "Time",
            "startX": str(end_angle),
            "startY": '450',
            "endX": str(start_angle),
            "endY": '450',
            "repetitions": '1',
            "fastSpeed": '0',
            "slowSpeed": '0',
            "startShapeRadius": '4',
            "endShapeRadius": '4',
            "fastDuration": '100',
            "slowDuration": '100',
            "startMode": "AFTER",
            "delay": '0',
            "duration": float_duration,  # ms
            "xType": "degrees"
        }
        self.stimulus_struct_spacer = {
            "exitCriteria": "Spacer",
            "startX": '0',
            "startY": '0',
            "endX": '0',
            "endY": '0',
            "repetitions": '1',
            "fastSpeed": '0',
            "slowSpeed": '0',
            "startShapeRadius": '4',
            "endShapeRadius": '4',
            "fastDuration": '0',
            "slowDuration": '0',
            "startMode": "AFTER",
            "delay": '0',
            "duration": '5000',  # ms
            "xType": "degrees"
        }
        self.current_stim_struct = None
        self.camera_queue = camera_queue
        self.closed_loop_pred_queue = closed_loop_pred_queue
        self.output_device = output_device
        self.batchNo = 0
        self.canvas = canvas
        self.app = app
        if calib_mode:
            self.current_stim_struct = copy.deepcopy(self.basic_stimulus_struct)
        else:
            self.current_stim_struct = copy.deepcopy(self.stimulus_struct_start_left)
        self.current_stimulus = Stimulus(self.current_stim_struct, canvas, app, 0)
        self.renderer = Renderer(int(self.current_stim_struct["startX"]), int(self.current_stim_struct["startShapeRadius"]))
        self.stim_id = 1
        self.stimuli_type = "floating"

    @staticmethod
    def calc_duration(start_angle, end_angle, angular_velocity):
        print(f"angle start {start_angle} angle end {end_angle}")
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


    def run_stimulus(self):
        # Ensure a stimulus is set and running
        if self.current_stimulus and self.current_stimulus.status == RUNNING:
            self.current_stimulus.move()
            # update the renderer's current angle and size
            self.renderer.reset_food(int(self.current_stimulus.current_degree), int(self.current_stimulus.currRadius))
            # Todo see when to send the pulse
            if not self.current_stimulus.trigger_out_sent:
                if self.output_device is not None:
                    self.output_device.give_pulse()
                    self.app.setDebugText("Sent pulse for i={0}".format(self.stim_id, self.current_stimulus))
                self.current_stimulus.trigger_out_sent = True
        elif not self.calib_mode: # stimuli is done - need to initiate new stimuli
            self.stop_stimulus()

            if self.stimuli_type == "floating": # change to spacer
                self.stimuli_type = "spacer"
                self.start_trial_from_left = not self.start_trial_from_left
                self.current_stim_struct = copy.deepcopy(self.stimulus_struct_spacer)

            elif self.stimuli_type == "moving": # change to floating from current point in the same direction
                self.stimuli_type = "floating"
                if self.start_trial_from_left:
                    self.current_stim_struct = copy.deepcopy(self.stimulus_struct_start_left)
                else:
                    self.current_stim_struct = copy.deepcopy(self.stimulus_struct_start_right)
                self.current_stim_struct["startX"] = self.renderer.current_angle
                self.current_stim_struct["duration"] = StimuliGeneratorClosedLoop.calc_duration(
                    int(self.current_stim_struct["startX"]),int(self.current_stim_struct["endX"]),stimuli_floating_speed)

            elif self.stimuli_type == "spacer":  # change to floating
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
                res = self.renderer.calc_new_angle_and_size(angle,distance)
                if res is None: # end of trial - the stimuli is out of range or the hunt is finished
                    # run spacer
                    print("trial end because hunt is finished")
                    self.stimuli_type = "spacer"
                    self.current_stimulus = Stimulus(self.stimulus_struct_spacer, self.canvas, self.app, self.stim_id)
                    self.start_trial_from_left = not self.start_trial_from_left
                else: # update with new movement angle and distance
                    self.stimuli_type = "moving"

                    new_angle, new_size = res
                    self.modify_stimulus_dict(new_angle,new_size,old_angle,old_size)
                    self.current_stimulus = Stimulus(self.current_stim_struct, self.canvas, self.app, self.stim_id)
                self.start_stimulus()



    def modify_stimulus_dict(self, new_angle, new_distance, old_angle = None, old_size = None):
        if self.calib_mode:
            self.current_stim_struct["startX"] = self.current_stim_struct["endX"]
            self.current_stim_struct["endX"] = str(new_angle)
            self.current_stim_struct["startShapeRadius"] = self.current_stim_struct["endShapeRadius"]
            self.current_stim_struct["endShapeRadius"] = str(new_distance)

        else:
            self.current_stim_struct["exitCriteria"] = "Time"
            self.current_stim_struct["startX"] = old_angle
            self.current_stim_struct["endX"] = str(new_angle)
            self.current_stim_struct["startShapeRadius"] = old_size
            self.current_stim_struct["endShapeRadius"] = str(new_distance)
            duration = StimuliGeneratorClosedLoop.calc_duration(int(self.current_stim_struct["startX"]), int(self.current_stim_struct["endX"])
                                                                , stimuli_moving_speed)
            print(f"duration{duration}")
            self.current_stim_struct["duration"] = duration


def start_closed_loop_background(queue_writer, state, pca_and_predict, bout_recognizer, min_frame,
                                 mean_frame, head_origin, tail_tip, queue_predictions):
    # Target function for real-time image processing
    logging.info("Closed loop started")
    image_processor = ImageProcessor(False)
    image_processor.calc_masks(min_frame, mean_frame, head_origin, number_of_frames_calibration)
    closed_loop_class = ClosedLoop(pca_and_predict, image_processor, head_origin, tail_tip, bout_recognizer,queue_predictions)
    while state.value == 1:
        if not queue_writer.empty():
            i, image_result = queue_writer.get(timeout=1)  # Fetch from the queue
            closed_loop_class.process_frame(image_result)  # Process the frame
        else:
            pass
            #logging.warning("Queue is empty, no image to process.")
    closed_loop_class.process_frame(None)
    # Clean-up logic for closed-loop background when state is not RUN
    logging.info("Closed loop background finished")
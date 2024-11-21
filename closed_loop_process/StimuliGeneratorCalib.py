import logging
import time
from multiprocessing import queues
from StimulusMemory.StimulusMemory import StimulusMemory
from closed_loop_process.print_time import reset_time, print_time, print_statistics, start_time_logger
from config_files.closed_loop_config import *
from closed_loop_process.main_closed_loop import ClosedLoop
from Stimulus.Stimulus import Stimulus
from constants import *
import copy
import pandas as pd
import datetime

class StimuliGeneratorCalib:
    def __init__(self, canvas, app):
        self.start_trial_from_left = False
        self.canvas = canvas
        self.app = app
        self.stim_id = 1
        self.stimuli_type = "start_left"
        self.current_stim_struct = copy.deepcopy(self.calib_struct_left)
        self.current_stimulus = Stimulus(self.current_stim_struct, canvas, app, 0)



    def _init_stimulus_structs(self):
        calib_duration = self.calc_duration(start_angle, end_angle, calibration_stimuli_speed)
        self.basic_stimulus_struct = {
            "exitCriteria": "Time", "startX": str(start_angle), "startY": '450', "endX": str(start_angle), "endY": '450',
            "repetitions": '1', "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": '4', "endShapeRadius": '4',
            "fastDuration": '100', "slowDuration": '100', "startMode": "AFTER", "delay": '0', "duration": '3000',
            "xType": "degrees"
        }
        self.calib_struct_left = copy.deepcopy(self.basic_stimulus_struct)
        self.calib_struct_left.update(
            {"startX": str(start_angle), "endX": str(end_angle), "duration": calib_duration})
        self.calib_struct_right = copy.deepcopy(self.basic_stimulus_struct)
        self.calib_struct_right.update(
            {"startX": str(end_angle), "endX": str(start_angle), "duration": calib_duration})


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

    def terminate_run(self):
        self.current_stimulus.terminate_run()


    def run_stimulus(self):
        # Ensure a stimulus is set and running
        if self.current_stimulus and self.current_stimulus.status == RUNNING:
            self.current_stimulus.move()
            # update the StimulusMemory's current angle and size
        else:
            self.stop_stimulus()
            if self.stimuli_type == "start_left":
                self.current_stim_struct = copy.deepcopy(self.calib_struct_right)
                self.stimuli_type = "start_right"
            else:
                self.current_stim_struct = copy.deepcopy(self.calib_struct_left)
                self.stimuli_type = "start_left"
            self.current_stimulus = Stimulus(self.current_stim_struct, self.canvas, self.app, self.stim_id)
            self.start_stimulus()

    def run_stimuli_calib(self):
        self.run_stimulus()


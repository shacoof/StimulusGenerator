import logging
import time
from multiprocessing import queues
from StimulusMemory.StimulusMemory import StimulusMemory
from closed_loop_process.print_time import reset_time, print_time, print_statistics, start_time_logger
from config_files.closed_loop_config import *
from closed_loop_process.main_closed_loop import ClosedLoop
from constants import *
import pandas as pd
import datetime
from Stimulus.Stimulus import Stimulus
from utils.utils import loadCSV


class StimuliGeneratorClosedLoop:
    def __init__(self, canvas, app, closed_loop_pred_queue=None, output_device=None):
        self.closed_loop_pred_queue = closed_loop_pred_queue
        self.output_device = output_device
        self.canvas = canvas
        self.app = app
        self.stim_id = 0
        self.pulseID = 0
        self.csv = loadCSV(STIMULUS_CONFIG)
        self.batches = dict()
        self.batchIndex = 0
        self._init_batches()
        self.batchKeys = sorted(self.batches.keys())
        self.numBatches = len(self.batches.keys())
        self.batchStimulusObjList = self.batches[self.batchKeys[self.batchIndex]]
        self.print_stimulus_list()
        self.number_of_done_stimulus = 0
        self.number_of_stimuli_in_batch = len(self.batchStimulusObjList)
        self.spacerMode = False
        self.spacer = self.init_spacer()
        self.finished_all_reps = False
        self.stimuli_log = pd.DataFrame(
            columns=['Pulse ID','Batch Number', 'TS', 'Event Type', 'End/Start','Reason', 'Stimulus ID', 'Stimulus Status', 'Predicted Angle',
                     'Predicted Distance',
                     'Current Angle', 'Current Size'])


    def init_spacer(self):
        spacer_struct = {
            "exitCriteria": "Spacer", "startX": '0', "startY": '0', "endX": '0', "endY": '0', "repetitions": '1',
            "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": '4', "endShapeRadius": '4', "fastDuration": '0',
            "slowDuration": '0', "startMode": "WITH", "delay": '0', "duration": spacer_time_between_batches,
            "xType": "degrees"}
        return Stimulus(spacer_struct, self.canvas, self.app, self.stim_id)

    def print_stimulus_list(self):
        for i in self.batchStimulusObjList:
            print(vars(i))

    def terminate_run(self):
        for st in self.batchStimulusObjList:
            st.stop_stimulus()

    def end_of_batch(self, reason):
        self.send_pulse_and_write_log("trial", "end", "NA", "NA", reason)
        self.batchIndex += 1
        self.terminate_run()

        if self.batchIndex < self.numBatches:
            self.spacer.init_shape(0)
            self.spacerMode = True
            self.batchStimulusObjList = self.batches[self.batchKeys[self.batchIndex]]
            self.number_of_done_stimulus = 0
            self.number_of_stimuli_in_batch = len(self.batchStimulusObjList)
        else:
            self.finished_all_reps = True

    def _init_batches(self):
        for st in self.csv:
            self.stim_id += 1
            batch_num = int(st["batchNum"])
            if batch_num in self.batches:
                self.batches[batch_num].append(StimulusMemory(self.canvas, self.app, st,self.stim_id,self))
            else:
                self.batches[batch_num] = [StimulusMemory(self.canvas, self.app, st, self.stim_id,self)]

    def save_csv(self, path):
        self.stimuli_log.to_csv(path + '\stimuli_log.csv', index=False)



    def send_pulse_and_write_log(self, event_type, start_end, predicted_angle, predicted_distance, reason):
        """Send a pulse if required for the current stimulus."""
        for st in self.batchStimulusObjList:
            self.stimuli_log = self.stimuli_log.append(
                {
                    'Pulse ID': self.pulseID,
                    'Batch Number': self.batchIndex,
                    'TS': datetime.datetime.now().strftime("%H:%M:%S:%f"),
                    'Event Type': event_type,
                    'End/Start': start_end,
                    'Reason': reason,
                    'Stimulus ID': st.stim_id,
                    'Stimulus Status': st.state,
                    'Predicted Angle': predicted_angle,
                    'Predicted Distance': predicted_distance,
                    'Current Angle': st.current_angle,
                    'Current Size': st.current_size},
                ignore_index=True
            )
        if self.output_device:
            self.app.setDebugText(f"Sent pulse for {event_type} {start_end}")
            self.output_device.give_pulse()
        self.pulseID += 1

    def run_stimulus(self):
        moving_stop = False
        for i, stimulus in enumerate(self.batchStimulusObjList):
            res = stimulus.update()
            if res != "":
                print(res)
            if res == "DONE: too big":
                self.end_of_batch("reached min distance")
                return
            if res != "" and res != "moving stop":
                self.number_of_done_stimulus += 1
            if res == "moving stop":
                moving_stop = True
        if moving_stop:
            self.send_pulse_and_write_log("movement", "end", "NA", "NA","NA")
        if self.number_of_done_stimulus >= self.number_of_stimuli_in_batch:
            self.end_of_batch("all stimulus out of range")

    def all_stimuli_inactive(self):
        for stim in self.batchStimulusObjList:
            if stim.state != "inactive":
                return False
        return True


    def run_stimuli_closed_loop(self):
        if self.finished_all_reps:
            return False
        if self.spacerMode:
            if self.spacer.status == RUNNING:
                self.spacer.move()
            else:
                self.spacerMode = False
                self.spacer.terminate_run()
                self.send_pulse_and_write_log("trial", "start", "NA", "NA", "NA")
        else:
            self.run_stimulus()
            # Check if there is a new movement from the queue
            if self.closed_loop_pred_queue and not self.closed_loop_pred_queue.empty():
                angle, distance = self.closed_loop_pred_queue.get()
                self.send_pulse_and_write_log("movement", "start", angle, distance,"NA")
                if self.all_stimuli_inactive():
                    return True
                for stimulus in self.batchStimulusObjList:
                    stimulus.move(angle, distance)
        return True


def empty_queue(queue):
    while not queue.empty():
        try:
            queue.get()
        except Exception as e:
            break


def start_closed_loop_background(queue_writer, state, pca_and_predict, bout_recognizer, tail_tracker, image_processor,
                                 queue_predictions):
    import psutil
    j = 0
    p = psutil.Process()  # Get current process
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    closed_loop_class = ClosedLoop(pca_and_predict, image_processor, tail_tracker, bout_recognizer,
                                   queue_predictions)
    print("Closed loop started")
    empty_queue(queue_writer)
    start_time_logger('CLOSED LOOP')
    while state.value == 1:
        reset_time()
        images_queue_size = queue_writer.qsize()
        if images_queue_size > 1:
            #print(f"Warning: images queue size is of size {images_queue_size}")
            pass

        print_time('before queue')
        try:
            i, image_result = queue_writer.get()
        except queues.Empty:
            print("Queue is empty, no item to retrieve.")
            continue
        print_time('after queue')
        if j % take_every_x_frame == 0:
            closed_loop_class.process_frame(image_result)  # Process the frame
            print_time('after process')
        j += 1

    print_statistics()
    closed_loop_class.process_frame(None)
    logging.info("Closed loop background finished")

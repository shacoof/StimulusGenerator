import logging
from image_processor.ImageProcessor import ImageProcessor
from main_closed_loop import ClosedLoop


class StimuliGeneratorClosedLoop:
    def __init__(self, canvas, app, output_device=None, queue=None):
        self.stimulus_struct = {
            "exitCriteria": "Time",
            "startX": 90,
            "startY": 600,
            "endX": 90,
            "endY": 600,
            "repetitions": 0,
            "fastSpeed": 0,
            "slowSpeed": 0,
            "startShapeRadius": 4,
            "endShapeRadius": 4,
            "fastDuration": 50,
            "slowDuration": 50,
            "startMode": "WITH",
            "delay": 0,
            "duration": 100,  # ms
            "xType": "degrees"
        }
        self.queue = queue
        self.output_device = output_device
        self.batchNo = 0
        self.canvas = canvas
        self.app = app
        self.stimulusObjList = []
        stim_id = 0
        # add initial stimulus
        self.stimulusObjList.append(Stimulus(self.stimulus_struct, canvas, app, stim_id))
        self.stim_id = 1
        self.print_stimulus_list()

    def terminate_run(self):
        for i in self.stimulusObjList:
            i.terminate_run()

    def print_stimulus_list(self):
        for i in self.stimulusObjList:
            print(vars(i))

    def get_stimulus_state(self, stimulus):
        return stimulus.batchNo, stimulus.status, stimulus.startMode

    def run_stimuli(self):
        while(not self.queue.Empty):
            # add to stimulus object list
            angle, distance = self.queue.pop()
            self.modify_stimulus_dict(angle,distance)
            self.stimulusObjList.append(Stimulus(self.stimulus_struct, canvas, app, stim_id))

        # last stimulus was completed
        i = 0
        s = ""
        # if we got here *then* the object in stimulusObjListLoc is not done
        # we will loop on all the running shapes and progress them
        while i < len(self.stimulusObjList) and self.stimulusObjList[i].status == RUNNING:
            self.stimulusObjList[i].move()
            if not self.stimulusObjList[i].trigger_out_sent:
                if self.output_device is not None:
                    self.output_device.give_pulse()
                    self.app.setDebugText("Sent pulse for i={0}".format(i, self.stimulusObjList[i]))
                self.stimulusObjList[i].trigger_out_sent = True
            s += str(self.stimulusObjList[i]) + "\n"
            i += 1
        return RUNNING

    def modify_stimulus_dict(self, angle, distance):
        self.stimulus_struct["startX"] = angle
        self.stimulus_struct["endShapeRadius"] = distance


def start_closed_loop_background(queue_writer, state, pca_and_predict, bout_recognizer,tail_tracker,min_frame,
                                 mean_frame, head_origin, queue_predictions):
    # Target function for real-time image processing
    logging.info("Closed loop started")
    image_processor = ImageProcessor(False)
    image_processor.calc_masks(min_frame, mean_frame, head_origin)
    closed_loop_class = ClosedLoop(pca_and_predict, image_processor, tail_tracker, bout_recognizer,queue_predictions)
    while state.value == 1:
        try:
            i, image_result = queue_writer.get(timeout=1)  # Fetch from the queue
            closed_loop_class.process_frame(image_result)  # Process the frame

            print("hi")
        except queue.Empty:
            logging.warning("Queue is empty, no image to process.")

    # Clean-up logic for closed-loop background when state is not RUN
    logging.info("Closed loop background finished")
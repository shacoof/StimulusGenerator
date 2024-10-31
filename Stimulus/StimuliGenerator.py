from utils.utils import loadCSV
from Stimulus.Stimulus import *
from constants import *



class StimulusGenerator:

    def __init__(self, canvas, app, output_device=None, queue=None):
        self.queue = queue
        self.output_device = output_device
        self.batchNo = 0
        self.canvas = canvas
        self.app = app
        self.stimulusObjList = []
        stim_id = 0
        for st in loadCSV(STIMULUS_CONFIG):
            stim_id += 1
            self.stimulusObjList.append(Stimulus(st, canvas, app, stim_id))
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
        # last stimulus was completed
        found = False
        i = 0

        # skip stimulus that are done 
        while i < len(self.stimulusObjList) and self.stimulusObjList[i].status == DONE:
            i += 1

        # if we reached the end we are done with this run 
        if i >= len(self.stimulusObjList):
            if self.queue:
                self.queue.put('exit')
                print('EXIT SENT ')
            return DONE

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
            found = True
            s += str(self.stimulusObjList[i]) + "\n"
            i += 1

        if found:
            pass
            # self.app.setDebugText(s)
        else:
            self.app.setDebugText("")

        # if there are no running shapes, i.e. no batch in progress we need to calculate next batch
        if not found and i < len(self.stimulusObjList):
            self.batchNo += 1
            self.stimulusObjList[i].init_shape(self.batchNo)
            if self.app.camera_control.lower() == "on":
                self.queue.put(i)
            i += 1
            # adding all subsequent WITH stimulus to the current one
            while i < len(self.stimulusObjList) and self.stimulusObjList[i].startMode.lower() == WITH:
                self.stimulusObjList[i].init_shape(self.batchNo)
                i += 1

        return RUNNING

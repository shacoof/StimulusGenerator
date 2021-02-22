from utils import loadCSV,writeCSV,sendF9Marker
import logging
from math import trunc
from Stimulus import *
class StimulusGenerator:
    STIMULUS_CONFIG = "StimulusConfig.csv"
    AFTER = "after"
    WITH = "with"

    def __init__ (self,canvas,app):
        self.batchNo = 0
        self.canvas = canvas
        self.app = app
        self.stimulusObjList = []
        for st in loadCSV(self.STIMULUS_CONFIG):            
            self.stimulusObjList.append(Stimulus(st,canvas,app))
        self.printStimulusList()

    def terminateRun(self):
        for i in self.stimulusObjList:
            i.terminateRun()
    
    def printStimulusList(self):
        for i in self.stimulusObjList:
            print(vars(i))

    def getStimulusState(self,stimulus):
        return (stimulus.batchNo,stimulus.status, stimulus.startMode)

    def runStimuli(self):
        #last stimulus was completed 
        found = False
        i = 0

        # skip stimulus that are done 
        while i < len(self.stimulusObjList) and self.stimulusObjList[i].status == Stimulus.DONE:
            i +=1

        # if we reached the end we are done with this run 
        if i >= len(self.stimulusObjList):
            return Stimulus.DONE

        # if we got here *then* the object in stimulusObjListLoc is not done
        # we will loop on all the running shapes and progress them
        while i < len(self.stimulusObjList) and self.stimulusObjList[i].status == Stimulus.RUNNING :
            self.stimulusObjList[i].move()
            i +=1
            found = True 

        # if there are no running shapes, i.e. no batch in progress we need to calculate next batch
        if not found and i < len(self.stimulusObjList):
            self.batchNo += 1 
            self.stimulusObjList[i].initShape(self.batchNo)
            i +=1             
            #adding all subsequent WITH stimulus to the current one 
            while i < len(self.stimulusObjList) and self.stimulusObjList[i].startMode.lower() == self.WITH:
                self.stimulusObjList[i].initShape(self.batchNo)
                i +=1             
            


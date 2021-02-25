from utils import loadCSV,writeCSV,sendF9Marker
import logging
from math import trunc
from Stimulus import *
class StimulusGenerator:

    def __init__ (self,canvas,app):
        self.batchNo = 0
        self.canvas = canvas
        self.app = app
        self.stimulusObjList = []
        id=0
        for st in loadCSV(constants.STIMULUS_CONFIG):            
            id+=1
            self.stimulusObjList.append(Stimulus(st,canvas,app,id))
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
        while i < len(self.stimulusObjList) and self.stimulusObjList[i].status == constants.DONE:
            i +=1

        # if we reached the end we are done with this run 
        if i >= len(self.stimulusObjList):
            return constants.DONE

        s=""
        # if we got here *then* the object in stimulusObjListLoc is not done
        # we will loop on all the running shapes and progress them
        while i < len(self.stimulusObjList) and self.stimulusObjList[i].status == constants.RUNNING :
            self.stimulusObjList[i].move()            
            found = True 
            s += str(self.stimulusObjList[i])+"\n"
            i +=1

        if found:
            self.app.setLabelText(s)
        else: 
            self.app.setLabelText("")

        # if there are no running shapes, i.e. no batch in progress we need to calculate next batch
        if not found and i < len(self.stimulusObjList):
            self.batchNo += 1 
            self.stimulusObjList[i].initShape(self.batchNo)
            i +=1             
            #adding all subsequent WITH stimulus to the current one 
            while i < len(self.stimulusObjList) and self.stimulusObjList[i].startMode.lower() == constants.WITH:
                self.stimulusObjList[i].initShape(self.batchNo)
                i +=1             
            


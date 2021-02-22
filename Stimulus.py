from utils import loadCSV,writeCSV,sendF9Marker
import logging
from math import trunc
import constants

class Stimulus:

    def __init__(self, stimulus,canvas,app):
        
        self.shape          = -1
        self.batchNo        = constants.EMPTY
        self.status         = constants.WAITING
        self.speedMode      = constants.FAST
        self.app             = app
        self.canvas          = canvas
        self.stXStart        = int(stimulus["startX"])*app.vsWidth/constants.VIRTUAL_SCREEN_LOGICAL_WIDTH
        self.stYStart        = int(stimulus["startY"])*app.vsHeight/constants.VIRTUAL_SCREEN_LOGICAL_HEIGHT
        self.stXEnd          = int(stimulus["endX"])*app.vsWidth/constants.VIRTUAL_SCREEN_LOGICAL_WIDTH
        self.stYEnd          = int(stimulus["endY"])*app.vsHeight/constants.VIRTUAL_SCREEN_LOGICAL_HEIGHT
        self.fastSpeed       = int(stimulus["fastSpeed"])
        self.slowSpeed       = int(stimulus["slowSpeed"])
        self.startShapeRadius = int(stimulus["startShapeRadius"])
        self.endShapeRadius  = int(stimulus["endShapeRadius"])
        self.fastDuration    = int(stimulus["fastDuration"])
        self.slowDuration    = int(stimulus["slowDuration"])
        self.startMode       = stimulus["startMode"]
        self.delay           = int(stimulus["delay"])
        self.repetitions    = int(stimulus["repetitions"])
        self.shapeX         = 0
        self.shapeY         = 0 

        self.currRadius         = self.startShapeRadius             
        self.speed              = self.fastSpeed
        self.timeInSpeed        = 0 
        self.sleepTime          = 0


        # since x,y is between 0-999 but actual width,height are different we need to normalize steps
        self.xNorm = self.app.vsWidth/constants.VIRTUAL_SCREEN_LOGICAL_WIDTH
        self.yNorm = self.app.vsHeight/constants.VIRTUAL_SCREEN_LOGICAL_HEIGHT

        self.stXOrientation = constants.LEFT_RIGHT
        self.stYOrientation = constants.TOP_DOWN   

        if self.stXStart > self.stXEnd:
            self.xNorm=-self.xNorm
            self.stXOrientation = constants.RIGHT_LEFT
        elif self.stXEnd == self.stXStart:
            self.xNorm = 0

        if self.stYStart > self.stYEnd:
            self.yNorm=-self.yNorm
            self.stYOrientation = constants.DOWN_TOP
        elif self.stYEnd == self.stYStart:
            self.yNorm=0

        self.radiuseNorm = (self.endShapeRadius-self.startShapeRadius)/constants.VIRTUAL_SCREEN_LOGICAL_HEIGHT

        self.repNo = 0
        self.speed = self.fastSpeed
        self.timeInSpeed = 0 
        self.sleepTime = 0

    def _create_circle(self,canvas, x, y, r, **kwargs):
        return canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

    def initShape(self,batchNo):
        self.batchNo = batchNo
        self.status = constants.RUNNING

        self.shapeX = self.app.vsX + self.stXStart
        self.shapeY = self.app.vsY + self.stYStart
        self.currRadius = self.startShapeRadius

        if  self.app.f9CommunicationEnabled:
            logging.info("sending f9 communication")
            sendF9Marker()

        self.shape = self._create_circle(self.canvas,
                                trunc(self.shapeX),
                                trunc(self.shapeY),
                                trunc(self.startShapeRadius),
                                fill = self.app.stimulusColor)

        logging.info("Shape created")

    def move(self):
        """
            every cycle is 1 milisecond
            every cycle the program wakes up and decide what to do 
                - if sleepTime == speed then move otherwise slepp for another milisecond
                - if timeInSpeed == fastDuration then chagne to slow mode (and vice versa)
                    - timeInSpeed = 0
                    - speed = fast/slow speed (as needed)

            speedMode = FAST or SLOW

        """
        x0, y0, x1, y1 = self.canvas.coords(self.shape)
        #is it time to move the shape
        if self.sleepTime == self.speed:
            self.canvas.delete(self.shape)
            self.currRadius += self.radiuseNorm
            self.shapeX += self.xNorm
            self.shapeY += self.yNorm            
            self.shape = self._create_circle(self.canvas,
                                trunc(self.shapeX),
                                trunc(self.shapeY),
                                trunc(self.currRadius),
                                fill = self.app.stimulusColor)
            #self.canvas.move(self.shape,self.xNorm,self.yNorm)
            self.sleepTime = 0
        else:
            self.sleepTime +=1

        # is it time to change speed 
        if self.speedMode==constants.FAST and self.timeInSpeed == self.fastDuration:
            self.timeInSpeed = 0
            self.speed       = self.slowSpeed
            self.speedMode   = constants.SLOW
            self.sleepTime   = 0
        elif self.speedMode==constants.SLOW and self.timeInSpeed == self.slowDuration:
            self.timeInSpeed = 0
            self.speed       = self.fastSpeed
            self.speedMode   = constants.FAST
            self.sleepTime   = 0
        else :
            self.timeInSpeed +=1

        
        #logging.info("moving shape to new location x="+str(x0)+" y="+str(y0))
        # This stimulus repitiion reached its end 
        if  ((self.stXOrientation==constants.LEFT_RIGHT and x1 > self.app.vsX + self.stXEnd+constants.SPACE_BUFFER) or
            (self.stXOrientation==constants.RIGHT_LEFT and x1 < self.app.vsX + self.stXEnd+constants.SPACE_BUFFER) or 
            (self.stYOrientation==constants.TOP_DOWN and y1 > self.app.vsY + self.stYEnd+constants.SPACE_BUFFER) or
            (self.stYOrientation==constants.DOWN_TOP and y1 < self.app.vsY + self.stYEnd+constants.SPACE_BUFFER)): 
            logging.info("repetition completed !")
            self.repNo += 1
            self.canvas.delete(self.shape)        
            # we finished all repitiions for this stimulus and we need to move to next stimulus
            if self.repNo >= self.repetitions:            
                self.status = constants.DONE
                repNo = 0
            else:
                logging.info("Starting repetition no="+str(self.repNo+1))
                self.initShape(self.batchNo) #creating the shape for the next repitition 

    def terminateRun(self):
        self.canvas.delete(self.shape)
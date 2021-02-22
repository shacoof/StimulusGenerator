from utils import loadCSV,writeCSV,sendF9Marker
import logging
from math import trunc

class Stimulus:
    LEFT_RIGHT  = 1
    RIGHT_LEFT  = -1
    TOP_DOWN    = 1 
    DOWN_TOP    = -1
    WAITING     = 0 
    RUNNING     = 1
    DONE        = 2
    EMPTY       = -1 
    FAST = 1
    SLOW = 2

    def __init__(self, stimulus,canvas,app):
        self.shape          = -1
        self.batchNo        = self.EMPTY
        self.status         = self.WAITING
        self.speedMode      = self.FAST
        self.app             = app
        self.canvas          = canvas
        self.stXStart        = int(stimulus["startX"])*app.vsWidth/1000
        self.stYStart        = int(stimulus["startY"])*app.vsHeight/1000
        self.stXEnd          = int(stimulus["endX"])*app.vsWidth/1000
        self.stYEnd          = int(stimulus["endY"])*app.vsHeight/1000
        self.fastSpeed       = int(stimulus["fastSpeed"])
        self.startShapeRadius = int(stimulus["startShapeRadius"])
        self.endShapeRadius  = int(stimulus["endShapeRadius"])
        self.slowSpeed       = int(stimulus["slowSpeed"])
        self.fastDuration    = int(stimulus["fastDuration"])
        self.slowDuration    = int(stimulus["slowDuration"])
        self.startMode       = stimulus["startMode"]
        self.delay           = int(stimulus["delay"])
        self.repetitions    = int(stimulus["repetitions"])
             
        self.speed              = self.fastSpeed
        self.timeInSpeed        = 0 
        self.sleepTime          = 0


        # since x,y is between 0-999 but actual width,height are different we need to normalize steps
        self.xNorm = self.app.vsWidth/1000
        self.yNorm = self.app.vsHeight/1000

        self.stXOrientation = self.LEFT_RIGHT
        self.stYOrientation = self.TOP_DOWN   

        if self.stXStart > self.stXEnd:
            self.xNorm=-self.xNorm
            self.stXOrientation = self.RIGHT_LEFT
        elif self.stXEnd == self.stXStart:
            self.xNorm = 0

        if self.stYStart > self.stYEnd:
            self.yNorm=-self.yNorm
            self.stYOrientation = self.DOWN_TOP
        elif self.stYEnd == self.stYStart:
            self.yNorm=0

        self.repNo = 0
        self.speed = self.fastSpeed
        self.timeInSpeed = 0 
        self.sleepTime = 0

    def _create_circle(self,canvas, x, y, r, **kwargs):
        return canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

    def initShape(self,batchNo):
        self.batchNo = batchNo
        self.status = self.RUNNING

        cx0 = self.app.vsX + self.stXStart
        cy0 = self.app.vsY + self.stYStart

        if  self.app.f9CommunicationEnabled:
            logging.info("sending f9 communication")
            sendF9Marker()

        self.shape = self._create_circle(self.canvas,
                                trunc(cx0),
                                trunc(cy0),
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
        #is it time to move the shape
        if self.sleepTime == self.speed:
            self.canvas.move(self.shape,self.xNorm,self.yNorm)
            self.sleepTime = 0
        else:
            self.sleepTime +=1

        # is it time to change speed 
        if self.speedMode==self.FAST and self.timeInSpeed == self.fastDuration:
            self.timeInSpeed = 0
            self.speed       = self.slowSpeed
            self.speedMode   = self.SLOW
            self.sleepTime   = 0
        elif self.speedMode==self.SLOW and self.timeInSpeed == self.slowDuration:
            self.timeInSpeed = 0
            self.speed       = self.fastSpeed
            self.speedMode   = self.FAST
            self.sleepTime   = 0
        else :
            self.timeInSpeed +=1

        x0, y0, x1, y1 = self.canvas.coords(self.shape)
        #logging.info("moving shape to new location x="+str(x0)+" y="+str(y0))
        # This stimulus repitiion reached its end 
        if  ((self.stXOrientation==self.LEFT_RIGHT and x1 > self.app.vsX + self.stXEnd+20) or
            (self.stXOrientation==self.RIGHT_LEFT and x1 < self.app.vsX + self.stXEnd+20) or 
            (self.stYOrientation==self.TOP_DOWN and y1 > self.app.vsY + self.stYEnd+20) or
            (self.stYOrientation==self.DOWN_TOP and y1 < self.app.vsY + self.stYEnd+20)): 
            logging.info("repetition completed !")
            self.repNo += 1
            self.canvas.delete(self.shape)        
            # we finished all repitiions for this stimulus and we need to move to next stimulus
            if self.repNo >= self.repetitions:            
                self.status = self.DONE
                repNo = 0
            else:
                logging.info("Starting repetition no="+str(self.repNo+1))
                self.initShape(self.batchNo) #creating the shape for the next repitition 

    def terminateRun(self):
        self.canvas.delete(self.shape)



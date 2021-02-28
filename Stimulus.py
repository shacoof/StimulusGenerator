from utils import loadCSV,writeCSV,sendF9Marker
import logging
from math import trunc
import constants

class Stimulus:

    def __init__(self, stimulus,canvas,app,stimulusID):
        
        self.stimulusID     = stimulusID
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
        self.startShapeRadius = self.app.convertDegreestoPixels(int(stimulus["startShapeRadius"]),"width")
        self.endShapeRadius  = self.app.convertDegreestoPixels(int(stimulus["endShapeRadius"]),"width")
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
        self.delaySoFar          = 0
        self.repNo              = 0

        self.PixelsPerMSFastX    = self.app.convertDegreestoPixels(self.fastSpeed,"X")/(1000/constants.SLEEP_TIME) #dividing by 1000 to convert to ms
        self.PixelsPerMSSlowX    = self.app.convertDegreestoPixels(self.slowSpeed,"X")/(1000/constants.SLEEP_TIME) #dividing by 1000 to convert to ms
        self.PixelsPerMSFastY    = self.app.convertDegreestoPixels(self.fastSpeed,"Y")/(1000/constants.SLEEP_TIME) #dividing by 1000 to convert to ms
        self.PixelsPerMSSlowY    = self.app.convertDegreestoPixels(self.slowSpeed,"Y")/(1000/constants.SLEEP_TIME) #dividing by 1000 to convert to ms


        self.stXOrientation = constants.LEFT_RIGHT
        self.stYOrientation = constants.TOP_DOWN   

        if self.stXStart > self.stXEnd:
            self.stXOrientation = constants.RIGHT_LEFT
            self.PixelsPerMSFastX *= -1
            self.PixelsPerMSSlowX *= -1
        elif self.stXEnd == self.stXStart:
            self.PixelsPerMSFastX = 0
            self.PixelsPerMSSlowX = 0
        
        if self.stYStart > self.stYEnd:
            self.PixelsPerMSFastY *= -1
            self.PixelsPerMSSlowY *= -1
            self.stYOrientation = constants.DOWN_TOP
        elif self.stYEnd == self.stYStart:
            self.PixelsPerMSFastY = 0
            self.PixelsPerMSSlowY = 0

        self.xChange = self.PixelsPerMSFastX
        self.yChange = self.PixelsPerMSFastY
        self.radiuseNorm = (self.endShapeRadius-self.startShapeRadius)/constants.VIRTUAL_SCREEN_LOGICAL_HEIGHT

    def initShape(self,batchNo):
        self.batchNo = batchNo
        self.status = constants.RUNNING

        self.shapeX = self.app.vsX + self.stXStart
        self.shapeY = self.app.vsY + self.stYStart
        self.currRadius = self.startShapeRadius

        if  self.app.f9CommunicationEnabled:
            logging.info("sending f9 communication")
            sendF9Marker()

        self.shape = self.canvas.create_oval(trunc(self.shapeX),
                                            trunc(self.shapeY),
                                            trunc(self.shapeX+self.startShapeRadius),
                                            trunc(self.shapeY+self.startShapeRadius),
                                            fill = self.app.stimulusColor, width=20,outline='')

        self.canvas.itemconfigure(self.shape, state='hidden')
        logging.info("Shape created")

    def move(self):
        """
            every cycle is 1 milisecond
            every cycle we add add 
                - if sleepTime == speed then move otherwise slepp for another milisecond
                - if timeInSpeed == fastDuration then chagne to slow mode (and vice versa)
                    - timeInSpeed = 0
                    - speed = fast/slow speed (as needed)

            speedMode = FAST or SLOW

        """
        
        self.delaySoFar +=1*constants.SLEEP_TIME
        if self.delaySoFar < self.delay:
            return
        self.canvas.itemconfigure(self.shape, state='normal')
        x0, y0, x1, y1 = self.canvas.coords(self.shape)
        self.shapeX += self.xChange
        self.shapeY += self.yChange
        self.currRadius += self.radiuseNorm
        #is it time to move the shape
        if  (trunc(self.shapeX) != x0 or
             trunc(self.shapeY) != y0 or 
             self.radiuseNorm != 0):
            #logging.debug(f"move shape = {self.stimulusID} ")
            self.canvas.coords(self.shape,
                               self.shapeX,
                               self.shapeY,
                               self.shapeX+self.currRadius,
                               self.shapeY+self.currRadius)


        # is it time to change speed 
        if self.speedMode==constants.FAST and self.timeInSpeed == self.fastDuration:
            self.timeInSpeed = 0
            self.speedMode   = constants.SLOW
            self.xChange = self.PixelsPerMSSlowX
            self.yChange = self.PixelsPerMSSlowY
        elif self.speedMode==constants.SLOW and self.timeInSpeed == self.slowDuration:
            self.timeInSpeed = 0
            self.speedMode   = constants.FAST
            self.xChange = self.PixelsPerMSFastX
            self.yChange = self.PixelsPerMSFastY
        else :
            self.timeInSpeed +=1*constants.SLEEP_TIME

        
        #logging.debug("moving shape to new location x="+str(x0)+" y="+str(y0))
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

    def __str__(self):
        return f'id = ({self.stimulusID} (x0,x1,y0,y1) {self.stXStart} {self.stXEnd} {self.stYStart} {self.stYEnd}(pixelsPerMS WxH) {round(self.PixelsPerMSFastX,1)}X{round(self.PixelsPerMSFastY,1)}'
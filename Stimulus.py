from utils import sendF9Marker
import logging
from math import trunc,sin,radians
from constants import *

class Stimulus:

    def __init__(self, stimulus, canvas, app, stimulus_id):
        
        self.stimulusID     = stimulus_id
        self.shape          = -1
        self.batchNo        = EMPTY
        self.status         = WAITING
        self.speedMode      = FAST
        self.app             = app
        self.canvas          = canvas
        # fast speed is in degrees per second
        self.fastSpeed       = int(stimulus["fastSpeed"])
        self.slowSpeed       = int(stimulus["slowSpeed"])
        self.startShapeRadius = self.app.convertDegreestoPixels(int(stimulus["startShapeRadius"]),"width")
        self.endShapeRadius  = self.app.convertDegreestoPixels(int(stimulus["endShapeRadius"]),"width")
        self.fastDuration    = int(stimulus["fastDuration"])
        self.slowDuration    = int(stimulus["slowDuration"])
        self.startMode       = stimulus["startMode"]
        self.delay           = int(stimulus["delay"])
        self.repetitions    = int(stimulus["repetitions"])
        self.duration       = int(stimulus["duration"])
        self.exitCriteria       = stimulus["exitCriteria"]
        self.shapeX         = 0
        self.shapeY         = 0 
        self.currRadius         = self.startShapeRadius
        self.speed              = self.fastSpeed
        self.timeInSpeed        = 0 
        self.delaySoFar         = 0
        self.timeInStimulus     = 0
        self.repNo              = 0
        self.f9CommunicationSent = False
        self.trigger_out_sent = False

        self.xType = stimulus["xType"]

        self.PixelsPerMSFastX = self.app.convertDegreestoPixels(self.fastSpeed, "X") / (
                    1000 / SLEEP_TIME)  # dividing by 1000 to convert to ms
        self.PixelsPerMSSlowX = self.app.convertDegreestoPixels(self.slowSpeed, "X") / (
                    1000 / SLEEP_TIME)  # dividing by 1000 to convert to ms
        self.PixelsPerMSFastY = self.app.convertDegreestoPixels(self.fastSpeed, "Y") / (
                    1000 / SLEEP_TIME)  # dividing by 1000 to convert to ms
        self.PixelsPerMSSlowY = self.app.convertDegreestoPixels(self.slowSpeed, "Y") / (
                    1000 / SLEEP_TIME)  # dividing by 1000 to convert to ms

        if self.xType == PIXELS_CALC_METHOD:
            # calculations for pixels
            self.stXStart        = int(stimulus["startX"])*app.vsWidth/VIRTUAL_SCREEN_LOGICAL_WIDTH
            self.stYStart        = int(stimulus["startY"])*app.vsHeight/VIRTUAL_SCREEN_LOGICAL_HEIGHT
            self.stXEnd          = int(stimulus["endX"])*app.vsWidth/VIRTUAL_SCREEN_LOGICAL_WIDTH
            self.stYEnd          = int(stimulus["endY"])*app.vsHeight/VIRTUAL_SCREEN_LOGICAL_HEIGHT
        else:
            # calculations for degrees
            self.start_degree = int(stimulus["startX"])
            self.end_degree = int(stimulus["endX"])
            self.current_degree = self.start_degree  # this will increase every cycle
            self.stXStart        = self.app.positionDegreesToVSTable[int(self.start_degree)] * self.app.vsWidth/VIRTUAL_SCREEN_LOGICAL_WIDTH
            self.stYStart        = int(stimulus["startY"])*app.vsHeight/VIRTUAL_SCREEN_LOGICAL_HEIGHT
            self.stXEnd          = self.app.positionDegreesToVSTable[int(self.end_degree)] * self.app.vsWidth/VIRTUAL_SCREEN_LOGICAL_WIDTH
            self.stYEnd          = int(stimulus["endY"])*app.vsHeight/VIRTUAL_SCREEN_LOGICAL_HEIGHT
            # calculating how many degrees we need to move each move, i.e. awakening
            if self.start_degree == self.end_degree or self.duration == 0:
                self.degrees_per_interval = 0
            else:
                self.degrees_per_interval = (self.end_degree - self.start_degree)*SLEEP_TIME/(self.duration)

        self.stXOrientation = LEFT_RIGHT
        self.stYOrientation = TOP_DOWN

        if self.stXStart > self.stXEnd:
            self.stXOrientation = RIGHT_LEFT
            self.PixelsPerMSFastX *= -1
            self.PixelsPerMSSlowX *= -1
        elif self.stXEnd == self.stXStart:
            self.PixelsPerMSFastX = 0
            self.PixelsPerMSSlowX = 0

        if self.stYStart > self.stYEnd:
            self.PixelsPerMSFastY *= -1
            self.PixelsPerMSSlowY *= -1
            self.stYOrientation = DOWN_TOP
        elif self.stYEnd == self.stYStart:
            self.PixelsPerMSFastY = 0
            self.PixelsPerMSSlowY = 0

        self.xChange = self.PixelsPerMSFastX
        self.yChange = self.PixelsPerMSFastY
        # change in size is determined based on the exitCriteria
        # if time then duration determines the pace of the change
        # if distance then it will be based on the time it will take the shape to travel from start to end

        if self.exitCriteria.lower() == TIME:
            self.radiusNorm = (self.endShapeRadius - self.startShapeRadius) / (self.duration / SLEEP_TIME)
        elif self.exitCriteria.lower() == DISTANCE:
            self.radiusNorm = (self.endShapeRadius - self.startShapeRadius) / (abs(self.stXStart - self.stXEnd) / self.xChange)
        elif self.exitCriteria.lower() == SPACER:
            self.radiusNorm =0

        # pre calculating edges to save time
        self.xEdge = self.app.vsX + self.stXEnd+SPACE_BUFFER
        self.yEdge = self.app.vsY + self.stYEnd+SPACE_BUFFER

    def init_shape(self, batch_no):

        self.batchNo = batch_no
        self.status = RUNNING
        self.f9CommunicationSent = False
        self.trigger_out_sent = False  # important to reset when shape is init (new move)

        # if SPACER then no need to set the data for the shape
        if self.exitCriteria.lower() == SPACER:
            logging.info("Spacer, no need to set other info")
            return

        self.shapeY = self.app.vsY + self.stYStart
        if self.xType == PIXELS_CALC_METHOD:
            self.shapeX = self.app.vsX + self.stXStart
        else: # DEGREES_CALC_METHOD
            # transform the location on logic 1000-pixel screen to our real virtual screen size
            self.shapeX = self.app.vsX + self.app.positionDegreesToVSTable[int(self.start_degree)] * self.app.vsWidth/1000

        self.currRadius = self.startShapeRadius
        # start with size 0, move does distortion correction so no need to display the shape here
        self.shape = self.canvas.create_oval(trunc(self.shapeX),
                                            trunc(self.shapeY),
                                            trunc(self.shapeX),
                                            trunc(self.shapeY),
                                            fill = self.app.stimulusColor, width=20,outline='')

        self.canvas.itemconfigure(self.shape, state='hidden')
        logging.info("Shape created")

    def move(self):
        """
            every cycle is 1 millisecond
            every cycle we add
                - if sleepTime == speed then move otherwise sleep for another millisecond
                - if timeInSpeed == fastDuration then change to slow mode (and vice versa)
                    - timeInSpeed = 0
                    - speed = fast/slow speed (as needed)

            speedMode = FAST or SLOW

        """
        
        self.timeInStimulus += SLEEP_TIME
        self.delaySoFar     += SLEEP_TIME

        if self.delaySoFar < self.delay:
            return

        if  self.app.f9CommunicationEnabled and not self.f9CommunicationSent:
            #logging.info("sending f9 communication")
            sendF9Marker()
            self.f9CommunicationSent = True


        if self.exitCriteria.lower() == SPACER:
            if self.timeInStimulus >= self.duration:
                logging.info("SPACER completed !")
                self.status = DONE

            return


        self.canvas.itemconfigure(self.shape, state='normal')
        #logging.debug(f' shape presented 1 {time.time()}')
        x0, y0, x1, y1 = self.canvas.coords(self.shape)


        if self.xType == PIXELS_CALC_METHOD:
            self.shapeX += self.xChange
        else:  # DEGREES_CALC_METHOD
            self.current_degree += self.degrees_per_interval
            # transform the location on logic 1000-pixel screen to our real virtual screen size
            self.shapeX = self.app.vsX + self.app.positionDegreesToVSTable[int(self.current_degree)] * self.app.vsWidth / 1000

        self.shapeY += self.yChange
        self.currRadius += self.radiusNorm

        """
        distortion correction
        Since the dish is oval there is a distortion on the sides, so the furthest the shape is from the center the 
        smaller it needs to be         
        """
        # 0 = 180, 10=170 , 80 = 100
        alpha = self.current_degree

        if self.current_degree > 90:
            alpha = 180 - alpha

        if alpha < 40:
            alpha=40

        adjusted_radius = self.currRadius * \
                          (1-((sin(radians(90-alpha))*sin(radians(45 -alpha/2)))/sin(radians(90 + alpha)/2)))

        logging.debug(f"current_radius={self.currRadius}, adjusted_radius={adjusted_radius}, "
                      f"current_degree={self.current_degree} ")

        #is it time to move the shape, small changes (eliminate by trunc) will not cause redraw
        #if  (trunc(self.shapeX) != x0 or
        #     trunc(self.shapeY) != y0 or
        #     self.radiusNorm != 0 or
        #    abs(adjusted_radius - self.currRadius) > 0.2 ):
            # logging.debug(f"move shape x = {self.shapeX} y = {self.shapeY} ")
        self.canvas.coords(self.shape,
                           self.shapeX,
                           self.shapeY,
                           self.shapeX+adjusted_radius,
                           self.shapeY+adjusted_radius)


        # is it time to change speed 
        if self.speedMode==FAST and self.timeInSpeed >= self.fastDuration:
            self.timeInSpeed = 0
            self.speedMode   = SLOW
            self.xChange = self.PixelsPerMSSlowX
            self.yChange = self.PixelsPerMSSlowY
        elif self.speedMode==SLOW and self.timeInSpeed >= self.slowDuration:
            self.timeInSpeed = 0
            self.speedMode   = FAST
            self.xChange = self.PixelsPerMSFastX
            self.yChange = self.PixelsPerMSFastY
        else :
            self.timeInSpeed +=1*SLEEP_TIME

        
        #logging.debug("moving shape to new location x="+str(x0)+" y="+str(y0))
        # This stimulus repetition reached its end
        if  ((self.stXOrientation==LEFT_RIGHT and x0 > self.xEdge) or
            (self.stXOrientation==RIGHT_LEFT and  x0 < self.xEdge) or
            (self.stYOrientation==TOP_DOWN and    y0 > self.yEdge) or
            (self.stYOrientation==DOWN_TOP and    y0 < self.yEdge) or
            (self.timeInStimulus >= self.duration and self.exitCriteria.lower() == TIME) ):
            logging.info("repetition completed !")
            self.repNo += 1
            self.canvas.delete(self.shape)        
            # we finished all repetitions for this stimulus, and we need to move to next stimulus
            if self.repNo >= self.repetitions:
                self.status = DONE
            else:
                logging.info("Starting repetition no="+str(self.repNo+1))
                self.init_shape(self.batchNo) #creating the shape for the next repetition

    def terminate_run(self):
        self.canvas.delete(self.shape)

    def __str__(self):
        return f'id = ({self.stimulusID} (x0,x1,y0,y1) {self.stXStart} {self.stXEnd} {self.stYStart} {self.stYEnd}(pixelsPerMS WxH) {round(self.PixelsPerMSFastX,1)}X{round(self.PixelsPerMSFastY,1)}'
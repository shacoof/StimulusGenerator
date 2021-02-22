
# imports every file form tkinter and tkinter.ttk 
from tkinter import Canvas,mainloop,Tk,BOTH
from tkinter import ttk
from tkinter.constants import MOVETO
from tkinter.ttk import tkinter
from utils import loadCSV,writeCSV,sendF9Marker
import csv
import logging
from math import trunc
import sys


#constants 
LOCATION    = "l"
SIZE        = "s"
UPDATE      = "u"
RUN         = "r"
PAUSE       = "p"
GO          = "g"
TOP_DOWN    = 1 
DOWN_TOP    = -1
LEFT_RIGHT  = 1
RIGHT_LEFT  = -1
LINE_WIDTH  = 10 
FILL_COLOR  = "RED"
FAST = 1
SLOW = 2
SLEEP_TIME = 1 # 1 ms
MOVE = 1
SLEEP = 0 
AFTER = "after"
WITH = "with"


# Global variables

# VS = visual screen 
vsWidth = 0
vsHeight = 0
vsX = 0 
vsY = 0
vs = 0
xBoundry = 0 
# stimulus  start and end 
stXStart        = 0
stYStart        = 0
stYEnd          = 0
stXEnd          = 0
stXOrientation  = 1 # 1 left-to-rigth -1 right-to-left
stYOrientation  = 1 # 1 top-down -1 down-top
startShapeRadius= 0
endShapeRadius  = 0
slowSpeed       = 0
fastDuration    = 0
slowDuration    = 0
timeInSpeed     = 0 
speed           = 0 
speedMode       = FAST # FAST or SLOW
sleepTime       = 0 
startMode       = AFTER # AFTER or WITH
delay           = 0 


canvas = 0
screen = 0
controlMode = "l" # l = location, s = size
appConfig = 0
stimulusList = []
shape = 0
state = PAUSE
stimulusListLoc = 0
repNo = 0 
xNorm = 0 
yNorm = 0 
stimulusState = "New" # New - never run, 
fastSpeed = 100
f9CommunicationEnabled = False # ON or OFF. if On then send f9 communication
xMode           = False # false - no x is displayed in the middle of the screen 
xVertical       = 0 
xHorizental     = 0 
bgColor         = 0
vsColor          = 0
stimulusColor   = 0

APP_CONFIG_FILE = "appConfig.csv"
STIMULUS_CONFIG = "StimulusConfig.csv"

def _create_circle(canvas, x, y, r, **kwargs):
    return canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

def printVirtualScreenData():
    logging.debug("X="+str(vsX)+" Y="+str(vsY)+" Width="+str(vsWidth)+" Height="+str(vsHeight))

def initApp():
    global vsWidth,vsHeight,vsX,vsY,appConfig,stimulusList,f9CommunicationEnabled,bgColor,vsColor,stimulusColor
    appConfig   = loadCSV(APP_CONFIG_FILE)   
    vsX         = getAppConfig("fishScreenStartX")
    vsY         = getAppConfig("fishScreenStartY")
    vsWidth     = getAppConfig("fishScreenWidth")
    vsHeight    = getAppConfig("fishScreenHeight")
    bgColor     = getAppConfig("backgroundColor","str")
    vsColor     = getAppConfig("virtualScreenColor","str")
    stimulusColor  = getAppConfig("stimulusColor","str")
    f9CommunicationEnabled = getAppConfig("f9CommunicationEnabled","str")
    if f9CommunicationEnabled.lower() == "on" :        
        f9CommunicationEnabled = True
    else :
        f9CommunicationEnabled = False
    logging.info("f9CommunicationEnabled="+str(f9CommunicationEnabled))
    printVirtualScreenData()

def getAppConfig(variableName, type="int"):
    """get appConfig value

    Args:
        variableName ([string]): [the name of the variable]
        type ([type]): [int or str]
    """

    ret = appConfig[0][variableName]
    if type=="int":
        return(int(ret))
    else:
        return (ret)   

def chagneVirtualScreenProperties(direction):
    """
    Change screen Location or Size properties based on controlMode

    Args:
        direction ([String]): the event.keySym Up,Down,Left,Rigth
    """
    global x,y # for some reason PYLANCE required this 
    global vsHeight,vsWidth,vsX,vsY
    x=0
    y=0

    if controlMode == LOCATION:
        if direction.lower() == "up":
            y -= 1
            vsY -=1
        elif direction.lower() == "down":
            y += 1
            vsY+=1
        elif direction.lower() == "left":
            x -= 1
            vsX-=1
        elif direction.lower() == "right":
            x += 1
            vsX+=1
    elif controlMode == SIZE:
        if direction.lower() == "up":
            vsHeight -= 1
        elif direction.lower() == "down":
            vsHeight += 1
        elif direction.lower() == "left":
            vsWidth -= 1
        elif direction.lower() == "right":
            vsWidth += 1

def calcGeometry(screen_width, screen_height):
    geometryStr = str(screen_width)+"x"+str(screen_height)+"+-10+0"
    return geometryStr

def processEvent(event):
    logging.debug(event)
    chagneVirtualScreenProperties(event.keysym)
    if controlMode == LOCATION:
        canvas.move(vs,x,y)
        canvas.move(xBoundry,x,y)
    elif controlMode == SIZE:
        x0, y0, x1, y1 = canvas.coords(vs)
        x1 = x0 + vsWidth
        y1 = y0 + vsHeight
        canvas.coords(vs, x0, y0, x1, y1)
        canvas.coords(xBoundry,x0-10,y0-10,x1+10,y1+10)
        
    return

def changeControlMode(event):
    global controlMode
    logging.debug(event)
    chagneVirtualScreenProperties(event.keysym)
    controlMode = event.keysym
    logging.debug("ControlMode="+event.keysym)

def updateConfig(event):
    """
    Update config file with current VS settings
    """
    global appConfig
    logging.debug(event)
    appConfig[0]["fishScreenStartX"]=vsX
    appConfig[0]["fishScreenStartY"]=vsY
    appConfig[0]["fishScreenWidth"]=vsWidth
    appConfig[0]["fishScreenHeight"]=vsHeight
    writeCSV(APP_CONFIG_FILE,appConfig[0])
    printVirtualScreenData()
    logging.debug("Config file updates successfully")

def leaveProg(event):
    logging.debug(event)
    logging.info("Bye bye !")
    sys.exit()

def printHelp(event): 
    print("NOTE : press the keyboard while focus is on the graphic screen")   
    print("========== General actions ==========")
    print('? = for help (this screen)')
    print('e - to exit')
    print("========== Controlling Virtual Screen ==========")
    print('l - switch to virtual screen location setting')
    print('s - switch to virtual screen size setting ')
    print('<Up> - move up or reduce virtual screen height')
    print('<Down> - move down or extend virtual screen hegith')
    print('<Left>  - move left or reduce virtual screen width')
    print('<Right> - move right or extend virtual screen size ')
    print('u - to update app config with current virtual screen location')
    print("========== Controlling Stimulus generator ==========")
    print('r - Run the stimuli, pressing r again will restart')
    print('p - Pause the run')
 
def initStimuli():
    """called whenever a new run starts. all stimulus will run from the beginig. 
    """
    global repNo, stimulusListLoc,stimulusList
    repNo = 0
    stimulusListLoc = 0 
    stimulusList=loadCSV(STIMULUS_CONFIG)
    initShape(stimulusList[0])

def initShape(stimulus):
    """create the shape and set the x,y changed for each iteration

    Args:
        stimulus : 1 stimulus from the stimulusList
    """
    global shape, xNorm, yNorm,fastSpeed, stXEnd,stXStart,stYEnd,stYStart
    global stXOrientation,stYOrientation,endShapeRadius,startMode,delay
    global slowSpeed,fastDuration,slowDuration,startShapeRadius,speed, timeInSpeed,sleepTime

    # converting the shape location into the VirtualScreen space
    # shape location is between 0-1000 and the actual virtualScreen width/height are different 
    stXStart        = int(stimulus["startX"])*vsWidth/1000
    stYStart        = int(stimulus["startY"])*vsHeight/1000
    stXEnd          = int(stimulus["endX"])*vsWidth/1000
    stYEnd          = int(stimulus["endY"])*vsHeight/1000
    fastSpeed       = int(stimulus["fastSpeed"])
    startShapeRadius = int(stimulus["startShapeRadius"])
    endShapeRadius  = int(stimulus["endShapeRadius"])
    slowSpeed       = int(stimulus["slowSpeed"])
    fastDuration    = int(stimulus["fastDuration"])
    slowDuration    = int(stimulus["slowDuration"])
    startMode       = stimulus["startMode"]
    delay           = int(stimulus["delay"])
    speed = fastSpeed
    timeInSpeed = 0 
    sleepTime = 0

    # since x,y is between 0-999 but actual width,height are different we need to normalize steps
    xNorm = vsWidth/1000
    yNorm = vsHeight/1000

    stXOrientation = LEFT_RIGHT
    stYOrientation = TOP_DOWN     

    if stXStart > stXEnd:
        xNorm=-xNorm
        stXOrientation = RIGHT_LEFT
    elif stXEnd == stXStart:
        xNorm = 0

    if stYStart > stYEnd:
        yNorm=-yNorm
        stYOrientation = DOWN_TOP
    elif stYEnd == stYStart:
        yNorm=0
    
    cx0 = vsX + stXStart
    cy0 = vsY + stYStart

    #shape = canvas.create_rectangle(trunc(cx0),
    #                                trunc(cy0),
    #                                trunc(cx0+int(stimulus["startShapeWidth"])),
    #                                trunc(cy0+int(stimulus["startShapeHeight"])),
    #                                fill = "black")

    if  f9CommunicationEnabled:
        logging.info("sending f9 communication")
        sendF9Marker()

    shape = _create_circle( canvas,
                            trunc(cx0),
                            trunc(cy0),
                            trunc(startShapeRadius),
                            fill = stimulusColor)

    logging.info("Shape created")

def calcMove():
    """
        every cycle is 1 milisecond
        every cycle the program wakes up and decide what to do 
            - if sleepTime == speed then move otherwise slepp for another milisecond
            - if timeInSpeed == fastDuration then chagne to slow mode (and vice versa)
                - timeInSpeed = 0
                - speed = fast/slow speed (as needed)

        speedMode = FAST or SLOW

    """
    global speed, timeInSpeed, speedMode,sleepTime,shape,xNorm,yNorm
    if sleepTime == speed:
        canvas.move(shape,xNorm,yNorm)
        sleepTime = 0
    else:
        sleepTime +=1

    if speedMode==FAST and timeInSpeed == fastDuration:
        timeInSpeed = 0
        speed       = slowSpeed
        speedMode   = SLOW
        sleepTime   = 0
    elif speedMode==SLOW and timeInSpeed == slowDuration:
        timeInSpeed = 0
        speed       = fastSpeed
        speedMode   = FAST
        sleepTime   = 0
    else :
        timeInSpeed +=1

def runStimuli():
    # Shape,startShapeWidth,startShapeHeight,StartX,StartY,EndX,EndY,Repetitions,fastSpeed      
    global shape, stimulusListLoc, repNo, stimulusList, stimulusState, xNorm,yNorm, canvas, fastSpeed,state

    if state == PAUSE:
        return

    calcMove() # decide whether it's time to move and in what speed 

    x0, y0, x1, y1 = canvas.coords(shape)
    #logging.info("moving shape to new location x="+str(x0)+" y="+str(y0))
    # This stimulus repitiion reached its end 
    if  ((stXOrientation==LEFT_RIGHT and x1 > vsX + stXEnd+20) or
        (stXOrientation==RIGHT_LEFT and x1 < vsX + stXEnd+20) or 
        (stYOrientation==TOP_DOWN and y1 > vsY + stYEnd+20) or
        (stYOrientation==DOWN_TOP and y1 < vsY + stYEnd+20)): 
        logging.info("repetition completed !")
        repNo += 1
        canvas.delete(shape)        
        # we finished all repitiions for this stimulus and we need to move to next stimulus
        if repNo >= int(stimulusList[stimulusListLoc]["repetitions"]):            
            stimulusListLoc += 1
            repNo = 0
            #we finsihed all stimuluses
            if stimulusListLoc >= len(stimulusList):
                logging.info("All stimuli were executed ! ")
            else:
                logging.info("Starting stimulus no="+str(stimulusListLoc+1))
                initShape(stimulusList[stimulusListLoc]) #creating the shape for the next repitition 
                canvas.after(SLEEP_TIME,runStimuli)
        else:
            logging.info("Starting repetition no="+str(repNo+1))
            initShape(stimulusList[stimulusListLoc]) #creating the shape for the next repitition 
            canvas.after(SLEEP_TIME,runStimuli)
    else:
        canvas.after(SLEEP_TIME,runStimuli)

def manageStimulus(event):    
    global state, shape, canvas
    logging.debug(event)
        
    if event.keysym == PAUSE:
        state = PAUSE     
        canvas.delete(shape)
    elif event.keysym == RUN:
        state = RUN
        initStimuli()        
        runStimuli()        
 
def showCrossAndBoundries(event):
    global xMode,xVertical,xHorizental, canvas,xBoundry
    logging.debug(event)    

    if xMode:
        xMode = False
        canvas.delete(xHorizental)
        canvas.delete(xVertical)
        canvas.delete(xBoundry)
    else:
        xMode = True
        xBoundry    = canvas.create_rectangle(vsX-10,vsY-10,vsX+vsWidth+10,vsY+vsHeight+10,fill = FILL_COLOR)
        xHorizental = canvas.create_line(vsX,vsY+vsHeight/2,vsX+vsWidth,vsY+vsHeight/2,fill=FILL_COLOR,width= LINE_WIDTH)        
        xVertical   = canvas.create_line(vsX+vsWidth/2,vsY,vsX+vsWidth/2,vsY+vsHeight,fill=FILL_COLOR,width= LINE_WIDTH)
        canvas.tag_lower(xBoundry)
 
def main():
    global vs,canvas,screen, xBoundry
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    initApp()

    # object of class Tk, resposible for creating 
    # a tkinter toplevel window 
    screen = Tk() 
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    geometryStr = calcGeometry(screen_width, screen_height)
    screen.geometry(geometryStr)
    canvas = Canvas(screen,background="black")
    canvas.pack(fill=BOTH,expand=1)    
    vs = canvas.create_rectangle(vsX,vsY,vsX+vsWidth,vsY+vsHeight,fill = vsColor) 
    screen.bind('<Up>', processEvent)
    screen.bind('<Down>', processEvent)
    screen.bind('<Left>',processEvent)
    screen.bind('<Right>',processEvent)
    screen.bind('l',changeControlMode)
    screen.bind('s',changeControlMode)
    screen.bind('u',updateConfig)
    screen.bind('e',leaveProg)
    screen.bind('r',manageStimulus)
    screen.bind('p',manageStimulus)
    screen.bind('x',showCrossAndBoundries)
    screen.bind('?',printHelp)
    printHelp("")

    mainloop() 

if __name__ == "__main__": 
   main() 

   

   
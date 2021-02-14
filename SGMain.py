
# imports every file form tkinter and tkinter.ttk 
from tkinter import Canvas,mainloop,Tk,BOTH
from tkinter import ttk
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
xMode = False # false - no x is displayed in the middle of the screen 
xVertical = 0 
xHorizental = 0 

APP_CONFIG_FILE = "appConfig.csv"
STIMULUS_CONFIG = "StimulusConfig.csv"

def _create_circle(canvas, x, y, r, **kwargs):
    return canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

def printVirtualScreenData():
    logging.debug("X="+str(vsX)+" Y="+str(vsY)+" Width="+str(vsWidth)+" Height="+str(vsHeight))

def initApp():
    global vsWidth,vsHeight,vsX,vsY,appConfig,stimulusList,f9CommunicationEnabled
    appConfig=loadCSV(APP_CONFIG_FILE)   
    vsX = int(appConfig[0]["fishScreenStartX"])
    vsY = int(appConfig[0]["fishScreenStartY"])
    vsWidth = int(appConfig[0]["fishScreenWidth"])
    vsHeight= int(appConfig[0]["fishScreenHeight"])
    if appConfig[0]["f9CommunicationEnabled"].lower() == "on" :        
        f9CommunicationEnabled = True
    logging.info("f9CommunicationEnabled="+str(f9CommunicationEnabled))
    printVirtualScreenData()
    stimulusList=loadCSV(STIMULUS_CONFIG)

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
    global repNo, stimulusListLoc
    repNo = 0
    stimulusListLoc = 0 
    initShape(stimulusList[0])

def initShape(stimulus):
    """create the shape and set the x,y changed for each iteration

    Args:
        stimulus : 1 stimulus from the stimulusList
    """
    global shape, xNorm, yNorm,fastSpeed, stXEnd,stXStart,stYEnd,stYStart,stXOrientation,stYOrientation,endShapeRadius,slowSpeed,fastDuration,slowDuration,startShapeRadius

    # converting the shape location into the VirtualScreen space
    # shape location is between 0-1000 and the actual virtualScreen width/height are different 
    stXStart = int(stimulus["startX"])*vsWidth/1000
    stYStart = int(stimulus["startY"])*vsHeight/1000
    stXEnd = int(stimulus["endX"])*vsWidth/1000
    stYEnd = int(stimulus["endY"])*vsHeight/1000
    fastSpeed = int(stimulus["fastSpeed"])
    startShapeRadius = int(stimulus["startShapeRadius"])
    endShapeRadius = int(stimulus["endShapeRadius"])
    slowSpeed = int(stimulus["slowSpeed"])
    fastDuration = int(stimulus["fastDuration"])
    slowDuration = int(stimulus["slowDuration"])
    
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
                            fill = "black")

    logging.info("Shape created")

def runStimuli():
    # Shape,startShapeWidth,startShapeHeight,StartX,StartY,EndX,EndY,Repetitions,fastSpeed      
    global shape, stimulusListLoc, repNo, stimulusList, stimulusState, xNorm,yNorm, canvas, fastSpeed

    canvas.move(shape,xNorm,yNorm)

    x0, y0, x1, y1 = canvas.coords(shape)
    #logging.info("moving shape to new location x="+str(x0)+" y="+str(y0))
    # This stimulus repitiion reached its end 
    if  ((stXOrientation==LEFT_RIGHT and x1 > vsX + stXEnd+5) or
        (stXOrientation==RIGHT_LEFT and x1 < vsX + stXEnd+5) or 
        (stYOrientation==TOP_DOWN and y1 > vsY + stYEnd+5) or
        (stYOrientation==DOWN_TOP and y1 < vsY + stYEnd+5)): 
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
                canvas.after(fastSpeed,runStimuli)
        else:
            logging.info("Starting repetition no="+str(repNo+1))
            initShape(stimulusList[stimulusListLoc]) #creating the shape for the next repitition 
            canvas.after(fastSpeed,runStimuli)
    else:
        canvas.after(fastSpeed,runStimuli)

def manageStimulus(event):    
    global state, shape
    logging.debug(event)    
    if event.keysym == PAUSE:
        state = PAUSE        
    elif event.keysym == RUN:
        state == RUN
        initStimuli()        
        runStimuli()        
 
def showCrossAndBoundries(event):
    global xMode,xVertical,xHorizental
    logging.debug(event)    

    if xMode:
        xMode = False
        canvas.delete(xHorizental)
        canvas.delete(xVertical)
    else:
        xMode = True
        xHorizental = canvas.create_line(vsX,(vsY+vsHeight)/2,(vsX+vsWidth),(vsY+vsHeight)/2,fill='black')        
        xVertical   = canvas.create_line((vsX+vsWidth)/2,vsY,(vsX+vsWidth)/2,vsY+vsHeight,fill='black')
 
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
    xBoundry = canvas.create_rectangle(vsX-10,vsY-10,vsX+vsWidth+10,vsY+vsHeight+10,fill = "yellow")
    vs = canvas.create_rectangle(vsX,vsY,vsX+vsWidth,vsY+vsHeight,fill = "white") 
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
    screen.bind('g',manageStimulus)
    screen.bind('x',showCrossAndBoundries)
    screen.bind('?',printHelp)
    printHelp("")

    mainloop() 

if __name__ == "__main__": 
   main() 

   
   #ActionItem add reload of csv
   #actionitem show/hide cross
   #actionItem changing speed
   #actionItem subsequent stimuli are running with delay>=0 with previous one , like power-point
   #increase size linearly 
   
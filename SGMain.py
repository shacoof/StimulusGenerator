
# imports every file form tkinter and tkinter.ttk 
from tkinter import *
from tkinter import ttk
from tkinter.ttk import *  
from utils import *
import csv
import logging
from math import *

#constants 
LOCATION    = "l"
SIZE        = "s"
UPDATE      = "u"
RUN         = "r"
PAUSE       = "p"
GO          = "g"


# Global variables
VSWidth = 0
VSHeight = 0
VSX = 0 
VSY = 0
VS = 0
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
speed = 100

APP_CONFIG_FILE = "appConfig.csv"
STIMULUS_CONFIG = "StimulusConfig.csv"

def printVirtualScreenData():
    logging.debug("X="+str(VSX)+" Y="+str(VSY)+" Width="+str(VSWidth)+" Height="+str(VSHeight))

def initApp():
    global VSWidth,VSHeight,VSX,VSY,appConfig,stimulusList
    appConfig=loadCSV(APP_CONFIG_FILE)   
    VSX = int(appConfig[0]["fishScreenStartX"])
    VSY = int(appConfig[0]["fishScreenStartY"])
    VSWidth = int(appConfig[0]["fishScreenWidth"])
    VSHeight= int(appConfig[0]["fishScreenHeight"])
    printVirtualScreenData();
    stimulusList=loadCSV(STIMULUS_CONFIG)

def chagneVirtualScreenProperties(direction):
    """
    Change screen Location or Size properties based on controlMode

    Args:
        direction ([String]): the event.keySym Up,Down,Left,Rigth
    """
    global x,y # for some reason PYLANCE required this 
    global VSHeight,VSWidth,VSX,VSY
    x=0
    y=0

    if controlMode == LOCATION:
        if direction.lower() == "up":
            y -= 1;
            VSY -=1
        elif direction.lower() == "down":
            y += 1;
            VSY+=1
        elif direction.lower() == "left":
            x -= 1;
            VSX-=1
        elif direction.lower() == "right":
            x += 1;
            VSX+=1
    elif controlMode == SIZE:
        if direction.lower() == "up":
            VSHeight -= 1;
        elif direction.lower() == "down":
            VSHeight += 1;
        elif direction.lower() == "left":
            VSWidth -= 1;
        elif direction.lower() == "right":
            VSWidth += 1;

def calcGeometry(screen_width, screen_height):
    geometryStr = str(screen_width)+"x"+str(screen_height)+"+-10+0"
    return geometryStr

def processEvent(event):
    logging.debug(event)
    chagneVirtualScreenProperties(event.keysym)
    if controlMode == LOCATION:
        canvas.move(VS,x,y)
    elif controlMode == SIZE:
        x0, y0, x1, y1 = canvas.coords(VS)
        x1 = x0 + VSWidth
        y1 = y0 + VSHeight
        canvas.coords(VS, x0, y0, x1, y1)
        
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
    appConfig[0]["fishScreenStartX"]=VSX
    appConfig[0]["fishScreenStartY"]=VSY
    appConfig[0]["fishScreenWidth"]=VSWidth
    appConfig[0]["fishScreenHeight"]=VSHeight
    writeCSV(APP_CONFIG_FILE,appConfig[0])
    printVirtualScreenData();
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
    print('r - to update app config with current virtual screen location')
    print('p - to update app config with current virtual screen location')
    print('g - to update app config with current virtual screen location')
 
def initStimulus():
    """called whenever a new run starts. all stimulus will run from the beginig. 
    """
    global repNo, stimulusListLoc
    repNo = 0
    stimulusListLoc = 0 
    createShape(stimulusList[0])

def createShape(stimulus):
    """create the shape and set the x,y changed for each iteration

    Args:
        stimulus : 1 stimulus from the stimulusList
    """
    global shape, xNorm, yNorm,speed
    x0 = int(stimulus["startX"])
    y0 = int(stimulus["startY"])
    x1 = int(stimulus["endX"])
    y1 = int(stimulus["endY"])
    speed = int(stimulus["speed"])
    # since x,y is between 0-999 but actual width,height are different we need to normalize steps
    xNorm = VSWidth/1000
    yNorm = VSHeight/1000

    if x0 > x1:
        xNorm=-xNorm
    elif x1 == x0:
        xNorm = 0

    if y0 > y1:
        yNorm=-yNorm
    elif y1 == y0:
        yNorm=0

    # converting the relative shape location into the VirtualScreen space
    cx0 = VSX + (x0*VSWidth)/1000
    cy0 = VSY + (y0*VSHeight)/1000

    shape = canvas.create_rectangle(trunc(cx0),
                                    trunc(cy0),
                                    trunc(cx0+int(stimulus["shapeWidth"])),
                                    trunc(cy0+int(stimulus["shapeHeight"])),
                                    fill = "black")

    logging.info("Shape created")

def runStimulus():
    # Shape,ShapeWidth,ShapeHeight,StartX,StartY,EndX,EndY,Repetitions,speed      
    global shape, stimulusListLoc, repNo, stimulusList, repetitionNo, stimulusState, xNorm,yNorm, canvas, speed

    canvas.move(shape,xNorm,yNorm)

    x0, y0, x1, y1 = canvas.coords(shape)
    #logging.info("moving shape to new location x="+str(x0)+" y="+str(y0))
    # This stimulus repitiion reached its end 
    if x1 > VSX+VSWidth or y1 > VSY + VSHeight or x1<0 or y1<0: 
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
                createShape(stimulusList[stimulusListLoc]) #creating the shape for the next repitition 
                canvas.after(speed,runStimulus)
        else:
            logging.info("Starting repetition no="+str(repNo+1))
            createShape(stimulusList[stimulusListLoc]) #creating the shape for the next repitition 
            canvas.after(speed,runStimulus)
    else:
        canvas.after(speed,runStimulus)

def manageStimulus(event):    
    global state, shape
    logging.debug(event)    
    if event.keysym == PAUSE:
        state = PAUSE        
    elif event.keysym == RUN:
        state == RUN
        initStimulus()        
        runStimulus()
        
 
def main():
    global VS,canvas,screen
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
    #rectangle = canvas.create_rectangle(10,10,100,20,fill = "blue")
    #rectangle2 = canvas.create_rectangle(100,100,150,40,fill = "red")
    VS = canvas.create_rectangle(VSX,VSY,VSX+VSWidth,VSY+VSHeight,fill = "white") 
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
    screen.bind('?',printHelp)
    printHelp("")

    mainloop() 

if __name__ == "__main__": 
   main() 
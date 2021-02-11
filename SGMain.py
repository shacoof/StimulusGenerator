
# imports every file form tkinter and tkinter.ttk 
from tkinter import *
from tkinter import ttk
from tkinter.ttk import *  
from utils import *
import csv
import logging

# Global variables
screenWidth = 0
screenHeight = 0
screenX = 0 
screenY = 0
fishVirtualScreen = 0
canvas = 0
screen = 0
controlMode = "l" # l = location, s = size
LOCATION = "l"
SIZE = "s"

APP_CONFIG_FILE = "appConfig.csv"

def printVirtualScreenLoc():
    logging.debug("X="+str(screenX)+" Y="+str(screenY)+" Width="+str(screenWidth)+" Height="+str(screenHeight))

def initVirtualScreen():
    global screenWidth,screenHeight,screenX,screenY 
    appConfigDict=loadCSV(APP_CONFIG_FILE)   
    screenX = int(appConfigDict[0].get("fishScreenStartX"))
    screenY = int(appConfigDict[0].get("fishScreenStartY"))
    screenWidth = int(appConfigDict[0].get("fishScreenWidth"))
    screenHeight= int(appConfigDict[0].get("fishScreenHeight"))
    printVirtualScreenLoc();


def chagneVirtualScreenProperties(direction):
    """
    Change screen Location or Size properties based on controlMode

    Args:
        direction ([String]): the event.keySym Up,Down,Left,Rigth
    """
    global x,y # for some reason PYLANCE required this 
    global screenHeight,screenWidth
    x=0
    y=0

    if controlMode == LOCATION:
        if direction.lower() == "up":
            y -= 1;
        elif direction.lower() == "down":
            y += 1;
        elif direction.lower() == "left":
            x -= 1;
        elif direction.lower() == "right":
            x += 1;
    elif controlMode == SIZE:
        if direction.lower() == "up":
            screenHeight -= 1;
        elif direction.lower() == "down":
            screenHeight += 1;
        elif direction.lower() == "left":
            screenWidth -= 1;
        elif direction.lower() == "right":
            screenWidth += 1;

def calcGeometry(screen_width, screen_height):
    geometryStr = str(screen_width)+"x"+str(screen_height)+"+-10+0"
    return geometryStr

def processEvent(event):
    logging.debug(event)
    chagneVirtualScreenProperties(event.keysym)
    if controlMode == LOCATION:
        canvas.move(fishVirtualScreen,x,y)
    elif controlMode == SIZE:
        x0, y0, x1, y1 = canvas.coords(fishVirtualScreen)
        x1 = x0 + screenWidth
        y1 = y0 + screenHeight
        canvas.coords(fishVirtualScreen, x0, y0, x1, y1)
        
    return

def changeControlMode(event):
    global controlMode
    logging.debug(event)
    chagneVirtualScreenProperties(event.keysym)
    controlMode = event.keysym
    logging.debug("ControlMode="+event.keysym)

def main():
    global fishVirtualScreen,canvas,screen
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    initVirtualScreen()

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
    fishVirtualScreen = canvas.create_rectangle(screenX,screenY,screenX+screenWidth,screenY+screenHeight,fill = "white") 
    screen.bind('<Up>', processEvent)
    screen.bind('<Down>', processEvent)
    screen.bind('<Left>',processEvent)
    screen.bind('<Right>',processEvent)
    screen.bind('l',changeControlMode)
    screen.bind('s',changeControlMode)


    mainloop() 

if __name__ == "__main__": 
   main() 
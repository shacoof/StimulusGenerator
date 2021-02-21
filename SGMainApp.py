from tkinter import Canvas,mainloop,Tk,BOTH
from tkinter import ttk
from tkinter.constants import MOVETO
from tkinter.ttk import tkinter
from utils import loadCSV,writeCSV,sendF9Marker
import logging
from math import trunc
import sys
from StimuliGenerator import *


class App:
    APP_CONFIG_FILE = "appConfig.csv"
    LOCATION    = "l"
    SIZE        = "s"
    UPDATE      = "u"
    RUN         = "r"
    PAUSE       = "p"
    GO          = "g"
    FILL_COLOR  = "Green"
    LOCATION    = "l"
    SIZE        = "s"
    UPDATE      = "u"
    RUN         = "r"
    PAUSE       = "p"
    GO          = "g"
    LINE_WIDTH  = 5 
    FAST = 1
    SLOW = 2
    SLEEP_TIME = 1 # 1 ms
    MOVE = 1
    SLEEP = 0 
    sg = ""

    def __init__(self,screen):
        self.controlMode = "l" # l = location, s = size
        self.xMode           = False # false - no x is displayed in the middle of the screen 
        self.xVertical       = 0 
        self.xHorizental     = 0 
        self.bgColor         = 0
        self.vsColor         = 0
        self.stimulusColor   = 0
        self.xBoundry        = 0 
        self.appConfig   = loadCSV(self.APP_CONFIG_FILE)   
        self.vsX         = self.getAppConfig("fishScreenStartX")
        self.vsY         = self.getAppConfig("fishScreenStartY")
        self.vsWidth     = self.getAppConfig("fishScreenWidth")
        self.vsHeight    = self.getAppConfig("fishScreenHeight")
        self.bgColor     = self.getAppConfig("backgroundColor","str")
        self.vsColor     = self.getAppConfig("virtualScreenColor","str")
        self.stimulusColor  = self.getAppConfig("stimulusColor","str")
        self.f9CommunicationEnabled = self.getAppConfig("f9CommunicationEnabled","str")
        if self.f9CommunicationEnabled.lower() == "on" :        
            self.f9CommunicationEnabled = True
        else :
            self.f9CommunicationEnabled = False
        logging.info("f9CommunicationEnabled="+str(self.f9CommunicationEnabled))
        self.printVirtualScreenData()
        self.screen_width = screen.winfo_screenwidth()
        self.screen_height = screen.winfo_screenheight()
        geometryStr = self.calcGeometry(self.screen_width, self.screen_height)
        screen.geometry(geometryStr)
        self.canvas = Canvas(screen,background="black")
        self.canvas.pack(fill=BOTH,expand=1)    
        self.vs = self.canvas.create_rectangle(self.vsX,self.vsY,self.vsX+self.vsWidth,self.vsY+self.vsHeight,fill = self.vsColor) 
        screen.bind('<Up>', self.processEvent)
        screen.bind('<Down>', self.processEvent)
        screen.bind('<Left>',self.processEvent)
        screen.bind('<Right>',self.processEvent)
        screen.bind('l',self.changeControlMode)
        screen.bind('s',self.changeControlMode)
        screen.bind('u',self.updateConfig)
        screen.bind('e',self.leaveProg)
        screen.bind('r',self.manageStimulus)
        screen.bind('p',self.manageStimulus)
        screen.bind('x',self.showCrossAndBoundries)
        screen.bind('?',self.printHelp)
        self.printHelp("")

    def updateConfig(self,event):
        """
        Update config file with current VS settings
        """
        logging.debug(event)
        self.appConfig[0]["fishScreenStartX"]=self.vsX
        self.appConfig[0]["fishScreenStartY"]=self.vsY
        self.appConfig[0]["fishScreenWidth"]=self.vsWidth
        self.appConfig[0]["fishScreenHeight"]=self.vsHeight
        writeCSV(self.APP_CONFIG_FILE,self.appConfig[0])
        self.printVirtualScreenData()
        logging.debug("Config file updates successfully")

    def leaveProg(self,event):
        logging.debug(event)
        logging.info("Bye bye !")
        sys.exit()

    def printHelp(self,event): 
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
    
    def changeControlMode(self,event):
        logging.debug(event)
        self.chagneVirtualScreenProperties(event.keysym)
        self.controlMode = event.keysym
        logging.debug("ControlMode="+event.keysym)

    def processEvent(self,event):
        logging.debug(event)
        self.chagneVirtualScreenProperties(event.keysym)
        if self.controlMode == self.LOCATION:
            self.canvas.move(self.vs,x,y)
            self.canvas.move(self.xBoundry,x,y)
        elif self.controlMode == self.SIZE:
            x0, y0, x1, y1 = self.canvas.coords(self.vs)
            x1 = x0 + self.vsWidth
            y1 = y0 + self.vsHeight
            self.canvas.coords(self.vs, x0, y0, x1, y1)
            self.canvas.coords(self.xBoundry,x0-10,y0-10,x1+10,y1+10)
            
        return

    def manageStimulus(self,event):    
        logging.debug(event)
            
        if event.keysym == self.PAUSE:
            self.state = self.PAUSE    
            if self.sg != "":
                self.sg.terminateRun()
        elif event.keysym == self.RUN:
            self.state = self.RUN
            self.sg = StimulusGenerator(self.canvas, self)
            self.runStimuli()           

    def runStimuli(self):
        if self.state == self.PAUSE:
            return
        if self.sg.runStimuli() == Stimulus.DONE:
            logging.info("All stimuli were executed ! ")
            self.state = self.PAUSE
        else:
            self.canvas.after(self.SLEEP_TIME,self.runStimuli)

    def calcGeometry(self,screen_width, screen_height):
        geometryStr = str(screen_width)+"x"+str(screen_height)+"+-10+0"
        return geometryStr
    
    def printVirtualScreenData(self):
        logging.debug("X="+str(self.vsX)+" Y="+str(self.vsY)+" Width="+str(self.vsWidth)+" Height="+str(self.vsHeight))
    
    def getAppConfig(self,variableName, type="int"):
        """get appConfig value

        Args:
            variableName ([string]): [the name of the variable]
            type ([type]): [int or str]
        """

        ret = self.appConfig[0][variableName]
        if type=="int":
            return(int(ret))
        else:
            return (ret)   

    def showCrossAndBoundries(self,event):        
        logging.debug(event)    

        if self.xMode:
            self.xMode = False
            self.canvas.delete(self.xHorizental)
            self.canvas.delete(self.xVertical)
            self.canvas.delete(self.xBoundry)
        else:
            self.xMode = True
            self.xBoundry    = self.canvas.create_rectangle(self.vsX-10,self.vsY-10,self.vsX+self.vsWidth+10,self.vsY+self.vsHeight+10,fill = self.FILL_COLOR)
            self.xHorizental = self.canvas.create_line(self.vsX,self.vsY+self.vsHeight/2,self.vsX+self.vsWidth,self.vsY+self.vsHeight/2,fill=self.FILL_COLOR,width= self.LINE_WIDTH)        
            self.xVertical   = self.canvas.create_line(self.vsX+self.vsWidth/2,self.vsY,self.vsX+self.vsWidth/2,self.vsY+self.vsHeight,fill=self.FILL_COLOR,width= self.LINE_WIDTH)
            self.canvas.tag_lower(self.xBoundry)
 
    def chagneVirtualScreenProperties(self,direction):
        """
        Change screen Location or Size properties based on controlMode

        Args:
            direction ([String]): the event.keySym Up,Down,Left,Rigth
        """
        global x,y # for some reason PYLANCE required this 
        global vsHeight,vsWidth,vsX,vsY
        x=0
        y=0

        if self.controlMode == self.LOCATION:
            if direction.lower() == "up":
                y -= 1
                self.vsY -=1
            elif direction.lower() == "down":
                y += 1
                self.vsY+=1
            elif direction.lower() == "left":
                x -= 1
                self.vsX-=1
            elif direction.lower() == "right":
                x += 1
                self.vsX+=1
        elif self.controlMode == self.SIZE:
            if direction.lower() == "up":
                self.vsHeight -= 1
            elif direction.lower() == "down":
                self.vsHeight += 1
            elif direction.lower() == "left":
                self.vsWidth -= 1
            elif direction.lower() == "right":
                self.vsWidth += 1


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
root = Tk()
app = App(root)
root.mainloop()
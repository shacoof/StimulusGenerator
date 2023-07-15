import time
from tkinter import Canvas, Tk, BOTH
from tkinter import ttk
from tkinter.ttk import tkinter
from screeninfo import get_monitors
import nidaqmx
import shutil

import utils
from NiDaqPulse import NiDaqPulse
from utils import writeCSV
from math import pi, cos
import sys
from StimuliGenerator import *
import constants
import multiprocessing
import image_reader_worker
import datetime
import image_writer_worker
from utils import opencv_create_video


# TODO save all config files in separate directory and allow to use specific config file by it's name (select screen) - no need to edit them
# TODO allow re-run - new file prefix
# TODO allow pause, run after pause is re-run
# TODO lose focuse at the end of the run... maybe because of print ?


class App:
    sg = ""

    def __init__(self, screen):
        # these 3 are for ni-daq output (port & line it's connected to, device is initialized later)
        self.screen = screen
        self.state = None
        self.port = 1
        self.stimulus_line = 7
        self.camera_line = 6
        self.stimulus_output_device = None
        self.debug = False
        self.controlMode = "l"  # l = location, s = size
        # user by the x (cross) button to show virtual screen cross
        self.xMode = False  # false - no x is displayed in the middle of the screen
        self.xVertical = 0
        self.xHorizental = 0
        self.xBoundry = 0

        # used by the y (mid button)  button to show mid screen marker
        self.yMode = False  # False - no midScreen marker is displayed
        self.yVertical = 0
        self.yHorizental = 0
        #####
        self.bgColor = 0
        self.vsColor = 0
        self.stimulusColor = 0
        self.appConfig = loadCSV(constants.APP_CONFIG_FILE)
        self.vsX = self.getAppConfig("fishScreenStartX")
        self.vsY = self.getAppConfig("fishScreenStartY")
        self.vsWidth = self.getAppConfig("fishScreenWidth")
        self.vsHeight = self.getAppConfig("fishScreenHeight")
        self.bgColor = self.getAppConfig("backgroundColor", "str")
        self.vsColor = self.getAppConfig("virtualScreenColor", "str")
        self.stimulusColor = self.getAppConfig("stimulusColor", "str")
        self.f9CommunicationEnabled = self.getAppConfig("f9CommunicationEnabled", "str")
        self.NiDaqPulseEnabled = self.getAppConfig("NiDaqPulseEnabled", "str")
        self.DishRadiusSize = self.getAppConfig("DishRadiusSize")
        self.VirtualScreenDegrees = self.getAppConfig("VirtualScreenDegrees")
        self.VirtualScreenWidthActualSize = self.getAppConfig("VirtualScreenWidthActualSize")
        self.VirtualScreenHeightActualSize = self.getAppConfig("VirtualScreenHeightActualSize")
        self.projectorOnMonitor = self.getAppConfig("projectorOnMonitor")
        self.camera_control = self.getAppConfig("cameraControl", "str")
        self.data_path = self.getAppConfig("data_path", "str")
        self.image_file_type = self.getAppConfig("image_file_type", "str")

        self.split_rate = self.getAppConfig("split_rate")
        self.run_start_time = None

        self.deltaX = 0
        self.deltaY = 0

        self.positionDegreesToVSTable = []
        self.calcConvertPositionToPixelsTable()  # populate the above table
        self.queue_reader = None
        self.queue_writer = None
        self.camera = None
        self.writer_process1 = None
        self.writer_process2 = None
        self.file_prefix = None

        screen.geometry("+800+800")

        screen.focus_force()
        if self.NiDaqPulseEnabled.lower() == "on":
            # try to add channel
            task = None
            task2 = None
            try:
                task = NiDaqPulse(device_name="Dev2/port{0}/line{1}".format(self.port, self.stimulus_line))
                self.stimulus_output_device = task

            except nidaqmx.DaqError as e:
                print(e)
                if task:
                    task.stop()
        else:
            self.stimulus_output_device = None

        if self.camera_control.lower() == "on":
            self.setup_camera()

        if self.f9CommunicationEnabled.lower() == "on":
            self.f9CommunicationEnabled = False
        else:
            self.f9CommunicationEnabled = False
        logging.info("f9CommunicationEnabled=" + str(self.f9CommunicationEnabled))
        self.printVirtualScreenData()

        # presenting on the projector in maximize mode
        monitor = self.get_monitors_dimensions(self.projectorOnMonitor)
        screen.geometry(f"{monitor['width']}x{monitor['height']}+{screen.winfo_screenwidth()}+0")

        self.canvas = Canvas(screen, background="black")
        self.textVar = tkinter.StringVar()
        self.label = ttk.Label(self.canvas, textvariable=self.textVar)
        self.label.config(font=("Courier", 20))
        self.label.grid(column=0, row=0)
        self.canvas.pack(fill=BOTH, expand=1)
        self.vs = self.canvas.create_rectangle(self.vsX, self.vsY, self.vsX + self.vsWidth, self.vsY + self.vsHeight,
                                               fill=self.vsColor)
        screen.bind('<Up>', self.processEvent)
        screen.bind('<Down>', self.processEvent)
        screen.bind('<Left>', self.processEvent)
        screen.bind('<Right>', self.processEvent)
        screen.bind(constants.DEBUG, self.turnDebug)
        screen.bind(constants.LOCATION, self.changeControlMode)
        screen.bind(constants.SIZE, self.changeControlMode)
        screen.bind(constants.UPDATE, self.updateConfig)
        screen.bind(constants.EXIT, self.leaveProg)
        screen.bind(constants.RUN, self.manageStimulus)
        screen.bind(constants.PAUSE, self.manageStimulus)
        screen.bind(constants.CROSS, self.showCrossAndBoundries)
        screen.bind(constants.MID_SCREEN, self.showMidScreen)
        screen.bind('?', self.printHelp)
        self.printHelp("")

    """
    setup the camera, this requires
        getting the fish name 
        creating the workers and the queues 
    """

    def setup_camera(self):
        # get experiment prefix for file names etc.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fish_name = input(f"Enter fish name: ")
        self.file_prefix = f"{timestamp}_{fish_name}"
        self.data_path = f"{self.data_path}\\{self.file_prefix}"
        while fish_name == '' or not utils.create_directory(f"{self.data_path}"):
            fish_name = input(f"Fish name must be unique and not empty: ")
            self.file_prefix = f"{timestamp}_{fish_name}"
            self.data_path = f"{self.data_path}\\{self.file_prefix}"
        shutil.copyfile(constants.STIMULUS_CONFIG, f"{self.data_path}\\{constants.STIMULUS_CONFIG}")
        self.queue_reader = multiprocessing.Queue()  # communication queue to the worker
        self.queue_writer = multiprocessing.Queue()  # communication queue to the worker
        self.camera = multiprocessing.Process(name='camera_control_worker',  # Creation of the worker
                                              target=image_reader_worker.camera_control_worker,
                                              args=(
                                                  self.queue_reader, self.queue_writer, self.data_path,
                                                  self.file_prefix))
        self.writer_process1 = multiprocessing.Process(name='image_writer_worker1',
                                                       target=image_writer_worker.image_writer_worker,
                                                       args=(self.queue_writer, self.data_path, self.image_file_type))
        self.writer_process2 = multiprocessing.Process(name='image_writer_worker2',
                                                       target=image_writer_worker.image_writer_worker,
                                                       args=(self.queue_writer, self.data_path, self.image_file_type))

    def updateConfig(self, event):
        """
        Update config file with current VS settings
        """
        logging.debug(event)
        self.appConfig[0]["fishScreenStartX"] = self.vsX
        self.appConfig[0]["fishScreenStartY"] = self.vsY
        self.appConfig[0]["fishScreenWidth"] = self.vsWidth
        self.appConfig[0]["fishScreenHeight"] = self.vsHeight
        writeCSV(constants.APP_CONFIG_FILE, self.appConfig[0])
        self.printVirtualScreenData()
        logging.debug("Config file updates successfully")

    def leaveProg(self, event):
        logging.debug(event)
        logging.info("Bye bye !")
        if self.stimulus_output_device is not None:
            self.stimulus_output_device.stop()
        if self.camera:
            self.queue_reader.put('exit')
            self.camera.join()
            self.camera.terminate()
        if self.writer_process1:
            self.writer_process1.join()
            self.writer_process1.terminate()
        if self.writer_process2:
            self.writer_process2.join()
            self.writer_process2.terminate()
        self.screen.destroy()
        sys.exit()

    def printHelp(self, event):
        print("NOTE : press the keyboard while focus is on the graphic screen")
        print("========== General actions ==========")
        print('? = for help (this screen)')
        print(constants.EXIT, ' - to exit')
        print(constants.DEBUG, ' - to activate debug')
        print("========== Controlling Virtual Screen ==========")
        print(constants.CROSS, ' - show cross')
        print(constants.LOCATION, ' - switch to virtual screen location setting')
        print(constants.SIZE, ' - switch to virtual screen size setting ')
        print('<Up> - move up or reduce virtual screen height')
        print('<Down> - move down or extend virtual screen hegith')
        print('<Left>  - move left or reduce virtual screen width')
        print('<Right> - move right or extend virtual screen size ')
        print(constants.UPDATE, ' - to update app config with current virtual screen location')
        print("========== Controlling Stimulus generator ==========")
        print(constants.RUN, ' - Run the stimuli, pressing r again will restart')
        print(constants.PAUSE, ' - Pause the run')

    def changeControlMode(self, event):
        logging.debug(event)
        self.chagneVirtualScreenProperties(event.keysym)
        self.controlMode = event.keysym
        logging.debug("ControlMode=" + event.keysym)

    def processEvent(self, event):
        logging.debug(event)
        self.chagneVirtualScreenProperties(event.keysym)
        if self.controlMode == constants.LOCATION:
            self.canvas.move(self.vs, self.deltaX, self.deltaY)
            self.canvas.move(self.xBoundry, self.deltaX, self.deltaY)
            self.canvas.move(self.xHorizental, self.deltaX, self.deltaY)
            self.canvas.move(self.xVertical, self.deltaX, self.deltaY)
        elif self.controlMode == constants.SIZE:
            x0, y0, x1, y1 = self.canvas.coords(self.vs)
            x1 = x0 + self.vsWidth
            y1 = y0 + self.vsHeight
            self.canvas.coords(self.vs, x0, y0, x1, y1)
            if self.xMode:
                self.deleteCross()
                self.createCross()

        return

    def manageStimulus(self, event):
        logging.debug(event)

        if event.keysym == constants.PAUSE:
            self.state = constants.PAUSE
            if self.sg != "":
                self.sg.terminate_run()
            if self.camera:
                self.queue_reader.put('exit')
                print('EXIT SENT ')
                self.camera.join()
                self.camera = None  # this will allow restart
        elif event.keysym == constants.RUN:
            self.run_start_time = time.time()
            self.state = constants.RUN
            if self.camera:
                self.camera.start()
                self.writer_process1.start()
                self.writer_process2.start()

            self.sg = StimulusGenerator(self.canvas, self, self.stimulus_output_device, self.queue_reader)
            self.runStimuli()

    def runStimuli(self):  # this is main loop of stimulus
        if self.state == constants.PAUSE:
            return
        if self.sg.run_stimuli() == constants.DONE:  # call specific stimuli list
            logging.info("All stimuli were executed ! ")
            self.setDebugText("Done")
            self.state = constants.PAUSE

            if self.camera:
                self.writer_process1.join()
                self.writer_process2.join()
                self.camera = None  # to allow re-run
                # if we got here than all other processes are done (due to join) and we have a message waiting for us
                i, image_result = self.queue_writer.get()
                width = image_result[0]
                height = image_result[1]
                file_prefix = image_result[2]

                f = open(f'{self.data_path}\\create_video.bat', "a")
                f.write(
                    f"C:\\Users\\owner\\Documents\\Code\\StimulusGenerator\\venv\\Scripts\\python .\\create_video.py \n")
                f.write("pause\nexit\n")
                f.close()

                self.data_path = self.getAppConfig("data_path", "str")
                self.data_path = f"{self.data_path}\\\\{self.file_prefix}"


                f = open(f'{self.data_path}\\create_video.py', "a")
                f.write("import sys\n")
                f.write(f"sys.path.append('C:\\\\Users\\\\owner\\\\Documents\\\\Code\\\\StimulusGenerator')\n")
                f.write(f"from utils import opencv_create_video\n")
                f.write(
                    f"opencv_create_video('{file_prefix}', {height}, {width}, '{self.data_path}', '{self.image_file_type}')\n")
                f.close()

                # opencv_create_video(file_prefix, height, width, self.data_path, self.image_file_type)
        else:
            """We will need that only if we want to split files as they get too big
            if time.time() - self.run_start_time > self.split_rate:
                self.run_start_time = time.time()
                print(f"we need to split")"""
            self.canvas.after(constants.SLEEP_TIME, self.runStimuli)

    def calcGeometry(self, screen_width, screen_height):
        geometryStr = str(screen_width) + "x" + str(screen_height) + "+-10+0"
        return geometryStr

    def printVirtualScreenData(self):
        logging.debug("X=" + str(self.vsX) + " Y=" + str(self.vsY) + " Width=" + str(self.vsWidth) + " Height=" + str(
            self.vsHeight))

    def getAppConfig(self, variableName, type="int"):
        """get appConfig value

        Args:
            variableName ([string]): [the name of the variable]
            type ([type]): [int or str]
        """

        ret = self.appConfig[0][variableName]
        if type == "int":
            return (int(ret))
        else:
            return (ret)

    def showCrossAndBoundries(self, event):
        logging.debug(event)

        if self.xMode:
            self.xMode = False
            self.deleteCross()
        else:
            self.xMode = True
            self.createCross()

    def showMidScreen(self, event):
        logging.debug(event)
        monitor = self.get_monitors_dimensions(self.projectorOnMonitor)

        if self.yMode:
            self.yMode = False
            self.deleteMidScreen()
        else:
            self.createMidScreen(monitor['width'], monitor['height'])
            self.yMode = True

    def deleteCross(self):
        self.canvas.delete(self.xHorizental)
        self.canvas.delete(self.xVertical)
        self.canvas.delete(self.xBoundry)

    def deleteMidScreen(self):
        self.canvas.delete(self.yHorizental)
        self.canvas.delete(self.yVertical)

    def createMidScreen(self, width, height):

        self.yVertical = self.canvas.create_line(width / 2, 0,
                                                 width / 2, height,
                                                 fill=constants.FILL_COLOR,
                                                 width=constants.THIN_LINE_WIDTH)

        self.yHorizental = self.canvas.create_line(0, height / 2,
                                                   width, height / 2,
                                                   fill=constants.FILL_COLOR,
                                                   width=constants.THIN_LINE_WIDTH)

    def createCross(self):
        self.xBoundry = self.canvas.create_rectangle(self.vsX - 10, self.vsY - 10, self.vsX + self.vsWidth + 10,
                                                     self.vsY + self.vsHeight + 10, fill=constants.FILL_COLOR)
        self.xHorizental = self.canvas.create_line(self.vsX, self.vsY + self.vsHeight / 2, self.vsX + self.vsWidth,
                                                   self.vsY + self.vsHeight / 2, fill=constants.FILL_COLOR,
                                                   width=constants.LINE_WIDTH)
        self.xVertical = self.canvas.create_line(self.vsX + self.vsWidth / 2, self.vsY, self.vsX + self.vsWidth / 2,
                                                 self.vsY + self.vsHeight, fill=constants.FILL_COLOR,
                                                 width=constants.LINE_WIDTH)
        self.canvas.tag_lower(self.xBoundry)

    def chagneVirtualScreenProperties(self, direction):
        """
        Change screen Location or Size properties based on controlMode

        Args:
            direction ([String]): the event.keySym Up,Down,Left,Rigth
        """
        # we need delta X and Y as tkinter move is relative to the existing locaiton whereas teh vsX and vsY are absolute 
        self.deltaX = 0
        self.deltaY = 0

        if self.controlMode == constants.LOCATION:
            if direction.lower() == "up":
                self.deltaY -= 1
                self.vsY -= 1
            elif direction.lower() == "down":
                self.deltaY += 1
                self.vsY += 1
            elif direction.lower() == "left":
                self.deltaX -= 1
                self.vsX -= 1
            elif direction.lower() == "right":
                self.deltaX += 1
                self.vsX += 1
        elif self.controlMode == constants.SIZE:
            if direction.lower() == "up":
                self.vsHeight -= 1
            elif direction.lower() == "down":
                self.vsHeight += 1
            elif direction.lower() == "left":
                self.vsWidth -= 1
            elif direction.lower() == "right":
                self.vsWidth += 1

    def convertDegreesToMM(self, degrees):
        return 2 * self.DishRadiusSize * sin(degrees * (pi / 180) / 2)

    def convertMMToPixels(self, mm, direction):
        if direction.lower() == "width":
            return (mm * self.vsWidth) / self.VirtualScreenWidthActualSize
        else:
            return (mm * self.vsHeight) / self.VirtualScreenHeightActualSize

    def convertDegreestoPixels(self, degrees, direction):
        """[summary]

        Args:
            degrees ([int]): [degrees to be convereted]
            direction ([string]): [width or height]
        """
        return (self.convertMMToPixels(self.convertDegreesToMM(degrees), direction))

    def setDebugText(self, str):
        if self.debug:
            s = f'VS WxH {self.vsWidth} X {self.vsHeight}'
            s1 = f'1 degree/ms (LtR) = {round(self.convertDegreestoPixels(1, "widht"), 1)} pixels/ms'
            self.textVar.set(s + '\n' + s1 + '\n' + str)

    def turnDebug(self, event):
        if self.debug:
            logging.info("disable debug")
            self.debug = False
            self.label.grid_remove()
        else:
            logging.info("enable debug")
            self.debug = True
            self.label.grid()

    def get_monitors_dimensions(self, monitor_number):
        monitors = []
        for m in get_monitors():
            print(f'monitor = {m}')
            monitors.append({"width": m.width,
                             "height": m.height})
        return monitors[monitor_number]

    def calcConvertPositionToPixelsTable(self):
        for i in range(0, 181):
            if i <= 90:
                d = 90 - i
                v = 500 - 500 * sin(radians(d)) * cos(radians(d / 2)) / sin(radians(90 - d / 2))
            else:
                d = i - 90
                v = 500 + 500 * sin(radians(d)) * cos(radians(d / 2)) / sin(radians(90 - d / 2))
            self.positionDegreesToVSTable.append(v)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    root = Tk()
    root.title("Shacoof fish Stimuli Generator ")
    utils.dark_title_bar(root)
    app = App(root)
    root.mainloop()

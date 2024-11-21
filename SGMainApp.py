from tkinter.ttk import tkinter
import queue
import threading
from closed_loop_process.closed_loop_support import StimuliGeneratorClosedLoop, start_closed_loop_background
import shutil
import datetime
from tkinter import Canvas, Tk, BOTH
from tkinter import ttk
import nidaqmx
from screeninfo import get_monitors
from NiDaq.NiDaqPulse import NiDaqPulse
from math import pi, cos
import sys
from Stimulus.StimuliGenerator import *
import multiprocessing
from config_files.closed_loop_config import camera_emulator_on, emulator_with_camera
from utils.utils import *
from closed_loop_process.calibration.calibrate import Calibrator
from camera import image_reader_worker, image_writer_worker
import constants
import psutil
from camera.camera_emulator import camera_emulator_function


class App:
    sg = ""

    def __init__(self, screen, calibrator):
        # Init vars
        self.pca_and_predict = None
        self.image_processor = None
        self.bout_recognizer = None
        self.tail_tracker = None
        self.calibrator = calibrator
        self.already_calibrated = False
        self.screen = screen
        self.state = None
        self.multiprocess_state_is_running = multiprocessing.Value('b', False)  # Initial value is False
        self.stimulus_output_device = None
        self.port = 1
        self.stimulus_line = 7
        self.camera_line = 6
        self.debug = False
        self.controlMode = "l"  # l = location, s = size
        self.xMode = False  # false - no x is displayed in the middle of the screen
        self.xVertical = 0
        self.xHorizental = 0
        self.xBoundry = 0
        self.yMode = False  # False - no midScreen marker is displayed
        self.yVertical = 0
        self.yHorizental = 0
        self.bgColor = 0
        self.vsColor = 0
        self.stimulusColor = 0
        self.run_start_time = None
        self.deltaX = 0
        self.deltaY = 0
        self.positionDegreesToVSTable = []
        self.calcConvertPositionToPixelsTable()  # populate the above table
        self.queue_reader = None
        self.queue_writer = None
        self.closed_loop_process = None
        self.images_queue = None
        self.camera = None
        self.writer_process1 = None
        self.writer_process2 = None
        self.file_prefix = None
        self.trial_num = 0

        # Load App Config
        self.load_app_config()

        # Init screen
        self.screen.geometry("+800+800")
        self.screen.focus_force()
        monitor = self.get_monitors_dimensions(self.projectorOnMonitor)
        self.screen.geometry(f"{monitor['width']}x{monitor['height']}+{screen.winfo_screenwidth()}+0")
        self.canvas = Canvas(screen, background="black")
        self.textVar = tkinter.StringVar()
        self.label = ttk.Label(self.canvas, textvariable=self.textVar)
        self.label.config(font=("Courier", 20))
        self.label.grid(column=0, row=0)
        self.canvas.pack(fill=BOTH, expand=1)
        self.vs = self.canvas.create_rectangle(self.vsX, self.vsY, self.vsX + self.vsWidth, self.vsY + self.vsHeight,
                                               fill=self.vsColor)

        # Init NiDaq
        self.init_NiDaq()

        # Init camera
        if self.camera_control.lower() == "on":
            self.get_fish_name()
            self.setup_camera()

        # Init f9communication
        self.init_f9_communication()

        # Bind buttons
        self.bind_buttons()


    def bind_buttons(self):
        self.screen.bind('<Up>', self.processEvent)
        self.screen.bind('<Down>', self.processEvent)
        self.screen.bind('<Left>', self.processEvent)
        self.screen.bind('<Right>', self.processEvent)
        self.screen.bind(constants.DEBUG, self.turnDebug)
        self.screen.bind(constants.LOCATION, self.changeControlMode)
        self.screen.bind(constants.SIZE, self.changeControlMode)
        self.screen.bind(constants.UPDATE, self.updateConfig)
        self.screen.bind(constants.EXIT, self.leaveProg)
        self.screen.bind(constants.RUN, self.manageStimulus)
        self.screen.bind(constants.PAUSE, self.manageStimulus)
        self.screen.bind(constants.CROSS, self.showCrossAndBoundries)
        self.screen.bind(constants.MID_SCREEN, self.showMidScreen)
        self.screen.bind('?', self.printHelp)
        self.printHelp("")

    def calibrate(self):
        stimuli_queue = queue.Queue()
        calibrating_thread = threading.Thread(target=self.start_calibrating, args=(), daemon=True)
        calibrating_thread.start()
        self.sg = StimuliGeneratorClosedLoop(self.canvas, self, calib_mode=True)
        self.runStimuliClosedLoop()
        while self.state == constants.RUN:
            self.canvas.update()  # This keeps the UI responsive during the loop
            time.sleep(0.0001)  # Small sleep to avoid busy-waiting and CPU overload
        self.sg.stop_stimulus()

    def start_calibrating(self):
        # Calibration process running in its own thread
        [self.pca_and_predict, self.image_processor, self.bout_recognizer,
         self.tail_tracker] = self.calibrator.start_calibrating()
        self.state = constants.PAUSE

    def init_f9_communication(self):
        if self.f9CommunicationEnabled.lower() == "on":
            self.f9CommunicationEnabled = False
        else:
            self.f9CommunicationEnabled = False
        logging.info("f9CommunicationEnabled=" + str(self.f9CommunicationEnabled))
        self.printVirtualScreenData()

    def init_NiDaq(self):
        if self.NiDaqPulseEnabled.lower() == "on":
            # try to add channel
            task = None
            try:
                task = NiDaqPulse(device_name="Dev2/port{0}/line{1}".format(self.port, self.stimulus_line))
                self.stimulus_output_device = task
            except nidaqmx.DaqError as e:
                print(e)
                if task:
                    task.stop()
        else:
            self.stimulus_output_device = None

    def load_app_config(self):
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
        self.closed_loop = self.getAppConfig("use_closed_loop", "str")
        self.split_rate = self.getAppConfig("split_rate")

    def get_fish_name(self):
        # get experiment prefix for file names etc.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fish_name = input(f"Enter fish name: ")
        self.file_prefix = f"{timestamp}_{fish_name}"
        self.data_path = f"{self.data_path}\\{self.file_prefix}\\trial_{self.trial_num}"
        while fish_name == '' or not create_directory(f"{self.data_path}"):
            fish_name = input(f"Fish name must be unique and not empty: ")
            self.file_prefix = f"{timestamp}_{fish_name}"
            self.data_path = f"{self.data_path}\\{self.file_prefix}"
        shutil.copyfile(constants.STIMULUS_CONFIG, f"{self.data_path}\\StimulusConfig.csv")

    def setup_camera(self):
        """
            setup the camera - creating the workers and the queues
        """
        self.queue_reader = multiprocessing.Queue()  # communication queue to the worker
        self.queue_writer = multiprocessing.Queue()  # communication queue to the worker
        if self.closed_loop.lower() == "on":
            self.images_queue = multiprocessing.Queue()  # images to read in closed loop image processing
        if camera_emulator_on:
            self.camera = multiprocessing.Process(name='camera_control_worker',  # Creation of the worker
                                                  target=camera_emulator_function,
                                                  args=(self.queue_reader,
                                                        self.queue_writer,
                                                        self.images_queue))
        else:
            self.camera = multiprocessing.Process(name='camera_control_worker',  # Creation of the worker
                                                  target=image_reader_worker.camera_control_worker,
                                                  args=(
                                                      self.queue_reader, self.queue_writer, self.data_path,
                                                      self.file_prefix, self.images_queue, self.NiDaqPulseEnabled.lower() == "on"))
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
        self.stop_processes()
        self.screen.quit()
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
        print('<Down> - move down or extend virtual screen height')
        print('<Left>  - move left or reduce virtual screen width')
        print('<Right> - move right or extend virtual screen size ')
        print(constants.UPDATE, ' - to update app config with current virtual screen location')
        print("========== Controlling Stimulus generator ==========")
        print(constants.RUN, ' - Run the stimuli')
        print(constants.PAUSE, ' - Stop the run - press r again to restart')

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

    def stop_processes(self):
        self.multiprocess_state_is_running.value = False
        self.state = constants.PAUSE
        if self.sg != "":
            self.sg.terminate_run()
            if self.closed_loop_process:
                self.sg.save_csv(self.data_path)
        if self.camera:
            self.queue_reader.put('exit')
            self.camera.join(timeout=1)
            self.camera.terminate()
        if self.writer_process1:
            self.writer_process1.join(timeout=1)
            self.writer_process1.terminate()
        if self.writer_process2:
            self.writer_process2.join(timeout=1)
            self.writer_process2.terminate()
        if self.closed_loop_process:
            self.closed_loop_process.join(timeout=1)
            self.closed_loop_process.terminate()

    def setup_closed_loop_process(self):
        # Start the closed-loop process
        self.queue_closed_loop_prediction = multiprocessing.Queue()  # communication queue to the worker
        self.closed_loop_process = multiprocessing.Process(
            target=start_closed_loop_background,
            args=(self.images_queue, self.multiprocess_state_is_running, self.pca_and_predict, self.bout_recognizer,
                  self.tail_tracker, self.image_processor, self.queue_closed_loop_prediction))

    def manageStimulus(self, event):
        logging.debug(event)
        if event.keysym == constants.PAUSE:
            self.stop_processes()
            self.trial_num += 1
            self.data_path =self.data_path[0:self.data_path.rfind("_") + 1] + str(self.trial_num)
            create_directory(f"{self.data_path}")
        elif event.keysym == constants.RUN and self.state != constants.RUN:
            if self.closed_loop.lower() == "on" and not self.already_calibrated:
                self.state = constants.RUN
                self.calibrate()
                self.already_calibrated = True
            self.run_start_time = time.time()
            self.state = constants.RUN
            self.multiprocess_state_is_running.value = True
            if self.camera_control.lower() == "on":
                self.setup_camera()
                self.camera.start()
                self.writer_process1.start()
                self.writer_process2.start()
                process1_psutil = psutil.Process(self.camera.pid)
                process2_psutil = psutil.Process(self.writer_process1.pid)
                process3_psutil = psutil.Process(self.writer_process2.pid)
                process1_psutil.cpu_affinity([0])
                process2_psutil.cpu_affinity([1])
                process3_psutil.cpu_affinity([2])

            if self.closed_loop.lower() == "on":
                # save preprocess_config.py
                with open("config_files/preprocess_config.py", 'r') as source_file:
                    content = source_file.read()
                with open(self.data_path + "/preprocess_config.txt", 'w') as target_file:
                    target_file.write(content)
                self.setup_closed_loop_process()
                self.closed_loop_process.start()  # Start the process in the background
                process4_psutil = psutil.Process(self.closed_loop_process.pid)
                process4_psutil.cpu_affinity([3])
                self.sg = StimuliGeneratorClosedLoop(self.canvas, self, self.queue_closed_loop_prediction,
                                                     self.stimulus_output_device)
                self.runStimuliClosedLoop()
            else:
                self.sg = StimulusGenerator(self.canvas, self, self.stimulus_output_device, self.queue_reader)
                self.runStimuli()

    def runStimuliClosedLoop(self):
        if self.state != constants.RUN:
            return
        self.sg.run_stimuli_closed_loop()
        self.canvas.after(constants.SLEEP_TIME, self.runStimuliClosedLoop)

    def runStimuli(self):  # this is main loop of stimulus
        if self.state == constants.PAUSE:
            return
        if self.sg.run_stimuli() == constants.DONE:  # call specific stimuli list
            logging.info("All stimuli were executed ! ")
            self.setDebugText("Done")

            if self.camera:
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
                self.leaveProg("stimulus end")
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
    appConfig = loadCSV(constants.APP_CONFIG_FILE)
    closed_loop = appConfig[0]["use_closed_loop"]
    camera_control = appConfig[0]["cameraControl"]
    if closed_loop.lower() == "on" and camera_control.lower() != "on":
        raise RuntimeError("can't run closed loop without camera - edit the appConfig file")
    calibrator = None
    if closed_loop.lower() == "on":
        if camera_emulator_on or emulator_with_camera:
            calibrator = Calibrator(live_camera=False,
                                    start_frame=197751,
                                    end_frame=197751 + 500,
                                    images_path="\\\ems.elsc.huji.ac.il\\avitan-lab\Lab-Shared\Data\ClosedLoop\\20231204-f2\\raw_data"
                                    )
        else:
            calibrator = Calibrator(live_camera=True)
    root = Tk()
    root.title("Shacoof fish Stimuli Generator ")
    dark_title_bar(root)
    app = App(root, calibrator)
    root.mainloop()
    sys.exit()

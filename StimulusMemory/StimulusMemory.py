from config_files.closed_loop_config import start_angle, end_angle
from constants import *

# moving / part1 / part2 / spacer
MOVING = "moving"
PART1 = "part1"
PART2 = "part2"
SPACER = "spacer"

class StimulusMemory:
    def __init__(self, stimulus_struct):
        """
        """
        self.startX = int(stimulus_struct["startX"])
        self.startY = int(stimulus_struct["startY"])
        self.endX = int(stimulus_struct["endX"])
        self.endY = int(stimulus_struct["endY"])
        self.startShapeRadius = int(stimulus_struct["startShapeRadius"])
        self.endShapeRadius = int(stimulus_struct["endShapeRadius"])
        self.delay = int(stimulus_struct["delay"])
        self.repetitions = int(stimulus_struct["repetitions"])
        self.useSpacer = int(stimulus_struct["useSpacer"])

        self.rep = 0
        if self.repetitions == -1:
            self.repetitions = float('inf')

        self.current_angle = self.startX
        self.current_size = self.startShapeRadius
        self.state = "inactive" # moving / part1 / part2 / spacer
        self.stimulus_obj = None
        self.time_ms = 0
        self.max_size = 18

    def calc_new_angle_and_size(self,angle,distance):
        if angle > 90:
            angle = 90
        if angle < -90:
            angle = -90
        self.current_angle = self.current_angle - angle
        if self.current_angle > 165 or self.current_angle < 15:
            return "angle" # end of trial
        if distance > 3:
            distance = 3
        if distance <= 0:
            distance = 0
        self.current_size = self.current_size + 2 * distance
        if self.current_size  > self.max_size:
            return "size" # end of trial
        return round(self.current_angle), round(self.current_size)


    def reset_loc(self, angle, size):
        self.current_angle = angle
        self.current_size = size

    def update(self):
        self.time_ms += 1
        if self.stimulus_obj: # is this stimulus initiated
            if self.stimulus_obj.status == RUNNING:
                self.stimulus_obj.move()
            else:
                self.rep += 1
                if self.rep <= self.repetitions:
                    # either go to spacer or to symmetric stimuli
                    if self.useSpacer:
                        pass # init spacer
                    else:
                        pass # init symmetric struct
        else:
            # see if we need to initiate this stimulus
            if self.delay <= self.time_ms:
                pass # init stim struct

    def _init_spacer(self):
        pass

    def _init_symmetric_stimulus(self):
        pass

    def _init_stimulus(self):
        pass








    def move(self, moving_angle, moving_dist):








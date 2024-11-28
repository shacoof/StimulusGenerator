from constants import *
from Stimulus.Stimulus import Stimulus
from utils.utils import polar_to_cartesian, signed_angle_between_vectors
import math
from config_files.closed_loop_config import stimuli_moving_time, cut_velocity_by_factor

MOVING = "moving"
PRESENTED = "presented"
SPACER = "spacer"
INACTIVE = "inactive"
DONE = "done"
OOR = "out of range"

DYNAMIC = "dynamic"
STATIC = "static"


class StimulusMemory:
    def __init__(self, canvas, app, stimulus_struct, stim_id):
        """
        """
        self.type = stimulus_struct["Type"]
        self.startX = int(stimulus_struct["startX"])
        self.startY = int(stimulus_struct["startY"])
        self.endX = int(stimulus_struct["endX"])
        self.endY = int(stimulus_struct["endY"])
        self.useAfterFirstMovement = stimulus_struct["useAfterFirstMovement"]
        self.startShapeRadius = int(stimulus_struct["startShapeRadius"])
        self.endShapeRadius = int(stimulus_struct["endShapeRadius"])
        self.delay = int(stimulus_struct["delay"])
        self.repetitions = int(stimulus_struct["repetitions"])
        self.useSpacer = stimulus_struct["useSpacer"]
        self.duration = int(stimulus_struct["duration"])
        self.useAfterFirstMovement = stimulus_struct["useAfterFirstMovement"]
        self.first_movement_occurred = False
        self.stimuli_speed = self.calc_angular_velocity(self.startX, self.endX, self.duration)
        self.spacerDuration = int(stimulus_struct["spacerDuration"])
        self.original_struct = stimulus_struct
        self.canvas = canvas
        if self.startX >= self.endX:
            self.direction_left_to_right = True
        else:
            self.direction_left_to_right = False
        if self.startX != self.endX and self.type == STATIC:
            raise ValueError("static type need to have the same start and end")
        self.app = app
        self.rep = 1
        self.stim_id = stim_id
        if self.repetitions == -1:
            self.repetitions = float('inf')
        self.current_angle = self.startX
        self.current_size = self.startShapeRadius
        self.state = INACTIVE
        self.stimulus_obj = None
        self.time_ms = 0
        self.max_size = 18
        self.min_size = 1
        self.size_to_mm_ratio = 4  # 1 mm the fish advances is equivalent to 4 units in size of stimulus
        self.distance_from_fish = self.size_to_dist_from_fish(self.current_size)

    def calc_new_angle_and_size(self, angle, distance):
        if angle > 90:
            angle = 90
        if angle < -90:
            angle = -90
        if distance > 3:
            distance = 3
        if distance <= 0:
            distance = 0
        # calc size and angle in reference to this stimuli
        angle_fish_stim, distance_fish_stim = self._calc_angle_dist_to_fish(angle, distance, self.current_angle,
                                                                            self.size_to_dist_from_fish(
                                                                                self.current_size))
        self.current_angle = angle_fish_stim
        self.current_size = self.dist_from_fish_to_size(distance_fish_stim)
        return round(self.current_angle), round(self.current_size)

    def is_stim_in_angle_range(self):
        if 0 < self.current_angle < 180:
            return True
        return False

    def is_stim_too_big(self):
        if self.current_size > self.max_size:
            return True
        return False

    def is_stim_too_small(self):
        if self.current_size < self.min_size:
            return True
        return False

    @staticmethod
    def _calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance):
        fish_angle = 90 + fish_angle
        fish_x, fish_y = polar_to_cartesian(fish_angle, fish_distance)
        # angle
        stimulus_orig_x, stimulus_orig_y = polar_to_cartesian(stim_angle, stim_distance)
        stimulus_new_x, stimulus_new_y = stimulus_orig_x - fish_x, stimulus_orig_y - fish_y
        stimulus_vec = [stimulus_new_x, stimulus_new_y]
        fish_vec = [fish_x, fish_y]
        angle_fish_stim = signed_angle_between_vectors(stimulus_vec, fish_vec)
        # distance
        distance_fish_stim = math.sqrt((fish_x - stimulus_orig_x) ** 2 + (fish_y - stimulus_orig_y) ** 2)
        angle_fish_stim = angle_fish_stim + 90
        if angle_fish_stim > 360:
            angle_fish_stim = angle_fish_stim - 360
        return angle_fish_stim, distance_fish_stim

    def update(self):
        print(self.state)
        if self.state == DONE:
            return ""
        self.time_ms += 1
        return_str = ""
        if self.stimulus_obj:  # is this stimulus initiated
            self.current_angle = self.stimulus_obj.current_degree
            self.current_size = self.stimulus_obj.currRadius
            if not self.is_stim_in_angle_range() and self.type == DYNAMIC and self.state in [MOVING, PRESENTED]:
                self.state = DONE
                self.stop_stimulus()
                #self.stimuliGenerator.send_pulse_and_write_log(f"stimuli {self.stim_id}", "end", "NA", "NA", "out of angle range")
                return "DONE: out of range"
            if self.is_stim_too_big() and self.state in [MOVING, PRESENTED]:
                self.state = DONE
                self.stop_stimulus()
                return "DONE: too big"
            if self.is_stim_too_small() and self.state in [MOVING, PRESENTED]:
                self.state = DONE
                self.stop_stimulus()
                return "DONE: too small"

            if self.stimulus_obj.status == RUNNING:
                self.stimulus_obj.move()
            else:
                self.stop_stimulus()
                if self.state == MOVING:
                    self._init_after_moving_stimulus()
                    return_str =  "moving stop"
                elif self.state == PRESENTED:
                    # self.stimuliGenerator.send_pulse_and_write_log(f"stimuli {self.stim_id}", "end", "NA",
                    #                                                "NA", f"finished rep {self.rep}")
                    self.rep += 1
                    if self.rep <= self.repetitions:

                        # either go to spacer or to symmetric stimuli
                        if self.useSpacer:
                            self._init_spacer()
                            self.state = SPACER
                        else:
                            self._init_symmetric_stimulus()
                            #self.stimuliGenerator.send_pulse_and_write_log(f"stimuli {self.stim_id}", "start", "NA",                                                  "NA", f"starting rep {self.rep}")
                    else:
                        self.state = DONE
                        return "DONE: finished all reps"
                elif self.state == SPACER:
                    # self.stimuliGenerator.send_pulse_and_write_log(f"stimuli {self.stim_id}", "start", "NA",
                    #                                                "NA", f"starting rep {self.rep}")
                    self._init_symmetric_stimulus()
                    self.state = PRESENTED
                self.start_stimulus()
        else:
            # see if we need to initiate this stimulus
            if self.delay <= self.time_ms and self.repetitions > 0:
                if not self.first_movement_occurred or self.useAfterFirstMovement == "TRUE":
                    # self.stimuliGenerator.send_pulse_and_write_log(f"stimuli {self.stim_id}", "start", "NA",
                    #                                                "NA", f"starting rep {self.rep}")
                    self._init_stimulus()
                    self.state = PRESENTED
                    self.start_stimulus()
        return return_str

    def _init_spacer(self):
        spacer_struct = {
            "exitCriteria": "Spacer", "startX": '0', "startY": '0', "endX": '0', "endY": '0', "repetitions": '1',
            "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": '4', "endShapeRadius": '4', "fastDuration": '0',
            "slowDuration": '0', "startMode": "WITH", "delay": '0', "duration": self.spacerDuration,
            "xType": "degrees"}
        self.stimulus_obj = Stimulus(spacer_struct, self.canvas, self.app, self.stim_id)

    def _init_moving_stimulus(self, new_angle, new_size):
        current_y = self.startY
        stimulus_struct = {
            "exitCriteria": "Time", "startX": str(self.current_angle), "startY": str(current_y), "endX": str(new_angle),
            "endY": str(current_y),
            "repetitions": '1', "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": str(self.current_size),
            "endShapeRadius": str(new_size),
            "fastDuration": '100', "slowDuration": '100', "startMode": "WITH", "delay": '0', "duration": stimuli_moving_time,
            "xType": "degrees"
        }
        self.stimulus_obj = Stimulus(stimulus_struct, self.canvas, self.app, self.stim_id)

    def _init_after_moving_stimulus(self):
        current_y = self.startY
        start_angle = self.stimulus_obj.current_degree
        start_size = self.stimulus_obj.currRadius
        if self.type == STATIC:
            stimulus_struct = {
                "exitCriteria": "Time", "startX": str(start_angle), "startY": str(current_y), "endX": str(start_angle),
                "endY": str(current_y),
                "repetitions": '1', "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": str(start_size),
                "endShapeRadius": str(start_size),
                "fastDuration": '100', "slowDuration": '100', "startMode": "WITH", "delay": '0', "duration": "10000",
                "xType": "degrees"
            }
        else:
            self.stimuli_speed = self.stimuli_speed * cut_velocity_by_factor
            duration = self.calc_duration(self.startX, self.endX, self.stimuli_speed)
            end_angle = 180
            if self.direction_left_to_right:
                end_angle = 0
            stimulus_struct = {
                "exitCriteria": "Time", "startX": str(start_angle), "startY": str(current_y), "endX": str(end_angle),
                "endY": str(current_y),
                "repetitions": '1', "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": str(start_size),
                "endShapeRadius": str(start_size),
                "fastDuration": '100', "slowDuration": '100', "startMode": "WITH", "delay": '0', "duration": duration,
                "xType": "degrees"
            }
        self.stimulus_obj = Stimulus(stimulus_struct, self.canvas, self.app, self.stim_id)

    def _init_symmetric_stimulus(self):
        if self.direction_left_to_right:
            self._init_stimulus()
        else:
            stimulus_struct = {
                "exitCriteria": "Time", "startX": str(self.endX), "startY": str(self.startY), "endX": str(self.startX),
                "endY": str(self.startY),
                "repetitions": '1', "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": str(self.endShapeRadius),
                "endShapeRadius": str(self.startShapeRadius),
                "fastDuration": '100', "slowDuration": '100', "startMode": "WITH", "delay": '0',
                "duration": self.duration,
                "xType": "degrees"
            }
            self.stimulus_obj = Stimulus(stimulus_struct, self.canvas, self.app, self.stim_id)
        self.direction_left_to_right = not self.direction_left_to_right

    def _init_stimulus(self):
        stimulus_struct = {
            "exitCriteria": "Time", "startX": str(self.startX), "startY": str(self.startY), "endX": str(self.endX),
            "endY": str(self.startY),
            "repetitions": '1', "fastSpeed": '0', "slowSpeed": '0', "startShapeRadius": str(self.startShapeRadius),
            "endShapeRadius": str(self.endShapeRadius),
            "fastDuration": '100', "slowDuration": '100', "startMode": "WITH", "delay": '0', "duration": self.duration,
            "xType": "degrees"
        }
        self.stimulus_obj = Stimulus(stimulus_struct, self.canvas, self.app, self.stim_id)

    @staticmethod
    def calc_duration(start_angle, end_angle, angular_velocity):
        duration = str(round(abs(end_angle - start_angle) / angular_velocity * 1000))
        return duration

    @staticmethod
    def calc_angular_velocity(start_angle, end_angle, duration):
        angular_velocity = str(round(abs(end_angle - start_angle) / duration * 1000))
        return angular_velocity

    def stop_stimulus(self):
        if self.stimulus_obj:
            self.stimulus_obj.terminate_run()
            self.stimulus_obj.status = DONE  # Set the status to DONE

    def start_stimulus(self):
        self.stimulus_obj.init_shape(0)
        print(f"New {self.state} stimulus added with ID: {self.stim_id}")

    def move(self, moving_angle, moving_dist):
        if self.state != INACTIVE and self.state != SPACER and self.stimulus_obj:
            res = self.calc_new_angle_and_size(moving_angle, moving_dist)
            new_angle, new_size = res
            self._init_moving_stimulus(new_angle, new_size)
            self.start_stimulus()

    def size_to_dist_from_fish(self, size):
        distance_from_fish = (self.max_size - size) / self.size_to_mm_ratio
        return distance_from_fish

    def dist_from_fish_to_size(self, dist):
        size = -self.size_to_mm_ratio * dist + self.max_size
        return size
#
#
# # testing angle, distance updates
# # stim , fish
# stim_angle = 90
# stim_distance = 1
# fish_angle = 180
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))
# stim_angle = 90
# stim_distance = 1
# fish_angle = 0
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))


# # q1, q1
# stim_angle = 20
# stim_distance = 1
# fish_angle = 40
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))
# # q2, q1
# stim_angle = 120
# stim_distance = 2
# fish_angle = 40
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))
# # q3, q1
# stim_angle = 200
# stim_distance = 2
# fish_angle = 40
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))
# # q4, q1
# stim_angle = 290
# stim_distance = 2
# fish_angle = 40
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))
# # q1, q2
# stim_angle = 40
# stim_distance = 2
# fish_angle = 110
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))
# # q2, q2
# stim_angle = 170
# stim_distance = 2
# fish_angle = 110
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))
# # q3, q2
# stim_angle = 250
# stim_distance = 2
# fish_angle = 110
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))
# # q4, q2
# stim_angle = 300
# stim_distance = 2
# fish_angle = 110
# fish_distance = 1
# print(StimulusMemory._calc_angle_dist_to_fish(fish_angle, fish_distance, stim_angle, stim_distance))

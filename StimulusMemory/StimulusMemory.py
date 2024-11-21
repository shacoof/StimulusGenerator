
class StimulusMemory:
    def __init__(self, stimulus_struct):
        """
        Translates the bout distance and angle to stimuli size angle in the range of 15-165 degrees
        """
        self.startX = int(stimulus_struct["startX"])
        self.startY = int(stimulus_struct["startY"])
        self.endX = int(stimulus_struct["endX"])
        self.endY = int(stimulus_struct["endY"])
        self.startShapeRadius = int(stimulus_struct["startShapeRadius"])
        self.endShapeRadius = int(stimulus_struct["endShapeRadius"])
        self.delay = int(stimulus_struct["delay"])
        self.rep = 1
        self.repetitions = int(stimulus_struct["repetitions"])
        self.current_angle = int(stimulus_struct["startX"])
        self.current_size = int(stimulus_struct["startShapeRadius"])
        self.state = "inactive"
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
        self.current_size = self.current_angle + 2 * distance
        if self.current_size  > self.max_size:
            return "size" # end of trial
        return round(self.current_angle), round(self.current_size)


    def reset_loc(self, angle, size):
        self.current_angle = angle
        self.current_size = size






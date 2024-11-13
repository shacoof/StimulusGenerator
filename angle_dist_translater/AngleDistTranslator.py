
class AngleDistTranslator:
    def __init__(self, start_angle, start_size):
        """
        Translates the bout distance and angle to stimuli size angle in the range of 15-165 degrees
        """
        self.current_angle = start_angle
        self.current_size = start_size
        self.current_distance_to_prey = 8


    def calc_new_angle_and_size(self,angle,distance):
        if angle > 90:
            angle = 90
        if angle < -90:
            angle = -90
        self.current_angle = self.current_angle - angle
        if self.current_angle > 165 or self.current_angle < 15:
            return None # end of trial
        if distance > 3:
            distance = 3
        if distance <= 0:
            distance = 0
        self.current_distance_to_prey = self.current_distance_to_prey - distance
        if self.current_distance_to_prey < 0:
            return None # end of trial
        else:
            self.current_size = round(self.current_distance_to_prey * (-7/4) + 18) # rescale to range of 18 to 4
        return self.current_angle, self.current_size


    def reset_food(self, angle, size):
        # wait X seconds
        self.current_angle = angle
        self.current_size = size
        self.current_distance_to_prey = 8






from cgitb import reset


class Renderer:
    def __init__(self):
        """
        Renders stimuli
        """
        self.current_angle = 90
        self.current_size = 4
        self.current_distance_to_prey = 8


    def calc_new_angle_and_size(self,angle,distance,VirtualScreenDegrees= 170):
        if angle > 90:
            angle = 90
        if angle < -90:
            angle = -90
        self.current_angle = self.current_angle + angle
        if self.current_angle > 165:
            self.current_angle = 165
        if self.current_angle < 15:
            self.current_angle = 15
        if distance > 3:
            distance = 3
        if distance < 0:
            distance = 0
        self.current_distance_to_prey = self.current_distance_to_prey - distance
        if self.current_distance_to_prey < 0:
            self.reset_food()
        else:
            self.size = round(self.current_distance_to_prey * (-7/4) + 18) # rescale to range of 18 to 4
        return self.angle, self.distance


    def reset_food(self):
        # wait X seconds
        self.current_angle = 90
        self.current_size = 4
        self.current_distance_to_prey = 8






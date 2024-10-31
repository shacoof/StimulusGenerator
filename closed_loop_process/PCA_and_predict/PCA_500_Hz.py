from closed_loop_process.PCA_and_predict.abstract_PCA import AbstractPCA
import scipy.io
import numpy as np


class PCA500Hz(AbstractPCA):
    def __init__(self):
        super().__init__()
        self.prediction_matrix_angle = scipy.io.loadmat(
            "\\\ems.elsc.huji.ac.il\\avitan-lab\Lab-Shared\Data\ClosedLoop\movement_train_intermediate.mat")[
            'angle_solution']
        self.prediction_matrix_distance = scipy.io.loadmat(
            "\\\ems.elsc.huji.ac.il\\avitan-lab\Lab-Shared\Data\ClosedLoop\movement_train_intermediate.mat")[
            'distance_solution']
        self.number_of_frames_for_predict = 35

    def reduce_dimensionality_and_predict(self, theta_mat, to_plot):
        super().reduce_dimensionality_and_predict(theta_mat, to_plot)
        reshaped_angle = self.reduced_dim[0:30].T.reshape(1, 90)
        angle = reshaped_angle @ self.prediction_matrix_angle
        reshaped_distance = self.reduced_dim[9:35].T.reshape(1, 78)
        distance = np.square(reshaped_distance) @ self.prediction_matrix_distance
        return round(angle[0][0], 2), round(distance[0][0], 2)
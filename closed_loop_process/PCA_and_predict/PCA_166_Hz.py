from closed_loop_process.PCA_and_predict.abstract_PCA import AbstractPCA
import scipy.io
import numpy as np

class PCA166Hz(AbstractPCA):
    def __init__(self):
        super().__init__()
        self.prediction_matrix_angle = \
        scipy.io.loadmat("\\\ems.elsc.huji.ac.il\\avitan-lab\Lab-Shared\Data\ClosedLoop\B_matrices_slow_imri.mat")[
            'angle_solution']
        self.prediction_matrix_distance = \
        scipy.io.loadmat("\\\ems.elsc.huji.ac.il\\avitan-lab\Lab-Shared\Data\ClosedLoop\B_matrices_slow_imri.mat")[
            'distance_solution']
        self.number_of_frames_for_predict = 12


    def reduce_dimensionality_and_predict(self, theta_mat, to_plot):
        super().reduce_dimensionality_and_predict(theta_mat, to_plot)
        reshaped_angle = self.reduced_dim[0:10, :].T.reshape(1, 30)
        angle = reshaped_angle @ self.prediction_matrix_angle
        reshaped_distance = self.reduced_dim[3:12, :].T.reshape(1, 27)
        distance = np.square(reshaped_distance) @ self.prediction_matrix_distance
        return round(angle[0][0],2), round(distance[0][0],2)
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.linalg import svd


class AbstractPCA(ABC):
    def __init__(self):
        self.V = scipy.io.loadmat("closed_loop_process/prediction_mats/V.mat")['V']
        self.S = scipy.io.loadmat("closed_loop_process/prediction_mats/S.mat")['S']
        self.prediction_matrix_angle = None
        self.prediction_matrix_distance = None
        self.number_of_frames_for_predict = None
        self.reduced_dim = None
        # calculate projection mat
        diagonal_elements = np.diag(self.S)
        self.S = diagonal_elements
        VT = self.V.T
        VT[:3, :] = VT[:3, :] * np.sign(VT[:3, 0])[:, np.newaxis]
        projection_matrix = VT / self.S[:, np.newaxis]  # the projection matrix
        self.projection_matrix = projection_matrix[:3, :].T


    @staticmethod
    def clean_nan_from_data(data, remove_nan=False):
        if remove_nan:
            mask = ~np.all(np.isnan(data), axis=1)
            # Apply the mask to filter out the rows with all None
            data = data[mask]
        else:  # take previous tail angles to replace with unrecognized frames
            # Step 1: Replace None with zeros for the first row if necessary
            if np.all(np.isnan(data[0])):
                data[0] = np.zeros(len(data[0]))
            # Step 2: Iterate through the matrix and replace None rows with previous row values
            for i in range(1, len(data[:, 1])):
                if np.all(np.isnan(data[i])):
                    data[i] = data[i - 1]
        return data

    @abstractmethod
    def reduce_dimensionality_and_predict(self, theta_mat, to_plot):
        """
        Args:
        theta_mat: a number_of_frames_for_predict x 98 matrix of tail angles (not normalized) for prediction
        to_plot: if true, plots the tail in time as a linear function of the 3 PCs and the coefficients
        Returns: predicted angle, distance
        """
        if theta_mat.shape[0] < self.number_of_frames_for_predict:
            raise RuntimeError(f"need {self.number_of_frames_for_predict} frames to predict")
        # normalize
        theta_mean = np.mean(theta_mat, axis=1)  # Mean angle per sample
        theta_mat = theta_mat - theta_mean[:, np.newaxis]  # Normalize by subtracting mean angle
        # clean nan values
        theta_matrix = AbstractPCA.clean_nan_from_data(theta_mat, remove_nan=False)
        self.reduced_dim = theta_matrix @ self.projection_matrix
        if to_plot:
            for i in range(self.number_of_frames_for_predict):
                self.visualize_predicted_tail(theta_matrix[i], i)
        # child classes multiply the reshaped theta mat with the prediction mat depending on fr


    def visualize_predicted_tail(self, tail_theta_vec, frame_num):
        num_PCs = 3  # number of Principal Components
        reduced_dim = tail_theta_vec @ self.projection_matrix

        dT = np.zeros(98)  # predicted theta vector
        for PC in range(num_PCs):
            dT += reduced_dim[PC] * self.S[PC] * self.V[:, PC]  # U is the coefficient of the PC

        dx = np.cos(dT)
        dy = np.sin(dT)
        all_x = np.cumsum(dx[::-1])
        all_y = np.cumsum(dy[::-1])

        # Set up the plot window
        plt.cla()  # Clear the current axes
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Plot of Predicted Tail - Frame {frame_num}')

        # Plotting the tail prediction
        plt.plot(all_x, all_y, linewidth=2, label=f'Frame {frame_num}')
        plt.legend()

        # Force the axis limits to stay fixed AFTER plotting
        plt.xlim(0, 100)  # Set fixed x-axis limits
        plt.ylim(-50, 50)  # Set fixed y-axis limits
        plt.gca().set_aspect('auto', adjustable='box')  # Ensure no forced equal aspect ratio

        # Update the plot without blocking
        plt.show(block=False)
        plt.pause(0.1)  # Pause briefly to ensure the plot window updates

    @staticmethod
    def plot_PCs(text, V):
        # x-values: Indices of the arrays
        plt.close()
        x_values = np.arange(1, 99)  # This creates an array from 1 to 98
        array1 = V[:,0]
        array2 = V[:,1]
        array3 = V[:,2]
        # Plotting the arrays
        plt.plot(x_values, array1, label='PC 1', color='blue', linewidth=2)
        plt.plot(x_values, array2, label='PC 2', color='green', linewidth=2)
        plt.plot(x_values, array3, label='PC 3', color='red', linewidth=2)

        # Adding labels and title
        plt.xlabel('Vector entry')
        plt.ylabel('Coefficient Value')
        plt.title('Plot of Three PCs')
        plt.legend()  # Show legend to distinguish between the arrays

        # Displaying the plot
        plt.savefig(f'{text}.png')  # Save the plot to a file
        plt.close()

    @staticmethod
    def plot_coefficients(reduced_dim, frame_number):
        # x-values: Indices of the arrays
        plt.close()
        x_values = np.arange(1, 36)  # This creates an array from 1 to 35
        array1 = reduced_dim[:, 0]
        array2 = reduced_dim[:, 1]
        array3 = reduced_dim[:, 2]
        # Plotting the arrays
        plt.plot(x_values, array1, label='coefficient 1', color='blue', linewidth=2)
        plt.plot(x_values, array2, label='coefficient 2', color='green', linewidth=2)
        plt.plot(x_values, array3, label='coefficient 3', color='red', linewidth=2)

        # Adding labels and title
        plt.xlabel('Vector entry')
        plt.ylabel('Coefficient Value')
        plt.title(f'Plot of Three PCs Coefficients frame: {frame_number}')
        plt.legend()  # Show legend to distinguish between the arrays

        # Displaying the plot
        plt.savefig(f'coefficient frame {frame_number}.png')  # Save the plot to a file
        plt.close()

    # not in use because we use predefined PCAs
    def calc_3_PCA(self, calibration_theta_matrix: np.array, plot_PC) -> None:
        """
        Updates U to be the 98x3 matrix whose columns are the PCs of the calibration data
        :param calibration_theta_matrix: numpy array of dimensions mx105x2 where m is the number of frames used in calibration and
         105 is the number of tail points
        :param plot_PC: is true, plots the PCs of the tail movements in the calibration_theta_matrix
        :return: None
        """
        self.calibration_theta_matrix = calibration_theta_matrix
        self.get_svd()
        if plot_PC:
            # imris
            V = scipy.io.loadmat('closed_loop_process/prediction_mats/V.mat')['V']
            AbstractPCA.plot_PCs("new_PCs", self.V)
            AbstractPCA.plot_PCs("imris_PCs", V)

    # not in use because we use predefined PCAs
    def get_svd(self, comp_num=3):
        [U, S, VT] = svd(self.calibration_theta_matrix, full_matrices=False)
        VT[:comp_num, :] = VT[:comp_num, :] * np.sign(VT[:comp_num, 0])[:, np.newaxis]
        projection_matrix = VT / S[:, np.newaxis]  # the projection matrix
        self.projection_matrix = projection_matrix[:comp_num, :].T
        self.S = S
        self.V = VT.T
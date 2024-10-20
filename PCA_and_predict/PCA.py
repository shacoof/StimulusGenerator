import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import scipy.io


class PCA:
    def __init__(self, prediction_matrix_angle=None, prediction_matrix_distance=None, V=None,
                 S=None, number_of_frames_for_predict = 35, plot_PC = False):
        """
        Either compute PCA from calibration data or provide input of projection_matrix
        :param prediction_matrix_angle: the already computed angle
        :param projection_matrix: the already computed principle component matrix 3x98
        """
        self.theta_matrix = None
        self.projection_matrix = None
        self.calibration_data = None
        self.prediction_matrix_angle = prediction_matrix_angle
        self.prediction_matrix_distance = prediction_matrix_distance
        self.V = V
        self.S = S
        self.number_of_frames_for_predict = number_of_frames_for_predict
        self.plot_PC = plot_PC

        if V is not None and S is not None:
            diagonal_elements = np.diag(S)
            self.S = diagonal_elements
            VT = V.T
            VT[:3, :] = VT[:3, :] * np.sign(VT[:3, 0])[:, np.newaxis]
            projection_matrix = VT / self.S[:, np.newaxis]  # the projection matrix
            self.projection_matrix = projection_matrix[:3, :].T



    def calc_3_PCA(self, calibration_data: np.array) -> None:
        """
        Updates U to be the 98x3 matrix whose columns are the PCs of the calibration data
        :param calibration_data: numpy array of dimensions mx105x2 where m is the number of frames used in calibration and
         105 is the number of tail points
        :return: None
        """
        self.calibration_data = calibration_data
        self.theta_matrix = PCA.calc_theta_matrix(calibration_data)
        self.theta_matrix = PCA.clean_nan_from_data(self.theta_matrix, remove_nan=True)
        self.get_svd()
        if self.plot_PC:
            # imris
            V = scipy.io.loadmat('Z:\Lab-Shared\Data\ClosedLoop\V.mat')['V']
            # S = scipy.io.loadmat('Z:\Lab-Shared\Data\ClosedLoop\S.mat')['S']
            # shais
            # V = scipy.io.loadmat("Z:\\adi.kishony\ClosedLoopPOC\saved_np_arrays\\20240916-f2_SVD.mat")['V']
            PCA.plot_PCs("adis_PCs", self.V)
            PCA.plot_PCs("imris_PCs", V)



    def get_svd(self, comp_num=3):
        [U, S, VT] = svd(self.theta_matrix, full_matrices=False)
        VT[:comp_num, :] = VT[:comp_num, :] * np.sign(VT[:comp_num, 0])[:, np.newaxis]
        projection_matrix = VT / S[:, np.newaxis]  # the projection matrix
        self.projection_matrix = projection_matrix[:comp_num, :].T
        self.S = S
        self.V = VT.T

    @staticmethod
    def calc_theta_matrix(tail_data):
        """
        Receives a mx105x2 matrix of tail points (or a single 105x2 array for a single sample)
        and returns a mx98 matrix of angles cutting the last points.
        :return: angles features matrix
        """
        extra_size = 6  # Trust issues with end-of-fit points
        # Determine if input is 2D or 3D
        if tail_data.ndim == 3:
            # 3D case: multiple samples
            tail_diff = np.diff(tail_data, axis=1)  # Difference along the points axis
            tail_line = tail_diff[:, :, 0] + 1j * tail_diff[:, :, 1]  # Complex representation of vector differences
            d_theta = np.angle(tail_line)  # Calculate angles
            d_theta = np.unwrap(d_theta, axis=1)[:, :-extra_size + 1]  # Unwrap and remove unreliable end points
            theta_mean = np.mean(d_theta, axis=1)  # Mean angle per sample
            theta_matrix = d_theta - theta_mean[:, np.newaxis]  # Normalize by subtracting mean angle
            theta_matrix = theta_matrix[:, 1:]  # Remove the first point to match desired output size
        elif tail_data.ndim == 2:
            # 2D case: single sample
            tail_diff = np.diff(tail_data, axis=0)  # Difference along the points axis
            tail_line = tail_diff[:, 0] + 1j * tail_diff[:, 1]  # Complex representation of vector differences
            d_theta = np.angle(tail_line)  # Calculate angles
            d_theta = np.unwrap(d_theta)[:-extra_size + 1]  # Unwrap and remove unreliable end points
            theta_mean = np.mean(d_theta)  # Mean angle
            theta_matrix = d_theta - theta_mean  # Normalize by subtracting mean angle
            theta_matrix = theta_matrix[1:]  # Remove the first point to match desired output size
            theta_matrix = theta_matrix[np.newaxis, :]  # Add new axis to make it 2D compatible
        return theta_matrix

    def reduce_dimensionality_and_predict(self, tail_data: np.array, to_plot, frame_number = "") -> np.array:
        """
        receives numpy array tail data with dimensions 30x105x2 where 30 is the number of frames, 105 are the
        interpolated tail points, and 2 are x,y coordinates
        :param tail_data: 30 consecutive frames for prediction
        :return: angle and distance of the fish predicted for these frames
        """
        if self.projection_matrix is None:
            raise RuntimeError("need to run calc_3_PCA first")
        if tail_data.shape[0] < self.number_of_frames_for_predict:
            raise RuntimeError(f"need {self.number_of_frames_for_predict} frames to predict")
        tail_data_theta_mat = PCA.calc_theta_matrix(tail_data)
        tail_data_theta_mat = PCA.clean_nan_from_data(tail_data_theta_mat, remove_nan=False)
        reduced_dim = tail_data_theta_mat @ self.projection_matrix
        if to_plot:
            self.plot_coefficients(reduced_dim, frame_number)
            for i in range(self.number_of_frames_for_predict):
                self.visualize_predicted_tail(tail_data_theta_mat[i], i)
        #fr 166
        reshaped_angle = reduced_dim[0:10,:].T.reshape(1, 30)
        angle = reshaped_angle @ self.prediction_matrix_angle
        reshaped_distance = reduced_dim[3:12, :].T.reshape(1, 27)
        distance = np.square(reshaped_distance) @ self.prediction_matrix_distance
        # regular
        # reshaped_angle = reduced_dim[0:30].T.reshape(1, 90)
        # angle = reshaped_angle @ self.prediction_matrix_angle
        # reshaped_distance = reduced_dim[9:35].T.reshape(1, 78)
        # distance = np.square(reshaped_distance) @ self.prediction_matrix_distance
        # using late frames in the prediction
        # reshaped_angle = reduced_dim[0:27].T.reshape(1, 81)
        # new_angle = np.concatenate((self.prediction_matrix_angle[3:30], self.prediction_matrix_angle[33:60], self.prediction_matrix_angle[63:90]))
        # angle = reshaped_angle @ new_angle
        # reshaped_distance = reduced_dim[9:32].T.reshape(1, 69)
        # new_distance = np.concatenate((self.prediction_matrix_distance[3:26], self.prediction_matrix_distance[29:52], self.prediction_matrix_distance[55:78]))
        # distance = np.square(reshaped_distance) @ new_distance
        return angle[0][0], distance[0][0]

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

    def visualize_predicted_tail(self, tail_theta_vec, frame_num):
        num_PCs = 3  # number of PCs
        reduced_dim = tail_theta_vec @ self.projection_matrix

        dT = np.zeros(98)  # m' predicted theta vector
        for PC in range(num_PCs):
            dT += reduced_dim[PC] * self.S[PC] * self.V[:, PC]  # U is the coefficient of the PC

        dx = np.cos(dT)
        dy = np.sin(dT)
        all_x = np.cumsum(dx[::-1])
        all_y = np.cumsum(dy[::-1])
        # Set up the plot window
        plt.cla()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot of Predicted Tail')
        # Add a margin to the limits for better visualization
        plt.xlim(0, 100)
        plt.ylim(-5, 5)
        plt.axis('equal')
        # Plotting
        plt.plot(all_x, all_y, linewidth=2, label=f'Frame {frame_num}')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1)  # Pause briefly to ensure the plot window updates

    @staticmethod
    def plot_PCs(text,V):
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


    def plot_coefficients(self,reduced_dim, frame_number):
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




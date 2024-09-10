import numpy as np
import scipy.linalg


def convolve_mean_by_window_size(
    window_size,
    target_array,
):
    target_array_len = len(
        target_array,
    )

    window = np.zeros(
        shape=target_array_len,
    )
    window[0:window_size] = 1 / window_size

    convolution_matrix = scipy.linalg.toeplitz(
        c=window,
    )
    vector_of_angles = np.array(
        object=target_array,
    )

    convolution = vector_of_angles.dot(
        convolution_matrix,
    )

    return convolution


def get_numerical_derivative(
    array,
    dx=1,
    use_abs=True,
    padded=True,
):
    numerical_derivative = array[0:-1] - array[1:]
    numerical_derivative = numerical_derivative / dx
    if padded:
        numerical_derivative = np.concatenate(
            (
                numerical_derivative,
                numerical_derivative[-1:],
            )
        )

    if use_abs:
        numerical_derivative = np.abs(
            numerical_derivative,
        )

    return numerical_derivative


def filter_epsilon_and_minimum_length_of_array(
    target_array,
    epsilon,
    minimum_sequence_length,
):
    target_array = np.concatenate(
        [
            target_array,
            [0],
        ]
    )
    good_sequences = []

    counter = 0
    for i in range(0, len(target_array)):
        if target_array[i] > epsilon:
            counter += 1
        elif counter >= minimum_sequence_length:
            good_sequences.append(
                [
                    i - counter,
                    i,
                ]
            )
            counter = 0
        else:
            counter = 0

    return good_sequences

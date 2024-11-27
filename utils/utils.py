import csv
import socket
import time
import logging
import os
import cv2
import glob
import ctypes as ct
import math


def loadCSV(csvFileName):
    with open(csvFileName, mode='r') as infile:
        reader = csv.DictReader(infile)
        result = []
        for row in reader:
            result.append(row)
        # print({rows[0]:rows[1] for rows in reader})
        return result


def writeCSV(csvFileName, dict):
    with open(csvFileName, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict.keys())
        writer.writeheader()
        writer.writerow(dict)


def array_to_csv(csv_file_name, arr):
    with open(csv_file_name, 'w', newline='') as file:
        my_writer = csv.writer(file, delimiter=',')
        my_writer.writerows(arr)


def sendF9Marker():
    HOST = '132.64.105.40'  # The IP address of the streams-7 macing, can be obtained using ipconfig 
    PORT = 65432  # The port used by the server

    logging.debug(f'Trying to connect host {HOST} Port {PORT}')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        tsStart = time.time()
        s.sendall(b'f9')  # sending f9 that cause stream-7 to create event-marker
        data = s.recv(1024)
        tsEnd = time.time()
        logging.debug(f'Comm round trip was {(tsEnd - tsStart) * 1000} ms')
        # logging.info ("Marker sent !")


def create_directory(dir_name):
    """

        Args:
            dir_name: the name of the directory to be created

        Returns:
            nothing

        """

    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_name)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_name)
        return True
    else:
        return False


def opencv_create_video(file_prefix, height, width, data_path, image_file_type):
    i = 0
    frame_rate = 30
    out = cv2.VideoWriter(f'{data_path}\\{file_prefix}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                          (width, height))

    print("create video in progress")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for filename in glob.glob(f'{data_path}\\*.{image_file_type}'):
        print(filename)
        img = cv2.imread(filename)
        text = f'frame={i}'
        cv2.putText(img, text, (1, 50), font, 2, (255, 255, 0), 2)
        out.write(img)
        # os.remove(filename)
        i = i + 1
        if i % 1000 == 0:
            print(f'{i} images processed')

    out.release()
    print("video is completed")


def dark_title_bar(window):
    """
    Added by sharon in order to change the title bar to black
    see youtube https://www.youtube.com/watch?v=4Gi1sKKn_Ts
    last two lines were added as the above code
    MORE INFO:
    https://docs.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
    """
    window.update()
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    set_window_attribute = ct.windll.dwmapi.DwmSetWindowAttribute
    get_parent = ct.windll.user32.GetParent
    hwnd = get_parent(window.winfo_id())
    rendering_policy = DWMWA_USE_IMMERSIVE_DARK_MODE
    value = 2
    value = ct.c_int(value)
    set_window_attribute(hwnd, rendering_policy, ct.byref(value), ct.sizeof(value))
    window.withdraw()  # added due to win10 adjustment, see comment inthe above youtueb
    window.deiconify()  # same as above

def polar_to_cartesian(theta, r):
    # Convert angle to radians
    theta_radians = math.radians(theta)
    # Calculate Cartesian coordinates
    x = r * math.cos(theta_radians)
    y = r * math.sin(theta_radians)
    return x,y


def signed_angle_between_vectors(u, v):
    # Compute the 2D cross product (scalar)
    cross_product = u[0] * v[1] - u[1] * v[0]

    # Compute the dot product
    dot_product = u[0] * v[0] + u[1] * v[1]

    # Use atan2 to find the signed angle
    angle_radians = math.atan2(cross_product, dot_product)

    # Optionally convert to degrees
    angle_degrees = math.degrees(angle_radians)
    if angle_degrees < 0:
        angle_degrees = angle_degrees + 360
    return angle_degrees



def angle_between_vectors_ccw(v1, v2):
    """
    Calculate the counterclockwise angle between two 2D vectors, returning a value in [0, 360).

    Parameters:
        v1 (tuple): First vector (x1, y1)
        v2 (tuple): Second vector (x2, y2)

    Returns:
        float: Counterclockwise angle in degrees
    """
    # Compute the cross product and dot product
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Compute the angle in radians using atan2
    angle_radians = math.atan2(cross_product, dot_product)

    # Convert the angle to degrees and ensure it's in the range [0, 360)
    angle_degrees = math.degrees(angle_radians)
    if angle_degrees < 0:
        angle_degrees += 360  # Ensure the angle is positive

    return angle_degrees
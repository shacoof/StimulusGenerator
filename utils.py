import csv
import socket
import time
import logging
import os
import cv2
import glob


def loadCSV(csvFileName):
    with open(csvFileName, mode='r') as infile:
        reader = csv.DictReader(infile)
        result = []
        for row in reader: 
            result.append(row)
        #print({rows[0]:rows[1] for rows in reader})
        return result


def writeCSV(csvFileName,dict):
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
    PORT = 65432        # The port used by the server

    logging.debug(f'Trying to connect host {HOST} Port {PORT}')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        tsStart = time.time()
        s.sendall(b'f9') #sending f9 that cause stream-7 to create event-marker 
        data = s.recv(1024)
        tsEnd=time.time()
        logging.debug(f'Comm round trip was {(tsEnd-tsStart)*1000} ms')
        #logging.info ("Marker sent !")


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

def opencv_create_video(file_prefix, height, width, data_path):
    i = 0
    frame_rate = 30
    out = cv2.VideoWriter(f'{data_path}\\{file_prefix}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                          (width, height))

    print("create video in progress")
    for filename in glob.glob(f'{data_path}\\*.jpeg'):
        img = cv2.imread(filename)
        out.write(img)
        # os.remove(filename)
        i = i + 1
        if i % 1000 == 0:
            print(f'{i} images processed')

    out.release()

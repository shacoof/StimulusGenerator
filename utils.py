import csv
import socket
import time
import logging

def loadCSV(csvFileName):
    with open(csvFileName, mode='r') as infile:
        reader = csv.DictReader(infile)
        result=[]
        for row in reader: 
            result.append(row)
        #print({rows[0]:rows[1] for rows in reader})
        return result

def writeCSV(csvFileName,dict):
    with open(csvFileName, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict.keys())
        writer.writeheader()
        writer.writerow(dict)

def sendF9Marker():
    HOST = '132.64.105.40'  # The IP address of the streams-7 macing, can be obtained using ipconfig 
    PORT = 65432        # The port used by the server

    print ("Trying to connect host "+HOST+" Port "+ str(PORT))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        tsStart = time.time()
        s.sendall(b'f9') #sending f9 that cause stream-7 to create event-marker 
        tsEnd=time.time()
        logging.debug("time diff =" + str((tsEnd-tsStart)/10))
        logging.info ("Marker sent !")
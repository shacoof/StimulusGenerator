import multiprocessing
import PySpin
from PIL import Image
import matplotlib.pyplot as plt

global continue_recording
continue_recording = True


def handle_close(evt):
    """
    This function will close the GUI when close event happens.

    :param evt: Event that occurs when the figure closes.
    :type evt: Event
    """
    global continue_recording
    continue_recording = False


def image_writer_worker(qin, data_path, image_file_type):
    name = multiprocessing.current_process().name
    print(f"process {name} starting")
    counter = 0
    # Figure(1) is default so you can omit this line.
    # Figure(0) will create a new window every time program hits this line
    fig = plt.figure(1)
    # Close the GUI when close event happens
    fig.canvas.mpl_connect('close_event', handle_close)

    while True:
        i, image_result = qin.get()  # Read from the queue and do nothing
        if i == -1:
            break
        else:
            l = len(str(i))
            s = '0' * (12 - l)
            im = Image.fromarray(image_result)
            im.save(f"{data_path}\\img{s + str(i)}.{image_file_type}")
            counter = counter + 1
            if counter % 1000 == 0:
                print(f"process {name}  processed {counter} files so far.... ")
            """ if counter % 10 == 0:
                plt.imshow(image_result, cmap='gray')
                plt.axis('off')
                # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                # Interval is in seconds.
                plt.pause(0.001)
                # Clear current reference of a figure. This will improve display speed significantly
                plt.clf()"""

    print(f"process {name} is done, processed {counter} files ")
    return

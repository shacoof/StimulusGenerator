import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import psutil


def plot_worker(shared_data, lock):
    """
    Worker function that listens to the plot_queue and handles the plotting
    of tail traces to avoid delays in the main process.
    """
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('off')

    while True:
        with lock:
            image = shared_data["image"]
            bout_index = shared_data["frame_number"]

        if bout_index == -1:
            break

        ax.clear()  # Clear the previous frame
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)



        ax.set_title(f"Frame {bout_index}")

        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.close()


class ImageRenderer:
    def __init__(self):
        """
        Image renderer class, creates a process to render images im real time
        """

        self.current_frame = 0
        self.shared_data = multiprocessing.Manager().dict()
        self.lock = multiprocessing.Lock()
        self.plot_process = multiprocessing.Process(target=plot_worker, args=(self.shared_data, self.lock))
        self.shared_data["image"] = np.zeros((100, 100))
        self.shared_data["frame_number"] = 0

        #TODO add flag
        self.plot_process.start()
        process5_psutil = psutil.Process(self.plot_process.pid)
        process5_psutil.cpu_affinity([5])

    def update_shared_data(self, image):
        """
        Prepare the data for plotting and send it to the plot queue.
        """
        with self.lock:
            self.shared_data['image'] = image
            self.shared_data['frame_number'] = self.current_frame

    def stop_plotting(self):
        """Terminate the plot worker."""
        with self.lock:
            self.shared_data["frame_number"] = -1
        self.plot_process.join()

    def process_frame(self, frame):
        """
        Processes a single frame and updates the rendering process
        Args:
            frame: a np.array 2D matrice of the camera acquired image
        Returns:
        """
        if frame is None:
            # TODO add flag here
            self.stop_plotting()
            return
        self.update_shared_data(frame)

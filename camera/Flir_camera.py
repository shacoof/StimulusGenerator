import PySpin
import numpy as np
import time


class SpinnakerCamera:
    def __init__(self):
        # Initialize the system and retrieve cameras
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() == 0:
            raise RuntimeError("No cameras found.")
        self.camera = self.cam_list[0]  # Use the first camera found
        self.camera.Init()  # Initialize the camera

    def set_image_dimensions(self, width=None, height=None, offsetX=None, offsetY=None):
        """
        Set the image dimensions (width and height) in pixels.
        """
        # Check for the maximum allowable dimensions
        max_width = self.camera.Width.GetMax()
        max_height = self.camera.Height.GetMax()

        if width:
            if width > max_width:
                raise ValueError(f"Requested width {width} exceeds maximum width {max_width}.")
            self.camera.Width.SetValue(width)

        if height:
            if height > max_height:
                raise ValueError(f"Requested height {height} exceeds maximum height {max_height}.")
            self.camera.Height.SetValue(height)

        # Optionally set offsets (if needed)
        if offsetX:
            self.camera.OffsetX.SetValue(offsetX)
        if offsetY:
            self.camera.OffsetY.SetValue(offsetY)

    def set_camera_settings(self, frame_rate=None, exposure_time=None, gain=None):
        """
        Set camera settings such as frame rate, exposure time, and gain.
        """
        if frame_rate:

            nodemap = self.camera.GetNodeMap()  # Access the camera's nodemap

            if frame_rate:
                # Disable frame rate auto if it exists
                node_frame_rate_auto = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionFrameRateAuto'))
                if PySpin.IsAvailable(node_frame_rate_auto) and PySpin.IsWritable(node_frame_rate_auto):
                    node_frame_rate_auto_off = node_frame_rate_auto.GetEntryByName('Off')
                    if PySpin.IsAvailable(node_frame_rate_auto_off) and PySpin.IsReadable(node_frame_rate_auto_off):
                        node_frame_rate_auto.SetIntValue(node_frame_rate_auto_off.GetValue())
                        print("Frame rate auto set to: Off")

                # Set the frame rate
                node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
                if PySpin.IsAvailable(node_acquisition_framerate) and PySpin.IsWritable(node_acquisition_framerate):
                    node_acquisition_framerate.SetValue(frame_rate)
                    print(f"Frame rate set to: {frame_rate} fps")
                else:
                    print("Frame rate is not writable or not available.")

        if exposure_time:
            self.camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            self.camera.ExposureTime.SetValue(exposure_time)
        if gain:
            self.camera.GainAuto.SetValue(PySpin.GainAuto_Off)
            self.camera.Gain.SetValue(gain)

    def get_frame(self):
        """
        Capture a single frame and return it as a NumPy array.
        """
        self.camera.BeginAcquisition()
        image = self.camera.GetNextImage()

        # Ensure image completion
        if image.IsIncomplete():
            raise RuntimeError("Image incomplete with status: {}".format(image.GetImageStatus()))

        # Convert image to numpy array
        np_image = np.array(image.GetNDArray())

        # Release image and stop acquisition
        image.Release()
        self.camera.EndAcquisition()

        return np_image

    def get_frames(self, num_frames, frame_rate):
        """
        Capture a sequence of frames at a specified frame rate.
        :param num_frames: The number of frames to capture.
        :param frame_rate: Frame rate in frames per second.
        :return: List of frames as NumPy arrays.
        """
        self.set_camera_settings(frame_rate=frame_rate)
        frames = []

        self.camera.BeginAcquisition()
        start_time = time.time()

        for _ in range(num_frames):
            image = self.camera.GetNextImage()

            if image.IsIncomplete():
                print("Image incomplete with status: {}".format(image.GetImageStatus()))
                continue

            np_image = np.array(image.GetNDArray())
            frames.append(np_image)
            image.Release()

            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / frame_rate) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.camera.EndAcquisition()
        return frames

    def close(self):
        """
        Close the camera and release system resources.
        """
        self.camera.DeInit()
        del self.camera
        self.cam_list.Clear()
        self.system.ReleaseInstance()

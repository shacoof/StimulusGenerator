# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2021 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
#  image_reader_worker.py shows how to create an AVI video from a vector of
#  images. It relies on information provided in the Enumeration, Acquisition,
#  and NodeMapInfo examples.
#
#  This example introduces the SpinVideo class, which is used to quickly and
#  easily create various types of AVI videos. It demonstrates the creation of
#  three types: uncompressed, MJPG, and H264.

import PySpin
import sys
import multiprocessing
import time
import datetime
import utils
from NiDaqPulse import NiDaqPulse

# this array of arrays is used to create a log of which frame was captured at the time of stimulus event
# see NUM_IMAGES
# each item is [timestamp, frame-number, stimulus-message ]
image_array = [['timestamp', 'image no', 'stimulus']]
global queue_reader, queue_writer, writer_process
global file_prefix, data_path, camera_output_device


class AviType:
    """'Enum' to select AVI video type to be created and saved"""
    UNCOMPRESSED = 0
    MJPG = 1
    H264 = 2


chosenAviType = AviType.UNCOMPRESSED  # change me!
NUM_IMAGES = 180  # 00000  # number of images to use in AVI file

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


def save_list_to_avi(nodemap, nodemap_tldevice, images):
    """
    This function prepares, saves, and cleans up an AVI video from a vector of images.

    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :param images: List of images to save to an AVI video.
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :type images: list of ImagePtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** CREATING VIDEO ***')
    global file_prefix, data_path

    try:
        result = True

        # Retrieve device serial number for filename
        device_serial_number = ''
        node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

        if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(node_serial):
            device_serial_number = node_serial.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Get the current frame rate; acquisition frame rate recorded in hertz
        #
        # *** NOTES ***
        # The video frame rate can be set to anything; however, in order to
        # have videos play in real-time, the acquisition frame rate can be
        # retrieved from the camera.

        node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))

        if not PySpin.IsAvailable(node_acquisition_framerate) and not PySpin.IsReadable(node_acquisition_framerate):
            print('Unable to retrieve frame rate. Aborting...')
            return False

        framerate_to_set = node_acquisition_framerate.GetValue()

        print('Frame rate to be set to %d...' % framerate_to_set)

        # Select option and open AVI filetype with unique filename
        #
        # *** NOTES ***
        # Depending on the filetype, a number of settings need to be set in
        # an object called an option. An uncompressed option only needs to
        # have the video frame rate set whereas videos with MJPG or H264
        # compressions should have more values set.
        #
        # Once the desired option object is configured, open the AVI file
        # with the option in order to create the image file.
        #
        # Note that the filename does not need to be appended to the
        # name of the file. This is because the AVI recorder object takes care
        # of the file extension automatically.
        #
        # *** LATER ***
        # Once all images have been added, it is important to close the file -
        # this is similar to many other standard file streams.

        avi_recorder = PySpin.SpinVideo()

        if chosenAviType == AviType.UNCOMPRESSED:
            # avi_filename = 'SaveToAvi-Uncompressed-%s' % device_serial_number
            avi_filename = f"{data_path}\\{file_prefix}-Uncompressed"

            option = PySpin.AVIOption()
            option.frameRate = framerate_to_set
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()

        elif chosenAviType == AviType.MJPG:
            # avi_filename = 'SaveToAvi-MJPG-%s' % device_serial_number
            avi_filename = f"{data_path}\\{file_prefix}-MJPG"
            option = PySpin.MJPGOption()
            option.frameRate = framerate_to_set
            option.quality = 75
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()

        elif chosenAviType == AviType.H264:
            # avi_filename = 'SaveToAvi-H264-%s' % device_serial_number
            avi_filename = f"{data_path}\\{file_prefix}-H264"
            option = PySpin.H264Option()
            option.frameRate = framerate_to_set
            option.bitrate = 1000000
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()

        else:
            print('Error: Unknown AviType. Aborting...')
            return False

        avi_recorder.Open(avi_filename, option)

        # Construct and save AVI video
        #
        # *** NOTES ***
        # Although the video file has been opened, images must be individually
        # appended in order to construct the video.
        print('Appending %d images to AVI file: %s.avi...' % (len(images), avi_filename))
        for i in range(len(images)):
            avi_recorder.Append(images[i])
            if i % 100 == 0:
                print(f"Images appended {i}")
        print(f"{len(images)} images appended")

        # Close AVI file
        #
        # *** NOTES ***
        # Once all images have been appended, it is important to close the
        # AVI file. Notice that once an AVI file has been closed, no more
        # images can be added.

        avi_recorder.Close()
        print('Video saved at %s.avi' % avi_filename)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('\n*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def acquire_images(cam, nodemap):
    global queue_reader, queue_writer
    global file_prefix
    """
    This function acquires 30 images from a device, stores them in a list, and returns the list.
    please see the Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images...')

        # Retrieve, convert, and save images
        images = list()

        i = 0
        msg = 'stay'
        print(f"******* Taking images...")

        # setting parameters to the JPEG coder
        op = PySpin.JPEGOption()
        op.quality = 50

        # todo Wait for signal to start recording

        # taking one image just to get the w/h
        image_result = cam.GetNextImage(1000)
        width = image_result.GetWidth()
        height = image_result.GetHeight()
        t1 = time.time()
        while msg != 'exit':
            try:
                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d...' % image_result.GetImageStatus())
                else:
                    if queue_reader.qsize() > 0:
                        msg = queue_reader.get()
                        if msg == 'exit':
                            # one for each writer, we have 2, the third is for main to use to create the movie
                            # the tupple (w,h,file_prefix) is used by the main
                            # See SGMainApp in section  "if self.sg.run_stimuli() == constants.DONE"
                            queue_writer.put((-1, (width, height, file_prefix)))
                            queue_writer.put((-1, (width, height, file_prefix)))
                            queue_writer.put((-1, (width, height, file_prefix)))
                            print(f"process camera_control_worker is done ")
                        else:
                            image_array.append([datetime.datetime.now().strftime("%H:%M:%S:%f"), i, msg])

                    # note that I already acquire the image, so even if exit sent I still have 1 image in the buffer
                    queue_writer.put((i, image_result.GetNDArray()))
                    image_result.Release()
                    i += 1

                    # TODO (LILACH) here send TTL based on i value
                    if (i % 166 == 0) and (camera_output_device is not None):
                       camera_output_device.give_pulse()
                       print(f"****** Give pulse frame number = {i}")


                    #  Retrieve next received image
                    image_result = cam.GetNextImage(1000)
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                result = False

        delta = time.time() - t1
        msg = f"{i} images taken in {delta} sec , frames per sec {i / delta}"
        print(msg)
        image_array.append([datetime.datetime.now().strftime("%H:%M:%S:%f"), 0, msg])
        utils.array_to_csv(f'{data_path}\\{file_prefix}_log.csv', image_array)
        # End acquisition
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result, images


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run example on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire list of images
        err, images = acquire_images(cam, nodemap)
        if err < 0:
            return err

        # now that we save images to the disk we don't need that
        # result &= save_list_to_avi(nodemap, nodemap_tldevice, images)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected:', num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):
        result &= run_single_camera(cam)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release instance
    system.ReleaseInstance()

    return result


def camera_control_worker(queue_reader_in, queue_writer_in, path_in, file_prefix_in):
    global queue_reader, queue_writer, file_prefix, data_path, writer_process, camera_output_device
    camera_output_device = None
    data_path = path_in
    file_prefix = file_prefix_in
    name = multiprocessing.current_process().name
    print(f"******** queue {name} running, camera output device = {camera_output_device}")
    queue_reader = queue_reader_in
    queue_writer = queue_writer_in
    main()
    if camera_output_device is not None:
        camera_output_device.stop()
    return


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)

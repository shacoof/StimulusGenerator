import os
import numpy as np
from PIL import Image
import time
from config_files.closed_loop_config import camera_frame_rate


def camera_emulator_function(queue_reader, queue_writer, images_queue):
    '''
    Allows running the closed loop without having a camera and instead injecting in a loop frames from a prerecorded
    scenario. created for dev purposes. To use, make sure to set camera_emulator_on = True in the closed_loop_config.py,
    and set emulator_with_camera = False
    Args:
        queue_reader: Not used
        queue_writer: puts images here to be saved to output dir
        images_queue: puts the sames images here to be processed by the closed loop algo
    Returns:
    '''
    frame_time = 1 / camera_frame_rate
    start_frame = 197751
    end_frame = 198450
    all_frame_mats = []
    images_path = "\\\ems.elsc.huji.ac.il\\avitan-lab\Lab-Shared\Data\ClosedLoop\\20231204-f2\\raw_data"
    for i in range(start_frame, end_frame + 1):
        # Format the image filename based on the numbering pattern
        img_filename = f"img{str(i).zfill(12)}.jpg"
        img_path = os.path.join(images_path, img_filename)
        try:
            with Image.open(img_path) as img:
                image_matrix = np.array(img)
                all_frame_mats.append(image_matrix)
        except Exception as e:
            print(f"Error loading image: {e}")
    i = 0
    number_of_frames = len(all_frame_mats)
    init_time = time.perf_counter()

    while True:
        index = i % number_of_frames
        image = all_frame_mats[index]
        queue_writer.put_nowait((i, image))
        images_queue.put_nowait((i, image))
        next_time = init_time + (i + 1) * frame_time
        while time.perf_counter() < next_time:
            pass
        i += 1




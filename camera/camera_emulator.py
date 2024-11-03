import os
import numpy as np
from PIL import Image
import time

from closed_loop_process.print_time import print_statistics, start_time_logger, reset_time, print_time

frame_time = 0.010

def camera_emulator_function(queue_reader, queue_writer, images_queue):
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
    print('EMULATOR: STARTING')
    start_time_logger('EMULATOR')
    total_time = None

    while True:
        reset_time()
        start_time = time.perf_counter()
        print('EMULATOR: Frame time: ', total_time, flush=True)
        index = i % number_of_frames
        image = all_frame_mats[index]
        queue_writer.put_nowait((i, image))
        images_queue.put_nowait((i, image))

        next_time = init_time + (i + 1) * frame_time
        delay = next_time - time.perf_counter()
        print('EMULATOR: Delay: ', delay, flush=True)

        while time.perf_counter() < next_time:
            pass
        i += 1
        total_time = time.perf_counter() - start_time
        print_time('end of emulation cycle')


    print('EMULATOR: end')
    print_statistics()

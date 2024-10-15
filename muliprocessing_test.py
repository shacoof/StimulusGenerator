import multiprocessing
import time
import os
from PIL import Image
import numpy as np

def image_proc(image):
    return np.sum(image)

if __name__ == "__main__":


    # directory settings
    start_frame = 197751
    end_frame = 198450
    images_path = f"Z:\Lab-Shared\Data\ClosedLoop\\20231204-f2\\raw_data"

    # load frames
    all_frame_mats = []

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


    pool = multiprocessing.Pool(processes=4)
    start = time.time()
    results = pool.map(image_proc, all_frame_mats)
    end = time.time()
    for i, result in enumerate(results):
        print(f"({i}) = {result}")

    print(f"time {end - start}")
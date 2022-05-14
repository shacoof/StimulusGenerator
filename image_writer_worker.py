import multiprocessing
import PySpin
from PIL import Image
from utils import opencv_create_video

def image_writer_worker(qin, data_path):
    name = multiprocessing.current_process().name
    print(f"process {name} starting")
    op = PySpin.JPEGOption()
    op.quality = 50

    while True:
        i, image_result = qin.get()  # Read from the queue and do nothing
        if i == -1:
            print(f"process {name} is done : start building movie ")
            # todo we are not waiting for the writer to complete, e.g. we need to do join
            width = image_result[0]
            height = image_result[1]
            file_prefix = image_result[2]
            opencv_create_video(file_prefix, height, width, data_path)
            break
        else:
            im = Image.fromarray(image_result)
            im.save(f"{data_path}\\img{i}.jpeg")

    print(f"process {name} is done ")
    return

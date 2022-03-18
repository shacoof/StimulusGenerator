import multiprocessing
import camera_control
import time

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    camera = multiprocessing.Process(name='camera_control_worker',
                                     target=camera_control.camera_control_worker,
                                     args=(queue,))

    camera.start()
    for i in range(1, 10):
        queue.put(i)
        time.sleep(1)
    queue.put('exit')
    camera.join()
    print("the end")




"""from math import sin, radians

for alpha in range(0,91):
    print(f"{alpha} = {(sin(radians(90-alpha))*sin(radians(45 -alpha/2)))/sin(radians(90 + alpha)/2)}")
"""
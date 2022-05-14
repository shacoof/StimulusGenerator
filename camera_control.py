"""import multiprocessing
import time
import datetime

def camera_control_worker(queue):
    name = multiprocessing.current_process().name
    print(f"queue {name} running")
    stay = True
    while stay:
        print(f"current time:-{datetime.datetime.now()}")
        if queue.qsize() == 0:
            time.sleep(0.1)
        else:
            q = queue.get()
            print(f"rec = {q}")
            if q == 'exit':
                stay = False

    print(name, 'Exiting')"""

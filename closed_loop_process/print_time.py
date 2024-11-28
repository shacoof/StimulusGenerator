import time
from config_files.closed_loop_config import debug_time
import numpy as np

GLOBAL_START = 0
PROCESS_NAME = None
TIMES_REC = dict()

def print_time(msg):
    t = (time.perf_counter() - GLOBAL_START) * 1000
    if msg not in TIMES_REC:
        TIMES_REC[msg] = []
    TIMES_REC[msg].append(t)
    if debug_time:
        print(PROCESS_NAME, msg, '{:.1f}'.format(t), flush=True)


def start_time_logger(name):
    global PROCESS_NAME
    PROCESS_NAME = name


def reset_time():
    global GLOBAL_START
    GLOBAL_START = time.perf_counter()


def print_statistics():
    print(PROCESS_NAME, 'Statistics:')
    for k, d in TIMES_REC.items():
        dd = d[1:]
        print(k, np.min(dd), np.max(dd), np.mean(dd), np.std(dd))

import multiprocessing
import time


def worker(x):
    return x ** 3


def main():
    with multiprocessing.Pool(processes=8) as pool:
        start_time = time.time()
        results = list(pool.imap_unordered(worker, range(100000000)))
        end_time = time.time()

    print(f"time {end_time - start_time}")


if __name__ == '__main__':
    main()
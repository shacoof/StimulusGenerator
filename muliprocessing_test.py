
import numpy as np
import os
import time
import multiprocessing
from PIL import Image


# Example function to process a single matrix (replace with your actual function)
def process_matrix(matrix):
    # For demonstration, we'll just return the sum of all elements in the matrix
    return np.sum(matrix)


# Function to handle multiprocessing
def process_matrices_in_parallel(matrix_array):
    # Create a multiprocessing pool with the number of available cores
    with multiprocessing.Pool() as pool:
        # Map the process_matrix function to each matrix in the array
        start_parallel = time.time()
        results = pool.map(process_matrix, matrix_array)
        end_parallel = time.time()
        print(f"parallel {end_parallel - start_parallel}")
    return results

if __name__ == "__main__":
    # every 1/500 sec call function with frame
    queue_closed_loop_prediction = multiprocessing.Queue()
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
            # Process the matrices in parallel
    start_parallel = time.time()
    processed_results = process_matrices_in_parallel(all_frame_mats)
    end_parallel = time.time()

    result = []
    start_regular = time.time()
    for mat in all_frame_mats:
        result.append(np.sum(mat))
    end_regular = time.time()
    # Output the results
    print(f"parallel {end_parallel - start_parallel}")
    print(f"regular {end_regular - start_regular}")










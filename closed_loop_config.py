debug_bout_detector = False
debug_tail = False
debug_PCA = False
use_camera = True
use_stytra_tracking = True

number_of_frames_calibration = 1000
fr_500 = False # program only supports frame rate 500 or 166
if fr_500:
    camera_frame_rate = 500
    camera_frame_width_in_pixels = 352
    camera_frame_height_in_pixels = 312
    camera_frame_offsetX_in_pixels = 852
    camera_frame_offsetY_in_pixels = 882
    frames_from_bout = 35
else: # fr 166
    camera_frame_rate = 166
    camera_frame_width_in_pixels = 1408
    camera_frame_height_in_pixels = 884
    camera_frame_offsetX_in_pixels = 340
    camera_frame_offsetY_in_pixels = 500
    frames_from_bout = 12


debug_bout_detector = False
debug_PCA = False
debug_mode = True
camera_emulator_on = False
start_angle = 30 # The entire screen spans 15 - 165 degrees
end_angle = 150
stimuli_moving_speed = 50 # degree per sec
spacer_duration = 6000 # ms
stimuli_floating_speed = 2 # degree per sec
number_of_frames_calibration = 10
use_stytra_tracking = True
plot_bout_detector = False
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


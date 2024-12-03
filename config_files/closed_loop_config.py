debug_mode = False
debug_time = False
emulator_with_camera = True
camera_emulator_on = False
number_of_tail_segments = 23
start_angle = 30 # The entire screen spans 15 - 165 degrees
end_angle = 150
stimuli_moving_time = 5000 # time until update in ms
calibration_stimuli_speed = 300
number_of_frames_calibration = 100
cut_velocity_by_factor = 1 # between zero and one - how to reduce velocity after movement
use_stytra_tracking = True
camera_frame_rate = 250 # either 166, 250 or 500
spacer_time_between_batches = 5000
fr_for_realtime_is_500 = False # program only supports frame rate 500 or 166, 500 is not recommended


if fr_for_realtime_is_500:
    frames_from_bout = 35
else:
    frames_from_bout = 12

if camera_frame_rate == 500:
    camera_frame_width_in_pixels = 352
    camera_frame_height_in_pixels = 312
    camera_frame_offsetX_in_pixels = 852
    camera_frame_offsetY_in_pixels = 882
    take_every_x_frame = 3

elif camera_frame_rate == 166: # fr 166
    camera_frame_width_in_pixels = 1408
    camera_frame_height_in_pixels = 884
    camera_frame_offsetX_in_pixels = 340
    camera_frame_offsetY_in_pixels = 500
    take_every_x_frame = 1

elif camera_frame_rate == 250:
    camera_frame_width_in_pixels = 416
    camera_frame_height_in_pixels = 400
    camera_frame_offsetX_in_pixels = 820
    camera_frame_offsetY_in_pixels = 780
    take_every_x_frame = 2



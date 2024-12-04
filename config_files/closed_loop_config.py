debug_mode = False # set True to see the tail tracking in realtime
debug_time = False # set True to output to console different timing specs
emulator_with_camera = False # set True for dev mode when you want to test without an actual fish and using pre-recorded
# data (in this case set camera_emulator_on False)
camera_emulator_on = True # set True for dev mode when you want to test without an actual fish and without camera and using pre-recorded
# data (in this case set emulator_with_camera False)
number_of_tail_segments = 23 # resolution of tail tracking the higher it is the longer tail tracking takes
start_angle = 30 # start angle of virtual screen for stimuli - The entire screen spans 15 - 165 degrees
end_angle = 150 # end angle of virtual screen for stimuli - The entire screen spans 15 - 165 degrees
stimuli_moving_time = 500 # time for all stimuli to move to their new predicted location for when a bout occurs
calibration_stimuli_speed = 10 # degree per second
number_of_frames_calibration = 100 # should be around 10000 - make sure the fish moves at least once during this time
cut_velocity_by_factor = 1 # between zero and one - how to reduce velocity of dynamic stimuli after movement
use_stytra_tracking = True # if false we use lab tracking which is slower and not recommended
camera_frame_rate = 250 # either 166, 250 or 500
spacer_time_between_batches = 5000 # time between batches in ms
fr_for_realtime_is_500 = False # The frame rate for images used for bout prediction (other frame rates can be used for image acquiring)
# program only supports frame rate 500 or 166, 500 is not recommended


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



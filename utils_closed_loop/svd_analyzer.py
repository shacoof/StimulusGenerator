import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.linear_model import HuberRegressor

class fish_svd_data():
    def __init__(self, fish):
        self.merge_events(fish)
        self.add_theta_matrix()
        self.clean_nan_from_data()
        self.get_svd()
        self.rotate_axes()

    def events_with_svd(self):
        assert self.new_axes is not None
        fish_events = []
        u_with_nan = self.new_axes.copy()
        u_with_nan[:,np.isnan(self.theta_mean)] = np.nan
        for event_idx in range(self.event_num[-1] + 1):
            event_u = u_with_nan[:,self.event_num==event_idx].copy()
            fish_events.append(event_u)
        return fish_events

    def merge_events(self, fish):
        # merge events into a single file
        event_num = np.zeros(0, np.int_)
        event_frame = np.zeros(0, np.int_)
        event_bouts_start = np.zeros(0, np.int_)
        event_bouts_end = np.zeros(0, np.int_)
        cumm_frames = 0
        all_events_interpolated_tail_path = np.zeros((0, *fish.events[0].tail.interpolated_tail_path.shape[1:]))
        directions_in_deg = np.zeros(0)
        tip_to_swimbladder_distance = np.zeros(0)
        head_origin = np.zeros((0, 2))
        for event_idx in range(len(fish.events)):
            event_num = np.concatenate((event_num, event_idx*np.ones(fish.events[event_idx].tail.interpolated_tail_path.shape[0], np.int_)))
            event_frame = np.concatenate((event_frame, np.arange(fish.events[event_idx].tail.interpolated_tail_path.shape[0])))
            event_bouts_start = np.concatenate((event_bouts_start, np.int64(fish.events[event_idx].tail.bout_start_frames)+cumm_frames))
            event_bouts_end = np.concatenate((event_bouts_end, np.int64(fish.events[event_idx].tail.bout_end_frames)+cumm_frames))

        #     fish_data.events{1}.head.origin_points.x = [fish_data.events{1}.head.origin_points.x fish_data.events{event_idx}.head.origin_points.x];
        #     fish_data.events{1}.head.origin_points.y = [fish_data.events{1}.head.origin_points.y fish_data.events{event_idx}.head.origin_points.y];
        #     fish_data.events{1}.tail.tail_path_list = [fish_data.events{1}.tail.tail_path_list fish_data.events{event_idx}.tail.tail_path_list];
        #     fish_data.events{1}.tail.swimbladder_points_list.x = [fish_data.events{1}.tail.swimbladder_points_list.x fish_data.events{event_idx}.tail.swimbladder_points_list.x];
        #     fish_data.events{1}.tail.swimbladder_points_list.y = [fish_data.events{1}.tail.swimbladder_points_list.y fish_data.events{event_idx}.tail.swimbladder_points_list.y];
            head_origin = np.concatenate((head_origin, np.array([fish.events[event_idx].head.origin_points.x, fish.events[event_idx].head.origin_points.y]).T))
            directions_in_deg = np.concatenate((directions_in_deg, fish.events[event_idx].head.directions_in_deg))
            tip_to_swimbladder_distance = np.concatenate((tip_to_swimbladder_distance, fish.events[event_idx].tail.tip_to_swimbladder_distance))
            all_events_interpolated_tail_path = np.concatenate((all_events_interpolated_tail_path, fish.events[event_idx].tail.interpolated_tail_path))
            cumm_frames = cumm_frames + fish.events[event_idx].tail.interpolated_tail_path.shape[0]

        self.event_num = event_num
        self.event_frame = event_frame
        self.event_bouts_start = event_bouts_start - 1  # due to python vs matlab indexing issues
        self.event_bouts_end = event_bouts_end  # should be -1 due to python vs matlab indexing issues, but end frame is included in matlab and not in python.
        self.all_events_interpolated_tail_path = all_events_interpolated_tail_path
        self.directions_in_deg = directions_in_deg
        self.tip_to_swimbladder_distance = tip_to_swimbladder_distance
        self.head_origin = head_origin

    def add_theta_matrix(self):
        extra_size = 6 # when fiting the tail, I don't trust the points in the end of the fit. 
                    # The reason is that the fitting errors will be larger and after these points the fit can be anything.

        tail_diff = np.diff(self.all_events_interpolated_tail_path,axis=1)
        tail_line = tail_diff[:,:,0] + 1j*tail_diff[:,:,1] 
        d_theta = np.angle(tail_line)
        d_theta = np.unwrap(d_theta)[:,:-extra_size+1]
        self.theta_mean = np.mean(d_theta,axis=1)
        theta_matrix = d_theta - self.theta_mean[:,np.newaxis]
        self.theta_matrix = theta_matrix[:,1:]

    def clean_nan_from_data(self, remove_nan = False):

        if remove_nan:
        # remove nans from data:
            nan_before_frame = np.cumsum(np.isnan(self.theta_mean))

            self.event_num = self.event_num[~np.isnan(self.theta_mean)]
            self.event_frame = self.event_frame[~np.isnan(self.theta_mean)]
            self.event_bouts_start -= nan_before_frame[self.event_bouts_start]
            self.event_bouts_end -= nan_before_frame[self.event_bouts_end]
            self.all_events_interpolated_tail_path = self.all_events_interpolated_tail_path[~np.isnan(self.theta_mean)]
            self.directions_in_deg = self.directions_in_deg[~np.isnan(self.theta_mean)]
            self.tip_to_swimbladder_distance = self.tip_to_swimbladder_distance[~np.isnan(self.theta_mean)]
            self.theta_matrix = self.theta_matrix[~np.isnan(self.theta_mean),:]
            self.theta_mean = self.theta_mean[~np.isnan(self.theta_mean)]
        else:
        # zero nans
            self.all_events_interpolated_tail_path[np.isnan(self.theta_mean)] = 0
            self.theta_matrix[np.isnan(self.theta_mean),:] = 0

    def get_svd(self, comp_num = 10):
        [u, s, vh] = svd(self.theta_matrix,full_matrices=False)
        u[:,:comp_num] = u[:,:comp_num]*np.sign(vh[:comp_num,-1])[np.newaxis,:]
        vh[:comp_num,:] = vh[:comp_num,:]*np.sign(vh[:comp_num,-1])[:,np.newaxis]
        total_variance = np.cumsum(s/np.sum(s))
        self.u = u[:,:comp_num]
        self.vh = vh[:comp_num,:]
        self.total_variance = total_variance[:comp_num]

    def rotate_axes(self): 
        rotation_matrix = np.array([[-0.5851, 0.7883, 0.1911],[-0.1779, -0.3541, 0.9242],[0.7641, 0.4786, 0.4455]])
        second_rotation = np.array([[-0.7311, -0.6823, 0],[-0.6823,0.7311, 0],[0, 0, 1]])

        rotation_angle = 10.*np.pi/180.
        third_rotation = np.array([[np.cos(rotation_angle ), 0., np.sin(rotation_angle )], [0., 1., 0.], [-np.sin(rotation_angle ),0.,np.cos(rotation_angle )]])

        self.new_axes = third_rotation@second_rotation@rotation_matrix@self.u[:,:3].T

    def get_fish_dict(self):
        fish_dict = {'c_all':self.new_axes,
                    'bout_start':self.event_bouts_start + 1,  # due to matlab numbering
                    'bout_end':self.event_bouts_end,
                    'event_frame':self.event_frame + 1,  # due to matlab numbering
                    'event_num':self.event_num + 1,  # due to matlab numbering
                    'eigenfish':self.vh[:3,:].T,
                    'tip_to_swimbladder_distance':self.tip_to_swimbladder_distance}
        return fish_dict
#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modified work: Konstantin Ivanchenko
# Date: December 25, 2019

from perception import lane_finder as lf
import numpy as np


class MarkedGrid(object):
    """
    Representation of a static and dynamic obstacles, as well as additional
    limitations such as lane boundaries.
    """
    def __init__(self):
        # initialize grid

        # scanner types
        self.lf = lf.LaneFinder()
        self.left_lines = None  # denotes left lines identified as momentary driving boundaries
        self.right_lines = None  # denotes right lines identified as momentary driving boundaries
        self.left_lines_gl = None
        self.right_lines_gl = None

        # TODO: implement static occupancy grid

    def find_lines(self, frame_depth, frame_sems):
        """
        Estimates lane boundaries as a set of lines projected on a driving
        surface.
        :param:
            frame_depth: carla.Image object - data from depth camera sensor.
            frame_sems: carla.Image object - data from depth camera sensor.
        :return: None
        """
        # link to the camera perception for lanes

        # update current camera frame data
        self.lf.update_frames(frame_depth, frame_sems)
        # camera data lane extraction
        self.left_lines, self.right_lines = self.lf.update_driving_surface()

    def set_lines_global(self, camera_x, camera_y, camera_yaw):
        """
        Applies transform of line coordinates into a global frame from the local
        camera frame attached to the ego. Once called it is run over 'left' and
        'right' lines currently identified.
        :param
            camera_x: float - camera x coordinate in a global frame. [m]
            camera_y: float . camera y coordinate in a global frame. [m]
            camera_yaw: float - camera yaw angle (same as ego yaw angle). [rad]
        :return: None
        """
        yaw = camera_yaw

        cos_rot = np.cos(yaw-np.pi/2)
        sin_rot = np.sin(yaw-np.pi/2)

        if self.left_lines is not None:
            self.left_lines_gl = np.copy(self.left_lines)

            x_loc_1 = self.left_lines[:, 0]
            y_loc_1 = self.left_lines[:, 1]
            x_loc_2 = self.left_lines[:, 2]
            y_loc_2 = self.left_lines[:, 3]

            self.left_lines_gl[:, 0] = camera_x + np.multiply(x_loc_1, cos_rot)-np.multiply(y_loc_1, sin_rot)
            self.left_lines_gl[:, 2] = camera_x + np.multiply(x_loc_2, cos_rot)-np.multiply(y_loc_2, sin_rot)

            self.left_lines_gl[:, 1] = camera_y + np.multiply(x_loc_1, sin_rot)+np.multiply(y_loc_1, cos_rot)
            self.left_lines_gl[:, 3] = camera_y + np.multiply(x_loc_2, sin_rot)+np.multiply(y_loc_2, cos_rot)

        else:
            self.left_lines_gl = None

        if self.right_lines is not None:
            self.right_lines_gl = np.copy(self.right_lines)
            x_loc_1 = self.right_lines[:, 0]
            y_loc_1 = self.right_lines[:, 1]
            x_loc_2 = self.right_lines[:, 2]
            y_loc_2 = self.right_lines[:, 3]

            self.right_lines_gl[:, 0] = camera_x + np.multiply(x_loc_1, cos_rot)-np.multiply(y_loc_1, sin_rot)
            self.right_lines_gl[:, 2] = camera_x + np.multiply(x_loc_2, cos_rot)-np.multiply(y_loc_2, sin_rot)

            self.right_lines_gl[:, 1] = camera_y + np.multiply(x_loc_1, sin_rot)+np.multiply(y_loc_1, cos_rot)
            self.right_lines_gl[:, 3] = camera_y + np.multiply(x_loc_2, sin_rot)+np.multiply(y_loc_2, cos_rot)

        else:
            self.right_lines_gl = None

    def get_lines(self):
        """
        Get the line global transforms currently identified.
        :return: tuple
            left_lines_gl : [] - List of projections on a driving surface for 'left' lines.
                Global frame.
                Format [x1,y1,x2,y2] in meters.
            right_lines_gl : [] - List of projections on a driving surface for 'right' lines.
                Global frame.
                Format [x1,y1,x2,y2] in meters.
        """
        return self.left_lines_gl, self.right_lines_gl

    def find_dynamic_obstacles(self):
        """
        Stub
        :return:
        """
        # link here to the lidar perception for obstacles (or any other object class)
        pass

    def get_grid(self):
        """
        Stub
        :return:
        """
        # get grid
        pass
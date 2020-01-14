#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Author: Konstantin Ivanchenko
# Date: December 25, 2019

import cv2
import numpy as np
import copy

class LaneFinder(object):
    def __init__(self):

        # frames as numpy arrays
        # initialized with self.update_frames() method with external call
        self.fr_dep = None
        self.fr_dep_enc = None
        self.fr_sems = None
        self.fr_sems_enc = None

        self.sem_visual_palette = None

        self._delta_k_left_max = 0.57735  # 30°deg
        self._delta_k_right_max = 0.57735  # 30°deg
        self._delta_b_left_max = 30  # pix
        self._delta_b_right_max = 30  # pix
        self._big_slope = 57.5  # in tan

        self._x_max = 800
        self._y_max = 600
        self._fov = 90 * np.pi / 180

        self._cam_angular_res_x = self._fov / self._x_max  # rad/pix
        self._cam_angular_res_y = self._fov / self._y_max  # rad/pix

        self._x_mid = self._x_max / 2
        self._y_mid = self._y_max / 2

    def set_palette(self, palette):
        """
        Set the visualization palette for a semantically segmented image.
        :param:
            palette: Carla palette - palette from the Carla environment.
        :return: None
        """
        self.sem_visual_palette = palette

    def update_frames(self, fdep, fsems):
        """
        Update frames for each of the utilized imaging sensors.
        Called periodically to update sensor data.
        :param:
            fdep: carla.Image object - data from depth camera sensor.
            fsems:carla.Image object - data from depth camera sensor.
        :return: None
        """
        # apply conversion of carla frames into cv frames
        self.fr_dep, self.fr_dep_enc = self.conv_carla_image_data_array(fdep, 'depth')
        self.fr_sems, self.fr_sems_enc = self.conv_carla_image_data_array(fsems, 'rgb_sem')

        # copy frame objects for visualization, as they might require additional post-processing which is
        # not applicable for task
        #fdep_copy = copy.copy(fdep)
        #fsems_copy = copy.copy(fsems)
        #self.visualize_frame_depth(fdep_copy)
        #self.visualize_frame_semseg(fsems_copy)

    def visualize_frame_depth(self, carla_depth_frame):
        """
        Visualize depth camera image.
        Not in use.
        :param: carla.Image object - data from depth camera sensor.
        :return: None
        """
        image = np.ndarray(
            shape=(carla_depth_frame.height, carla_depth_frame.width, 4),
            dtype=np.uint8, buffer=carla_depth_frame.raw_data)
        image = np.clip(image, 0, 100.0)  # limit visualization values to 100.0 meter
        image = cv2.normalize(image, None, 255, 0, cv2.NORM_L2, cv2.CV_32FC1)
        cv2.imshow("Depth", image)
        cv2.waitKey(1)

    def visualize_frame_semseg(self, carla_semseg_frame):
        """
        Visualize semantic segmentation camera image.
        Not in use.
        :param: carla.Image object - data from semantic segmentation camera sensor.
        :return: None
        """
        #sems_rgb = self.fr_sems[:, :, :3]  # remove alpha channel
        # linked with carla ColorConverter with semantic segmentation special color coding
        if self.sem_visual_palette is not None:
            carla_semseg_frame.convert(self.sem_visual_palette)
        image = np.ndarray(
            shape=(carla_semseg_frame.height, carla_semseg_frame.width, 4),
            dtype=np.uint8, buffer=carla_semseg_frame.raw_data)
        cv2.imshow("SemSeg", image)
        cv2.waitKey(1)

    def visualize_postprocessed(self, np_image):
        """
        Simple visualization of an image array.
        :param: np_image : numpy array - image converted into numpy array.
        :return: None
        """
        cv2.imshow("Postproccessed", np_image)
        cv2.waitKey(1)

    def display_lines(self, img, lines):
        """
        Applies identified lines on an image for further visualization.
        :param:
            img: numpy array - image converted into numpy array.
            lines: list - set of identified lines as a list of points
                of a format [x1,y1,x2,y2]
        :return:
            line_image: numpy array - image with set of lines applied.
        """
        line_image = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        return line_image

    def make_points(self, line, vertical_offset_coeff=0.6):
        """
        Calculates two points for the given line.
        :param:
            line : list - line parameters
                Format: [slope, intercept]
        :return:
            Two line points : [[]] - return list of format [x1,x2,y1,y2]
                enveloped in a list.
        """
        slope = line[0]
        intercept = line[1]
        y1 = int(self._y_max - 1)  # bottom of the image
        y2 = int(self._y_max * vertical_offset_coeff)  # lower than the middle
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        # Estimate line coordinates only within the camera image frame
        if x1 >= self._x_max:
            x1 = self._x_max - 1
            y1 = int(x1 * slope + intercept)

        if x1 < 0:
            x1 = 0
            y1 = int(x1 * slope + intercept)

        if x2 >= self._x_max:
            x2 = self._x_max - 1
            y2 = int(x2 * slope + intercept)

        if x2 < 0:
            x2 = 0
            y2 = int(x2 * slope + intercept)

        # additionally trim
        y1 = np.maximum(y1, 0)
        y1 = np.minimum(y1, self._y_max - 1)
        y2 = np.maximum(y2, 0)
        y2 = np.minimum(y2, self._y_max - 1)

        return [[x1, y1, x2, y2]]

    def extract_line_points(self, fit):
        """
        Extracts line points for each line in a fit.
        :param:
            fit - a filtered set of lines of a format [[slope, intercept]] where slope, intercept are floats.
        :return:
            line_points : list - list of  [[x1, y1, x2, y2]] points for each line.
            Note: enveloped into list due to visualization needs.
        """
        line_points = []

        if fit is None:
            return None

        for line in fit:
            line_points.append(self.make_points(line))

        return line_points

    def average_slope_intercept(self, lines):
        """
        Line separation two 'left' and 'right' boundaries by slopes.
        It assumes that each of them has slopes of opposite signs. It applies additional filters
        for slope and offset values to filter away and group them correspondingly.
        Note: additional slope-based grouping may be implemented for reduction of the line amount
         as an optimization possibility.
        Note: adjacent (collinear but offset) road line grouping is to be implemented for future lane
            estimations.
        :param:
            lines:list - list of lines
                Format: [[slope, intercept]]
        :return: tuple
            (left_fit, right_fit, adjacent_fit) : tuple of grouped lines each of a format
                [slope, intercept] where slope, intercept are floats.
        """
        left_fit = []
        right_fit = []
        adjacent_fit = []

        if lines is None:
            return None, None, None

        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]

                if slope < -self._delta_k_left_max:  # y is reversed in image
                    # check here for left
                    slope = np.maximum(slope, -self._big_slope)  ## throw big absolute values for verticals

                    if left_fit:
                        for ix in range(len(left_fit)):
                            # TODO: merge lines by k range as well
                            if np.abs(left_fit[ix][1] - intercept) < self._delta_b_left_max:
                                left_fit[ix][0] = (left_fit[ix][0] + slope) / 2
                                left_fit[ix][1] = (left_fit[ix][1] + intercept) / 2
                                break
                            else:
                                left_fit.append([slope, intercept])
                                break
                    else:
                        left_fit.append([slope, intercept])

                elif slope > self._delta_k_right_max:

                    slope = np.minimum(slope, self._big_slope)  ## throw big absolute values for verticals

                    # check here for right
                    if right_fit:
                        for ix in range(len(right_fit)):
                            # TODO: merge lines by k range as well
                            if np.abs(right_fit[ix][1] - intercept) < self._delta_b_right_max:
                                right_fit[ix][0] = (right_fit[ix][0] + slope) / 2
                                right_fit[ix][1] = (right_fit[ix][1] + intercept) / 2
                                break
                            else:
                                right_fit.append([slope, intercept])
                                break
                    else:
                        right_fit.append([slope, intercept])

                else:
                    # adjacent road lines
                    pass

        return left_fit, right_fit, adjacent_fit

    def line_projection(self, lines):
        """
        Projects lines on a driving surface using a synchronized depth camera. Assumes it is ideal.
        :param:
            lines : [[]] - list of lines of a format [x1,y1,x2,y2].
            Note: requires knowledge on a camera's field of view and angular resolution.
        :return:
            line_proj : [[]] - list of lines of a format [x1,y1,x2,y2] in meters.
        """
        if lines is None:
            return None

        line_proj = np.empty(shape=(len(lines), 4))

        i = 0

        for line in lines:
            # get z-buffer value for given coordinates
            # and convert in x'
            x_1 = line[0][0]
            y_1 = line[0][1]

            try:
                z_1 = self.fr_dep[y_1][x_1]      # y,x - access; in [m]
            except IndexError:
                print("Attempted image coordinates: ", x_1, y_1)

            x_1_mid = -self._x_mid + x_1        # left coordinate is negative ; right is positive
            x_p_1 = -z_1 * np.tan(x_1_mid * self._cam_angular_res_x)  # in [m]

            x_2 = line[0][2]
            y_2 = line[0][3]

            try:
                z_2 = self.fr_dep[y_2][x_2]  # y,x - access; in [m]
            except IndexError:
                print("Attempted image coordinates: ", x_2, y_2)

            x_2_mid = -self._x_mid + x_2
            x_p_2 = -z_2 * np.tan(x_2_mid * self._cam_angular_res_x)  # in [m]
            line_proj[i] = [x_p_1, z_1, x_p_2, z_2]
            i += 1

        return line_proj

    def update_driving_surface(self):
        """
        Periodically called method for the data extraction for the driving surface.
        In the current context it used to estimate driving lane boundaries simplified
        as lines.
        :param: None
        :return: tuple
            (lines_left_proj, lines_right_proj) - left and right line boundaries
                as projections on a driving surface. Format: list of [x1,x2,y1,y2].
        """
        lines_left, lines_right = self.extract_line_data(self.fr_sems)

        lines_left_proj = self.line_projection(lines_left)
        lines_right_proj = self.line_projection(lines_right)

        # finding the lane coordinates as projection on driving surface
        #if lines_left_proj is not None and len(lines_left_proj) > 0:
        #    print("Left line example: ", lines_left_proj[0])

        #if lines_right_proj is not None and len(lines_right_proj) > 0:
        #    print("Right line example: ", lines_right_proj[0])

        return lines_left_proj, lines_right_proj

    def extract_line_data(self, np_image):
        """
        Extracts line data from the visual sensors.
        :param
            np_image : image numpy array - semantically segmented image as a raw array.
        :return: tuple
            left_xy, right_xy : tuple of line points projected on a driving surface
            of a format [x1,x2,y1,y2] in meters in a local ego frame attached to the camera.
        """
        lower_val = np.array([0, 0, 6, 255])  # road line color
        upper_val = np.array([0, 0, 6, 255])  # road line color
        mask = cv2.inRange(np_image, lower_val, upper_val)
        res = cv2.bitwise_not(np_image, np_image, mask=mask)

        res_grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        ## Apply Canny here
        canny = cv2.Canny(res_grey, 50, 150)
        #self.visualize_postprocessed(canny)
        lines = cv2.HoughLinesP(canny, 2, np.pi / 180, 60, np.array([]), minLineLength=30, maxLineGap=100)

        #averaged_lines = average_slope_intercept(np_image, lines)
        #line_image = display_lines(np_image, averaged_lines)
        left_kb, right_kb, _ = self.average_slope_intercept(lines)
        left_xy = self.extract_line_points(left_kb)
        right_xy = self.extract_line_points(right_kb)

        # Visualization
        line_image_left = self.display_lines(np_image, left_xy)
        line_image_right = self.display_lines(np_image, right_xy)
        line_image = cv2.addWeighted(line_image_left, 1, line_image_right, 1, 1)
        combined_image = cv2.addWeighted(np_image, 0.8, line_image, 1, 1)
        self.visualize_postprocessed(combined_image)
        # ~Visualization

        return left_xy, right_xy

    def conv_carla_image_data_array(self, carla_image, image_type = 'rgb'):
        """
        Converts carla image to a numpy data array.
        For example:
        The RGB camera provides a 4-channel int8 color format (bgra).
        The DEPTH camera provides a 1-channel float32 data. Converted to [0..1000] meters
        The SEMANTIC SEGMENTATION camera provides a 4-channel int8 color format (bgra).

        :param:
            carla_image: carla.Image object
            image_type : string as 'rgb', 'rgb_sem', 'depth' denoting class of image frame
        :return: tuple:
            image : numpy data array containing the image information
            encoding : string
        """

        image = np.ndarray(
            shape=(carla_image.height, carla_image.width, 4),
            dtype=np.uint8, buffer=carla_image.raw_data)

        if image_type == 'rgb':
            encoding = 'bgra8'
        if image_type == 'rgb_sem':
            encoding = 'bgra8'
        elif image_type == 'depth':
            encoding = '32fc1'

            # From CARLA docs:
            # The image codifies the depth in 3 channels of the RGB color space,
            # from less to more significant bytes: R -> G -> B. The actual distance in meters can be decoded with
            # normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            # in_meters = 1000 * normalized
            #
            # https://carla.readthedocs.io/en/latest/cameras_and_sensors/#sensorcameradepth

            coeff = (np.array([65536.0, 256.0, 1.0, 0]) / (256**3-1))*1000
            image = np.dot(image, coeff).astype(np.float32)

        return image, encoding
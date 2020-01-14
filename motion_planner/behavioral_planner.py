#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Author: Ryan De Iaco
# Additional Comments: Carlos Wang
# Date: November 21, 2018
# Modified: Konstantin Ivanchenko
# Date: December 27, 2018

import numpy as np
import math
import common as cn

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
# Stop speed threshold
STOP_THRESHOLD = 0.9
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10


class BehaviouralPlanner:
    def __init__(self, lookahead=8.0, lookahead_time=2.0, lead_vehicle_lookahead=20.0, stopsign_fences=None):

        self.lookahead_dist = lookahead  # m
        self.lookahead_time = lookahead_time  # s ##3.0
        self.lookahead_lead_vcl_time = lead_vehicle_lookahead  # m

        self._lookahead = lookahead
        self._stopsign_fences = stopsign_fences
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state = FOLLOW_LANE
        self._follow_lead_vehicle = False
        self._goal_state = cn.Waypoint(0, 0, 0, 0)
        self._goal_index = 0
        self._stop_count = 0

        self._stop_sign_found = False  # This variable has been moved to a private global

        self._DEBUG_STOP_COUNTER = 0
        self._DEBUG_FENCE_CROSSED = False

    def get_goal_state_ix(self):
        """
        Get goal index.
        :param: None
        :return:
            self._goal_index : int - index of the goal state within global waypoint list
        """
        return self._goal_index


    def get_goal_state(self):
        """
        Get goal state.
        :param: None
        :return:
            self._goal_state : Waypoint object - goal state as the global waypoint object
        """
        return self._goal_state


    def set_lookahead(self, speed):
        """
        Adjust lookahead based on current speed. To be called within the main control loop
        :param:
            speed : float - current speed of the ego vehicle
        :return: None
        """
        self._lookahead = speed*self.lookahead_time + self.lookahead_dist

    def transition_state(self, waypoints, ego_state):
        """Handles state global transitions and computes the goal state.
            This is to modified in future to handle more top-level behavior rules.
        :param:
            waypoints: current waypoints to track (global frame).
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[Waypoint]]
            ego_state: ego state vector for the vehicle. (global frame)
                format: Egostate object
        :returns: None
        """

        if self._state == FOLLOW_LANE:
            closest_len, closest_index = self.get_closest_index(waypoints, ego_state)

            goal_index = self.get_goal_index(waypoints, closest_len, closest_index)
            # if not self._DEBUG_FENCE_CROSSED:
            goal_index, self._stop_sign_found = self.check_for_stop_signs(waypoints, closest_index, goal_index)
            self._goal_index = goal_index

            self._goal_state = waypoints[goal_index]

            # print("---Begin debug state list FOLLOW LANE---")
            # print("Ego state:     ", ego_state)
            # print("Goal state:    ", self._goal_state)
            # print("Closest index: ", closest_index)
            # print("Goal index:    ", goal_index)
            # print("---End debug state list FOLLOW LANE---")

            if self._stop_sign_found and not self._DEBUG_FENCE_CROSSED:
                self._DEBUG_FENCE_CROSSED = True
                self._state = DECELERATE_TO_STOP
                self._DEBUG_STOP_COUNTER += 1
                print("DEBUG STOP COUNTER: !!!!!!!!!", self._DEBUG_STOP_COUNTER)

        elif self._state == DECELERATE_TO_STOP:

            # print("---Begin debug state list DECELERATE---")
            # print("Ego state:     ", ego_state)
            # print("Goal state:    ", self._goal_state)
            # print("Goal index:    ", self._goal_index)
            # print("---End debug state list DECELERATE---")

            if ego_state.s <= STOP_THRESHOLD:  # check against real close loop speed
                self._state = STAY_STOPPED

        elif self._state == STAY_STOPPED:
            if self._stop_count == STOP_COUNTS:
                closest_len, closest_index = self.get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, closest_len, closest_index)

                self._stop_sign_found = False
                # TODO: check if not 'self.check_for_stop_signs(waypoints, closest_index, goal_index)'
                print("After STOP counter is over the closest len is ", closest_len, " and the closest index is ",
                      closest_index)
                print("Goal index after STOP counter is over:  ", goal_index)
                print("Goal state is ", waypoints[goal_index])

                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]  # TODO: check if the _goal_state expects the same as waypoints

                # print("---Begin debug state list AFTER STOP---")
                # print("Ego state:     ", ego_state)
                # print("Goal state:    ", self._goal_state)
                # print("Closest index: ", closest_index)
                # print("Goal index:    ", goal_index)
                # print("---End debug state list AFTER STOP---")

                if not self._stop_sign_found:
                    self._stop_count = 0
                    self._state = FOLLOW_LANE

            # Otherwise, continue counting.
            else:
                self._stop_count += 1
                print("STAYING STOPPED FOR:  ", self._stop_count)
        else:
            raise ValueError('Invalid state value.')

    def get_goal_index(self, waypoints, closest_len, closest_index):
        """Gets the goal index for the vehicle.

        Set to be the earliest waypoint that has accumulated arc length
        greater than or equal to self._lookahead.

        :param:
            waypoints: - current waypoints to track (global frame).
                (includes speed to track at each x,y location.)
                format: [[Waypoint]]
            closest_len: float - length (m) to the closest waypoint from the vehicle.
            closest_index: int - index of the waypoint which is closest to the vehicle.
        :returns:
            wp_index: int - Goal index for the vehicle to reach
        """
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        while wp_index < len(waypoints) - 1 and arc_length < self._lookahead:
            arc_length += np.linalg.norm([waypoints[wp_index + 1].x - waypoints[wp_index].x,
                                          waypoints[wp_index + 1].y - waypoints[wp_index].y])
            wp_index += 1

        return wp_index

    def check_for_stop_signs(self, waypoints, closest_index, goal_index):
        """Checks for a stop sign that is intervening the goal path.

        Checks for a stop sign that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a stop line), and a
        boolean flag indicating if a stop sign obstruction was found.

        #TODO:requires vectorization.
        #TODO: not in use in the current version.

        :param:
            waypoints: - current waypoints to track (global frame).
                (includes speed to track at each x,y location.)
                format: [[Waypoint]]
            closest_index: int - index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
            goal_index (current): int - Current goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
        :returns:
            Tuple
                goal_index : int - goal waypoint index
                bool - True if the trajectory intersects the stop line
        """

        if self._stopsign_fences is None:
            return goal_index, False

        for i in range(closest_index + 1, goal_index):  # TODO: closest index+1; otherwise car may be identify the same
            # TODO: stop sign twice while passing the 'fence'

            # Check to see if path segment crosses any of the stop lines.
            intersect_flag = False
            for stopsign_fence in self._stopsign_fences:
                wp_1 = np.array([waypoints[i].x, waypoints[i].y, waypoints[i].s])
                wp_2 = np.array([waypoints[i+1].x, waypoints[i+1].y, waypoints[i+1].s])
                s_1 = np.array(stopsign_fence[0:2])
                s_2 = np.array(stopsign_fence[2:4])

                v1 = np.subtract(wp_2, wp_1)
                v2 = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2 = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1 = np.subtract(s_2, s_1)
                v2 = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2 = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))

                # Check if the line segments intersect.
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True

                # Check if the collinearity cases hold.
                if (sign_1 == 0) and pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True

                # If there is an intersection with a stop line, update
                # the goal state to stop before the goal line.
                if intersect_flag:
                    goal_index = i
                    return goal_index, True

        return goal_index, False

    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.
        TODO: not in use in the current version.

        :param:
            ego_state: ego state vector for the vehicle. (global frame)
                format: Egostate object
            lead_car_position: The position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        if not self._follow_lead_vehicle:

            lead_car_delta_vector = [lead_car_position.x - ego_state.x,
                                     lead_car_position.y - ego_state.y]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector,
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state.yaw),
                                  math.sin(ego_state.yaw)]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector,
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position.x - ego_state.x,
                                     lead_car_position.y - ego_state.y]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state.yaw), math.sin(ego_state.yaw)]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False

    def get_closest_index(self, waypoints, ego_state):
        """Gets closest index a given list of waypoints to the vehicle position.
        :param:
            waypoints: - current waypoints to track (global frame).
                (includes speed to track at each x,y location.)
                format: [[Waypoint]]
            ego_state: ego state vector for the vehicle. (global frame)
                format: Egostate object
        :returns:
            Tuple
                closest_len: length (m) to the closest waypoint from the vehicle.
                closest_index: index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        """
        closest_len = float('Inf')
        closest_index = 0

        for index in range(len(waypoints) - 1):
            len_i = np.linalg.norm([waypoints[index].x - ego_state.x, waypoints[index].y - ego_state.y])
            if len_i < closest_len:
                closest_len = len_i
                closest_index = index

        return closest_len, closest_index

def pointOnSegment(p1, p2, p3):
    """
    # Checks if p2 lies on segment p1-p3.
    :param:
        p1, p2, P3: np.array as [x, y, s] - points
    :return: bool - True of p2 is on  the p1-p3 .segment
    """
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
            (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False

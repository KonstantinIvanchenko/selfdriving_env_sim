#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Author: Ryan De Iaco
# Additional Comments: Carlos Wang
# Date: October 29, 2018
#
# Modified work: Konstantin Ivanchenko
# Date: December 25, 2019

import numpy as np
import copy
from motion_planner import collision_checker, path_optimizer, velocity_planner
from math import sin, cos, pi
import common as cn

import multiprocessing
from joblib import Parallel, delayed


class LocalPlanner:
    def __init__(self, num_paths=7,
                 path_offset=1.0,  # m
                 circle_offsets=[-1.0,1.0,3.0],  # m
                 circle_radii=[1.5,1.5,1.5],  # m
                 path_select_weight=10,
                 time_gap=1.0,  # s
                 a_max=0.5,  # m/s^^2
                 slow_speed=2.0,  # m/s
                 stop_line_buffer=3.5,  # m
                 turn_dead_angle=0.1):  # rad

        self._num_paths = num_paths
        self._path_offset = path_offset
        self._path_optimizer = path_optimizer.PathOptimizer()
        self._collision_checker = collision_checker.CollisionChecker(circle_offsets, circle_radii, path_select_weight)
        self._velocity_planner = velocity_planner.VelocityPlanner(time_gap, a_max, slow_speed, stop_line_buffer)
        self._prev_best_path = None

        self.heading_curr = 0
        self.goal_t = 0
        self.turn_dead_angle = turn_dead_angle

    def get_num_path(self):
        """
        Returns number of paths. Used on a global context.
        :return: None
        :param: _num_paths : int - number of locally planned paths.
        """
        return self._num_paths

    def get_goal_state_set(self, goal_index, waypoints, ego_state):
        """Gets the goal states given a goal position.
        :param:
            goal_index: Goal index for the vehicle to reach
                i.e. waypoints[goal_index] gives the goal waypoint
            waypoints: current waypoints to track. length and speed in m and m/s.
                (includes speed to track at each x,y location.) (global frame)
                format: [[Waypoint]]
                example:
                    waypoints[2].y:
                    returns the 3rd waypoint's y position
                    waypoints[5]:
                    returns (6th waypoint)
            ego_state: ego state vector for the vehicle, in the global frame.
                format: Egostate object
                    Egostate.x and Egostate.y     : position (m)
                    Egostate.yaw             : top-down orientation [-pi to pi]
                    Egostate.s : close loop speed (m/s)
        :return:
            goal_state_set: Set of goal states (offsetted laterally from one
                another) to be used by the local planner to plan multiple
                proposal paths. This goal state set is in the vehicle frame.
                format: [Waypoint]
                , where len is the total number of goal states;
                  all units are in m, m/s and radians
        """
        # Compute the final heading based on the next index.
        # If the goal index is the last in the set of waypoints, use
        # the previous index instead.
        if goal_index >= len(waypoints) - 1:
            # goal_index -= 1
            heading = self.heading_curr
        else:
            delta_x = waypoints[goal_index].x - waypoints[goal_index+1].x
            delta_y = waypoints[goal_index].y - waypoints[goal_index+1].y
            heading = np.arctan2(-delta_y, -delta_x)

            #if -pi < heading <= 0:
            #    heading += pi
            #elif 0 < heading <= pi:
            #    heading -= pi

            self.heading_curr = heading


        # Compute the center goal state in the local frame using 
        # the ego state. The following code will transform the input
        # goal state to the ego vehicle's local frame.
        # The goal state will be of the form (x, y, t, v).
        goal_state_local = copy.copy(waypoints[goal_index])

        # Translate so the ego state is at the origin in the new frame.
        # This is done by subtracting the ego_state from the goal_state_local.
        goal_state_local.x -= ego_state.x
        goal_state_local.y -= ego_state.y

        # Rotate such that the ego state has zero heading in the new frame.
        # Recall that the general rotation matrix is [cos(theta) -sin(theta)
        #                                             sin(theta)  cos(theta)]
        cos_rot = np.cos(-ego_state.yaw)
        sin_rot = np.sin(-ego_state.yaw)

        goal_x = np.add(np.multiply(goal_state_local.x, cos_rot), -np.multiply(goal_state_local.y, sin_rot))
        goal_y = np.add(np.multiply(goal_state_local.x, sin_rot), np.multiply(goal_state_local.y, cos_rot))

        # Compute the goal yaw in the local frame by subtracting off the 
        # current ego yaw from the heading variable.
        self.goal_t = heading - ego_state.yaw

        # Velocity is preserved after the transformation.
        goal_v = waypoints[goal_index].s

        # Keep the goal heading within [-pi, pi] for the optimizer.
        if self.goal_t > pi:
            self.goal_t -= 2*pi
        elif self.goal_t < -pi:
            self.goal_t += 2*pi

        # Compute and apply the offset for each path such that
        # all of the paths have the same heading of the goal state, 
        # but are laterally offset with respect to the goal heading.
        goal_state_set = []
        for i in range(self._num_paths):
            # Compute offsets that span the number of paths set for the local
            # planner. Each offset goal will be used to generate a potential
            # path to be considered by the local planner.
            offset = (i - self._num_paths // 2) * self._path_offset

            # Compute the projection of the lateral offset along the x
            # and y axis. To do this, multiply the offset by cos(goal_theta + pi/2)
            # and sin(goal_theta + pi/2), respectively.
            x_offset = np.multiply(offset, np.cos(self.goal_t + pi/2))
            y_offset = np.multiply(offset, np.sin(self.goal_t + pi/2))

            goal_state_set.append(cn.WaypointLocal(goal_x + x_offset,
                                                    goal_y + y_offset,
                                                    self.goal_t,
                                                    goal_v))
           
        return goal_state_set

    def get_goal_direction(self):
        """
        Get local goal direction respectively to the current yaw angle.
        Checks self.goal_t float angle within [-pi, pi] range.
        :param: None
        :return:
            direction : int, [-1 - left, 0 - straight, 1 - right]
        """
        if -pi <= self.goal_t < -self.turn_dead_angle:
            return -1
        elif -self.turn_dead_angle <= self.goal_t < self.turn_dead_angle:
            return 0
        else:
            return 1

    def run_optimizer(self, goal_state):
        """
        Method for running optimizer in parallel mode. Suited for mp pool.
        Used within plan_paths method
        :param:
            goal_state - list of a format [x_points, y_points, t_points]
        :return:
            path - list of a format [[x_points, y_points, t_points]]
        """
        path = self._path_optimizer.optimize_spiral(goal_state.x,
                                                    goal_state.y,
                                                    goal_state.yaw)
        return path

    def plan_paths(self, goal_state_set):
        """
        Plans the path set using the polynomial spiral optimization.
        Uses polynomial spiral optimization to each of the
        goal states.

        :param:
            goal_state_set: Set of goal states (offsetted laterally from one
                another) to be used by the local planner to plan multiple
                proposal paths. These goals are with respect to the vehicle
                frame.
                format: [WaypointLocal object]
                WaypointLocal.x and WaypointLocal.y     : position (m)
                WaypointLocal.yaw             : top-down orientation [-pi to pi]
                WaypointLocal.s : open loop speed (m/s)
        :return:
            paths: A list of optimized spiral paths which satisfies the set of 
                goal states. A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m) along the spiral
                        y_points: List of y values (m) along the spiral
                        t_points: List of yaw values (rad) along the spiral
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
                Note that this path is in the vehicle frame, since the
                optimize_spiral function assumes this to be the case.
            path_validity: List of booleans classifying whether a path is valid
                (true) or not (false) for the local planner to traverse. Each ith
                path_validity corresponds to the ith path in the path list.
        """
        paths         = []
        path_validity = []

        mp_pool = multiprocessing.Pool(3)
        raw_path = mp_pool.map(self.run_optimizer, [goal_state_set[i] for i in range(len(goal_state_set))])

        while len(raw_path) != self._num_paths:
            continue

        mp_pool.close()

        print("LP: raw_path len is ", len(raw_path))

        # len(raw_path) = len(goal_state_set)
        for ix in range(len(goal_state_set)):
            if np.linalg.norm([raw_path[ix][0][-1] - goal_state_set[ix].x,
                               raw_path[ix][1][-1] - goal_state_set[ix].y,
                               raw_path[ix][2][-1] - goal_state_set[ix].yaw]) > 0.1:
                path_validity.append(False)
            else:
                paths.append(raw_path[ix])
                path_validity.append(True)

        return paths, path_validity

    def run_transform(self, args):
        """
        Method for running transformer in parallel mode. Suited for mp pool.
        Used within transform_paths method
        :param:
            ego_state - Egostate object containing the current ego state in the global frame.
            path - list of a format [[x_points, y_points, t_points]]
        :return:
            [x_transformed, y_transformed, t_transformed] - list of lists for x,y,t in global frame. Forms path in the
            global frame.
        """
        x_transformed = []
        y_transformed = []
        t_transformed = []

        path, ego_state = args

        for i in range(len(path[0])):
            x_transformed.append(ego_state.x + path[0][i] * cos(ego_state.yaw) - path[1][i] * sin(ego_state.yaw))
            y_transformed.append(ego_state.y + path[0][i] * sin(ego_state.yaw) + path[1][i] * cos(ego_state.yaw))
            t_transformed.append(path[2][i] + ego_state.yaw)

        return [x_transformed, y_transformed, t_transformed]

    def transform_paths(self, paths, ego_state):
        """
        Converts the to the global coordinate frame.
        :param:
            paths: A list of paths in the local (vehicle) frame.
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        , x_points: List of x values (m)
                        , y_points: List of y values (m)
                        , t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            ego_state: ego state vector for the vehicle, in the global frame.
                format: Egostate object
                    Egostate.x and Egostate.y     : position (m)
                    Egostate.yaw             : top-down orientation [-pi to pi]
                    Egostate.s : close loop speed (m/s)
        :return:
            transformed_paths: A list of transformed paths in the global frame.
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        , x_points: List of x values (m)
                        , y_points: List of y values (m)
                        , t_points: List of yaw values (rad)
                    Example of accessing the ith transformed path, jth point's
                    y value:
                        paths[i][1][j]
        """
        mp_pool = multiprocessing.Pool(3)
        args = [(paths[i], ego_state) for i in range(len(paths))]
        transformed_paths = mp_pool.map(self.run_transform, args)
        mp_pool.close()

        return transformed_paths

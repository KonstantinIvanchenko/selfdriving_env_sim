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
import scipy.spatial
from math import sin, cos, pi, sqrt

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii   = circle_radii
        self._weight         = weight
        self._lane_norm_max_collision = 0.5  # in [m]


    def collision_check(self, paths, obstacles):
        """
        Returns a bool array on whether each path is collision free.
        :param:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            obstacles: A list of [x, y] points that represent points along the
                border of obstacles, in the global frame.
                Format: [[x0, y0],
                         [x1, y1],
                         ...,
                         [xn, yn]]
                , where n is the number of obstacle points and units are [m, m]
        :return:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
        """
        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            path           = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):
                # Compute the circle locations along this point in the path.
                # These circle represent an approximate collision
                # border for the vehicle, which will be used to check
                # for any potential collisions along each path with obstacles.

                # The circle offsets are given by self._circle_offsets.
                # The circle offsets need to placed at each point along the path,
                # with the offset rotated by the yaw of the vehicle.
                # Each path is of the form [[x_values], [y_values],
                # [theta_values]], where each of x_values, y_values, and
                # theta_values are in sequential order.

                # circle_x = point_x + circle_offset*cos(yaw)
                # circle_y = point_y circle_offset*sin(yaw)
                # for each point along the path.
                # point_x is given by path[0][j], and point _y is given by
                # path[1][j]. 
                circle_locations = np.zeros((len(self._circle_offsets), 2))

                # By default three circles are applied around each path point
                circle_locations[:, 0] = np.add(path[0][j], np.multiply(self._circle_offsets[:], np.cos(path[2][j])))
                circle_locations[:, 1] = np.add(path[1][j], np.multiply(self._circle_offsets[:], np.sin(path[2][j])))


                # Assumes each obstacle is approximated by a collection of
                # points of the form [x, y].
                # Here, we will iterate through the obstacle points, and check
                # if any of the obstacle points lies within any of our circles.
                # If so, then the path will collide with an obstacle and
                # the collision_free flag should be set to false for this flag
                for k in range(len(obstacles)):
                    collision_dists = scipy.spatial.distance.cdist(obstacles[k], circle_locations)
                    collision_dists = np.subtract(collision_dists, self._circle_radii)
                    collision_free = collision_free and not np.any(collision_dists < 0)

                    if not collision_free:
                        break
                if not collision_free:
                    break

            collision_check_array[i] = collision_free

        return collision_check_array

    def lane_boundary_check(self, paths, collision_check_array, lanes):
        """
        Returns a bool array on whether each path intersects a set of lane boundaries.
        :param:
            paths: A list of paths in the global frame.
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            collision check array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false).
                If it is already no free, skip checking for the lane boundary
                intersection.
            lanes: A list of lists of the format [[x1, y1, x2, y2]]
                    that contain two points identifying the driving lane boundary.
        :return:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
        """
        if lanes is None:
            print("is none")
            return collision_check_array

        if len(paths) != len(collision_check_array):
            print("not equal")
            return collision_check_array

        for i in range(len(paths)):
            if collision_check_array[i]: #is True:  # if path is collision free check if there is line crossing
                ## check further on lane border crossing
                for j in range(len(paths[i][0])):
                    px = paths[i][0][j]
                    py = paths[i][1][j]

                    for l in range(len(lanes)):
                        dxa = lanes[l][2] - lanes[l][0]
                        dya = lanes[l][3] - lanes[l][1]
                        dxp = lanes[l][0] - px
                        dyp = lanes[l][1] - py
                        dist = np.abs(dxa*dyp - dxp*dya)/np.linalg.norm([dxa, dya])

                        # print("Distance norm: ", dist)

                        if dist < self._lane_norm_max_collision:
                            collision_check_array[i] = False

        print("Collision check: ", collision_check_array)

        return collision_check_array

    def select_best_path_index(self, paths, collision_check_array, goal_state):
        """Returns the path index which is best suited for the vehicle to
        traverse. Selects a path index which is closest to the center line as well as far
        away from collision paths.

        :param:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
            goal_state: Waypoint object- Goal state for the vehicle to reach (centerline goal).
        :return:
            best_index: The path index which is best suited for the vehicle to
                navigate with.
        """
        best_index = None
        best_score = float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i]:
                # Compute the "distance from centerline" score.
                # The centerline goal is given by goal_state.
                # The exact choice of objective function is up to you.
                # A lower score implies a more suitable path.
                # Last point sqrt((X-X_goal)**2 + (Y-Y_goal)**2) value

                # index_0_2 = int(len(paths[i])*0.2)
                index_count = len(paths[i][0])-1
                index_0 = int(index_count*0.25)
                index_1 = int(index_count*0.5)
                index_2 = int(index_count*0.75)
                index_3 = index_count

                # score = np.linalg.norm([paths[i][0][-1] - goal_state[0], paths[i][1][-1] - goal_state[1]])
                score = 4*np.linalg.norm([paths[i][0][index_0] - goal_state.x, paths[i][1][index_0] - goal_state.y])
                score += 2*np.linalg.norm([paths[i][0][index_1] - goal_state.x, paths[i][1][index_1] - goal_state.y])
                score += np.linalg.norm([paths[i][0][index_2] - goal_state.x, paths[i][1][index_2] - goal_state.y])
                # score += np.linalg.norm([paths[i][0][index_3] - goal_state[0], paths[i][1][index_3] - goal_state[1]])

                # Compute the "proximity to other colliding paths" score and
                # add it to the "distance from centerline" score.
                for j in range(len(paths)):
                    if j == i:
                        continue
                    else:
                        if not collision_check_array[j]:
                            # Penalize for the proximity to a path with obstacle; by 10x distance it; adjustment may be
                            # required
                            score += self._weight * np.linalg.norm([paths[i][0][-1] - paths[j][0][-1],
                                                              paths[i][1][-1] - paths[j][1][-1]])

                            pass

            # Handle the case of colliding paths.
            else:
                score = float('Inf')
                
            # Set the best index to be the path index with the lowest score
            if score < best_score:
                best_score = score
                best_index = i

        return best_index

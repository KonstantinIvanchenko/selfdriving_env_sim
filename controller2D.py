#!/usr/bin/env python3

#Based on standard source code from CARLA demo.
#
#Controller2D implements standard velocity and track follower based on:
#--'Bicycle' vehicle model simplification with P- heading controller
#--PID velocity controller
#
# Modified work: Konstantin Ivanchenko
# Date: December 25, 2019

import numpy as np
import common as cn

class Controller2D(object):
    def __init__(self, waypoints):
        self._lookahead_distance = 3.0
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_timestamp = 0
        self._set_throttle = 0
        self._set_brake = 0
        self._set_steer = 0
        self._waypoints = waypoints
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi

        self.var_kp = 0.30  #0.9#0.50
        self.var_ki = 2.0  #0.5#0.30
        self.var_integrator_min = 0.0
        self.var_integrator_max = 100000.0  #10.0
        self.var_kd = 0.00000000001  #0.13
        self.var_kp_heading = 0.5 #
        #self.var_ki_heading = 0.001
        self.var_k_speed_crosstrack = 0.5  #0.1
        self.var_cross_track_deadband = 0.001  # 0.01
        self.var_x_prev = 0.0
        self.var_y_prev = 0.0
        self.var_yaw_prev = 0.0
        self.var_v_prev = 0.0
        self.var_t_prev = 0.0
        self.var_v_error = 0.0
        self.var_v_error_prev = 0.0
        self.var_v_error_integral = 0.0

    def update_values(self, ego_state):
        self._current_x = ego_state.x
        self._current_y = ego_state.y
        self._current_yaw = ego_state.yaw
        self._current_speed = ego_state.vel
        self._current_timestamp = ego_state.ts/1000

    def get_lookahead_index(self, lookahead_distance):
        min_idx = 0
        min_dist = float("inf")
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i].x - self._current_x,
                self._waypoints[i].y - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        total_dist = min_dist
        lookahead_idx = min_idx
        for i in range(min_idx + 1, len(self._waypoints)):
            if total_dist >= lookahead_distance:
                break
            total_dist += np.linalg.norm(np.array([
                self._waypoints[i].x - self._waypoints[i - 1].x,
                self._waypoints[i].y - self._waypoints[i - 1].y]))
            lookahead_idx = i
        return lookahead_idx

    def update_desired_speed(self):
        min_idx = 0
        min_dist = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i].x - self._current_x,
                self._waypoints[i].y - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        self._desired_speed = self._waypoints[min_idx+1].s        #FIXME: index is changed to +1

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_steer, self._set_brake, self._set_throttle

    def get_desired_speed(self):
        return self._desired_speed

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        self.update_desired_speed()
        v_desired = self._desired_speed
        t = self._current_timestamp
        waypoints = self._waypoints
        throttle_output = 0
        steer_output = 0
        brake_output = 0

        self.var_v_error = v_desired - v
        self.var_v_error_integral += self.var_v_error * (t - self.var_t_prev)
        v_error_rate_of_change = (self.var_v_error - self.var_v_error_prev) / (t - self.var_t_prev)

        # cap the integrator sum to a min/max
        self.var_v_error_integral = np.fmax(np.fmin(self.var_v_error_integral,
                                                    self.var_integrator_max),
                                                    self.var_integrator_min)

        throttle_output = self.var_kp * self.var_v_error + \
                          self.var_ki * self.var_v_error_integral + \
                          self.var_kd * v_error_rate_of_change

        #print("t= ", t)
        #print("self.var_v_error= ", self.var_v_error)
        #print("self.var_v_error_integral", self.var_v_error_integral)
        #print("----------------------------------------------------")

        # Find cross track error (assume point with closest distance)
        crosstrack_error = float("inf")
        crosstrack_vector = np.array([float("inf"), float("inf")])

        ce_idx = self.get_lookahead_index(self._lookahead_distance)
        crosstrack_vector = np.array([waypoints[ce_idx].x - x - self._lookahead_distance * np.cos(yaw),
                                      waypoints[ce_idx].y - y - self._lookahead_distance * np.sin(yaw)])
        crosstrack_error = np.linalg.norm(crosstrack_vector)

        # set deadband to reduce oscillations
        # print(crosstrack_error)
        if crosstrack_error < self.var_cross_track_deadband:
            crosstrack_error = 0.0

        # Compute the sign of the crosstrack error
        crosstrack_heading = np.arctan2(crosstrack_vector[1], crosstrack_vector[0])
        crosstrack_heading_error = crosstrack_heading - yaw
        crosstrack_heading_error = (crosstrack_heading_error + self._pi) % self._2pi - self._pi

        crosstrack_sign = np.sign(crosstrack_heading_error)

        # Compute heading relative to trajectory (heading error)
        # First ensure that we are not at the last index. If we are,
        # flip back to the first index (loop the waypoints)
        if ce_idx < len(waypoints) - 1:
            vect_wp0_to_wp1 = np.array(
                [waypoints[ce_idx + 1].x - waypoints[ce_idx].x,
                 waypoints[ce_idx + 1].y - waypoints[ce_idx].y])
            trajectory_heading = np.arctan2(vect_wp0_to_wp1[1],
                                            vect_wp0_to_wp1[0])
        else:
            vect_wp0_to_wp1 = np.array(
                [waypoints[0].x - waypoints[-1].x,
                 waypoints[0].y - waypoints[-1].y])
            trajectory_heading = np.arctan2(vect_wp0_to_wp1[1],
                                            vect_wp0_to_wp1[0])

        heading_error = trajectory_heading - yaw
        heading_error = (heading_error + self._pi) % self._2pi - self._pi

        # if v + self.var_k_speed_crosstrack == 0

        if v <= 1e-1:
            steer_output = 0
        else:
            # Compute steering command based on error
            steer_output = heading_error + np.arctan(self.var_kp_heading * crosstrack_sign * crosstrack_error /
                                                     (v + self.var_k_speed_crosstrack))

        ######################################################
        # SET CONTROLS OUTPUT
        ######################################################
        self.set_throttle(throttle_output)  # in percent (0 to 1)
        self.set_steer(steer_output)  # in rad (-1.22 to 1.22)
        self.set_brake(brake_output)  # in percent (0 to 1)

        self.var_x_prev = x
        self.var_y_prev = y
        self.var_yaw_prev = yaw
        self.var_v_prev = v
        self.var_v_error_prev = self.var_v_error
        self.var_t_prev = t

    # TODO: vectorize
    def fine_interpolation(self, local_waypoints, fine_interp_dist = 0.01):
        if local_waypoints != None:
            # Update the controller waypoint path with the best local path.
            wp_distance = []  # distance array
            # local_waypoints_np = np.array(local_waypoints)
            for i in range(1, len(local_waypoints)):
                wp_distance.append(
                    np.sqrt((local_waypoints[i].x - local_waypoints[i - 1].x) ** 2 +
                            (local_waypoints[i].y - local_waypoints[i - 1].y) ** 2))
            wp_distance.append(0)  # last distance is 0 because it is the distance
            # from the last waypoint to the last waypoint

            # Linearly interpolate between waypoints and store in a list
            wp_interp = []  # interpolated values
            # (rows = waypoints, columns = [x, y, v])
            for i in range(len(local_waypoints) - 1):
                # Add original waypoint to interpolated waypoints list (and append it to the list)
                wp_interp.append(local_waypoints[i])

                # Interpolate to the next waypoint. First compute the number of
                # points to interpolate based on the desired resolution and
                # incrementally add interpolated points until the next waypoint
                # is about to be reached.
                num_pts_to_interp = int(np.floor(wp_distance[i] / float(fine_interp_dist)) - 1)

                wp_vector_dx = local_waypoints[i + 1].x - local_waypoints[i].x
                wp_vector_dy = local_waypoints[i + 1].y - local_waypoints[i].y
                wp_vector_ds = local_waypoints[i + 1].s - local_waypoints[i].s
                wp_vector_norm = np.sqrt(wp_vector_dx ** 2 + wp_vector_dy ** 2 + wp_vector_ds ** 2)

                wp_uvector_dx = wp_vector_dx / wp_vector_norm
                wp_uvector_dy = wp_vector_dy / wp_vector_norm
                wp_uvector_ds = wp_vector_ds / wp_vector_norm

                for j in range(num_pts_to_interp):
                    next_wp_vector_dx = fine_interp_dist * float(j + 1) * wp_uvector_dx
                    next_wp_vector_dy = fine_interp_dist * float(j + 1) * wp_uvector_dy
                    next_wp_vector_ds = fine_interp_dist * float(j + 1) * wp_uvector_ds
                    wp_interp.append(cn.Waypoint(local_waypoints[i].x + next_wp_vector_dx,
                                                 local_waypoints[i].y + next_wp_vector_dy,
                                                 local_waypoints[i].s + next_wp_vector_ds))
            # add last waypoint at the end
            wp_interp.append(local_waypoints[-1])

            return wp_interp
        else:
            return local_waypoints

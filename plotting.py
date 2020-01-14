#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modified work: Konstantin Ivanchenko
# Date: December 25, 2019

from external_source import live_plotter as lp


class MotionPlotter(object):
    def __init__(self, waypoints, num_local_path, trajectory = True, controls = True):
        self.tr_plot = lp.LivePlotter(tk_title="Trajectory Visualization")
        self.ct_plot = lp.LivePlotter(tk_title="Control Outputs")

        self.waypoints = waypoints
        self.wx = [waypoints[i].x for i in range(len(waypoints))]
        self.wy = [waypoints[i].y for i in range(len(waypoints))]

        #TODO: immplement dynamic window sizing
        self.win_size = len(self.wx)*10

        self.fig_names = {"waypoints" : "waypoints",
                          "trajectory" : "trajectory",
                          "ego_a_pos" : "ego_A_pos",
                          "ego_b_pos" : "ego_B_pos",
                          "ego" : "ego",
                          "speed" : "speed",
                          "des_speed" : "des_speed",
                          "throttle" : "throttle",
                          "brake" : "brake",
                          "steer" : "steer",
                          "paths" : "local_path",
                          "llines" : "llines",
                          "rlines" : "rlines"}

        if trajectory is True and waypoints is not None:
            self.tr_fig = self.tr_plot.plot_new_dynamic_2d_figure(
                title='Motion Trajectory',
                figsize=(8, 8),  # inches
                edgecolor="black",
                rect=[0.1, 0.1, 0.8, 0.8]  # inches
            )

            self.tr_fig.set_invert_x_axis()  # to be inverted due to reversed coordinates in Carla
            self.tr_fig.set_axis_equal()
            ###
            for i in range(20):
                self.tr_fig.add_graph(self.fig_names["llines"]+str(i), window_size=200,
                                    x0=None, y0=None, color='b')
            ###
            for i in range(20):
                self.tr_fig.add_graph(self.fig_names["rlines"]+str(i), window_size=200,
                                    x0=None, y0=None, color='g')



            self.tr_fig.add_graph(self.fig_names["waypoints"], window_size=len(waypoints),
                                    x0=self.wx, y0=self.wy,
                                    linestyle="-", marker="", color='g')

            # Add trajectory markers

            self.tr_fig.add_graph(self.fig_names["trajectory"], window_size=self.win_size,
                                    x0=[self.wx[0]] * self.win_size,
                                    y0=[self.wy[0]] * self.win_size,
                                    color=[1, 0.5, 0])


            # Add ego A marker
            self.tr_fig.add_graph(self.fig_names["ego_a_pos"], window_size=1,
                                     x0=[self.wx[0]], y0=[self.wy[0]],
                                     marker=11, color=[1, 0.5, 0],
                                     markertext="A", marker_text_offset=1)

            # Add ego B marker
            self.tr_fig.add_graph(self.fig_names["ego_b_pos"], window_size=1,
                                     x0=[self.wx[-1]], y0=[self.wy[-1]],
                                     marker=11, color=[1, 0.5, 0],
                                     markertext="B", marker_text_offset=1)

            # Add car marker
            self.tr_fig.add_graph(self.fig_names["ego"], window_size=1,
                                     marker="s", color='b', markertext="ego",
                                     marker_text_offset=1)

            # Add local path proposals
            for i in range(num_local_path):
                self.tr_fig.add_graph(self.fig_names["paths"] + str(i), window_size=200,
                                         x0=None, y0=None, color=[0.0, 0.0, 1.0])

            """
            # Add lead car information
            trajectory_fig.add_graph("leadcar", window_size=1,
                                     marker="s", color='g', markertext="Lead Car",
                                     marker_text_offset=1)
            # Add stop sign position
            trajectory_fig.add_graph("stopsign", window_size=1,
                                     x0=[stopsign_fences[0][0]], y0=[stopsign_fences[0][1]],
                                     marker="H", color="r",
                                     markertext="Stop Sign", marker_text_offset=1)
            # Add stop sign "stop line"
            trajectory_fig.add_graph("stopsign_fence", window_size=1,
                                     x0=[stopsign_fences[0][0], stopsign_fences[0][2]],
                                     y0=[stopsign_fences[0][1], stopsign_fences[0][3]],
                                     color="r")
            # Load parked car points
            parkedcar_box_pts_np = np.array(parkedcar_box_pts)
            trajectory_fig.add_graph("parkedcar_pts", window_size=parkedcar_box_pts_np.shape[0],
                             x0=parkedcar_box_pts_np[:,0], y0=parkedcar_box_pts_np[:,1],
                             linestyle="", marker="+", color='b')

            # Add lookahead path
            trajectory_fig.add_graph("selected_path",
                             window_size=INTERP_MAX_POINTS_PLOT,
                             x0=[start_x]*INTERP_MAX_POINTS_PLOT,
                             y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                             color=[1, 0.5, 0.0],
                             linewidth=3)
            
            # Add local path proposals
            for i in range(NUM_PATHS):
            trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                 x0=None, y0=None, color=[0.0, 0.0, 1.0])
            """

        if controls is True:
            self.ego_speed_fig = self.ct_plot.plot_new_dynamic_figure(title="Ego Speed (m/s)")
            self.ego_speed_fig.add_graph(self.fig_names["speed"], label="speed", window_size=self.win_size)
            self.ego_speed_fig.add_graph(self.fig_names["des_speed"], label="des_speed", window_size=self.win_size)

            self.throttle_fig = self.ct_plot.plot_new_dynamic_figure(title="Throttle Signal (0..1)")
            self.throttle_fig.add_graph(self.fig_names["throttle"], label="throttle", window_size=self.win_size)

            self.brake_fig = self.ct_plot.plot_new_dynamic_figure(title="Brake Signal (0..1)")
            self.brake_fig.add_graph(self.fig_names["brake"], label="brake", window_size=self.win_size)

            self.steer_fig = self.ct_plot.plot_new_dynamic_figure(title="Steer Signal (0..1)")
            self.steer_fig.add_graph(self.fig_names["steer"], label="steer", window_size=self.win_size)

    def refresh_plots(self):
        self.tr_plot.refresh()
        self.ct_plot.refresh()

    def get_step(self):
        return 1/self.win_size


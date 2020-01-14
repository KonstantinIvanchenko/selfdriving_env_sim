#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Author: Konstantin Ivanchenko
# Date: December 25, 2019

import numpy as np
import networkx as nx
import csv
import matplotlib.pyplot as plt
import math
import cubic_spline
import common as cn

class Route(object):
    """
    Implements global route building in a Carla environment.
    """
    def __init__(self, world_map, start_point):
        self._map_topology_decimation = 15.0  # generate decimated points for the global map; distance in [m]
        self.click_max = 2
        self.clicks = 0
        self.nav_points_map = []  # raw waypoints as user clicks
        if start_point is not None:
            self.nav_points_map.append((start_point.x, start_point.y))
            self.click_max -= 1
        self.wp_offset = 1        # offset for generating new waypoints between global waypoints

        self.map_all_waypoints = world_map.generate_waypoints(self._map_topology_decimation)
        self.map_topology = world_map.get_topology()
        #nx_graph_topology = nx.MultiDiGraph()
        #nx_graph_topology.add_edges_from(self.map_topology)
        #self.draw_topology()
        self.nx_graph_topology, self.node_id_map_topology = self.create_nx_repr()

        self.nx_path = None  # list of nx nodes from A-to-B
        self.nx_path_exist = True
        self.nx_path_built = False
        self.waypoint_a = cn.Waypoint(0, 0, 0)  # node A as waypoint location
        self.waypoint_b = cn.Waypoint(0, 0, 0)  # node B as waypoint location
        self.waypoints = []  # path as waypoint locations
        self.extended_wp = list()

        # self.draw_topology()
        self.draw_raw_points()

    def create_nx_repr(self):
        """
        Creates NX graph representation of Carla vertexes. It utilizes node
        and segment information that annotates junctions and links between
        them for the currently selected map.
        :return:
            nx_grap : NX directed graph object - nx graph built for
                the set of nodes and edges.
            node_id_map : {} - dictionary containing nodes. Each node is
             addressed by its unique ID information from the Carla
             environment used for hashing.
        """
        nx_graph = nx.DiGraph()#DiGraph()
        node_id_map = dict()

        for segment in self.map_topology:
            for vertex in segment:  # eventually loop over all nodes from map_topology
                if vertex.id not in node_id_map:
                    new_id = len(node_id_map)
                    # node_id_map[vertex] = new_id
                    node_id_map[vertex.id] = vertex
                    nx_graph.add_node(vertex.id, id=new_id)
                    # each node in graph is annotated with its unique label
                    # nx_graph.add_node(vertex, id=new_id)  # xyz=vertex.transform.location)


            # for each segment get entry and exit node IDs
            # n_entry_id = node_id_map[segment[0].id]
            # n_exit_id = node_id_map[segment[1].id]

            # nx_graph.add_edge(n_entry_id, n_exit_id)
            def distance(v_xyz_a, v_xyz_b): return math.sqrt(
                (v_xyz_a.x - v_xyz_b.x) ** 2 + (v_xyz_a.y - v_xyz_b.y) ** 2 + (v_xyz_a.z - v_xyz_b.z) ** 2)

            nx_graph.add_edge(segment[1].id, segment[0].id,
                            length=distance(segment[0].transform.location,
                                            segment[1].transform.location),
                            intersection=segment[0].is_junction)  # TODO: check later correctness of junction

        # return:
        # graph as nx DiGraph object
        # node ID map as dictionary 'node_XYZ:ID'
        return nx_graph, node_id_map

    def draw_topology(self):
        """
        Draws a NX representation of graph vertexes.
        Not used
        :param: None
        :return: None
        """
        nx.draw(self.nx_graph_topology, pos=nx.random_layout(self.nx_graph_topology), nodecolor='b', edge_color='r')
        plt.show()

    def draw_raw_points(self):
        """
        Visualizes the set of map points that implement nodes.
        User uses the drawn topology for AB point selection in
        a convenient way.
        :param: None
        :return: None
        """
        # wp_xyz_list = list(self.nx_graph_topology.nodes(data='xyz')
        xp_xyz_list = list()
        for vertex_id in self.nx_graph_topology.nodes():
            xp_xyz_list.append(self.node_id_map_topology[vertex_id].transform.location.x)
            xp_xyz_list.append(self.node_id_map_topology[vertex_id].transform.location.y)

        # all_elements1 = xp_xyz_list[::2]
        # p1 = xp_xyz_list[::2]
        # p2 = xp_xyz_list[1::2]

        segment_list = list()

        for edge in self.nx_graph_topology.edges():
            segment_list.append(edge)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(xp_xyz_list[::2], xp_xyz_list[1::2], s=5)
        ax.set_title('Global navi waypoints')

        for i in range(len(segment_list)):
            ax.plot([self.node_id_map_topology[segment_list[i][0]].transform.location.x,
                     self.node_id_map_topology[segment_list[i][1]].transform.location.x],
                    [self.node_id_map_topology[segment_list[i][0]].transform.location.y,
                     self.node_id_map_topology[segment_list[i][1]].transform.location.y],
                    'k-',
                    lw=1)

        def onclick(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            if self.clicks < self.click_max:
                self.clicks += 1
                self.nav_points_map.append((event.xdata, event.ydata))

            # return event.xdata, event.ydata

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()
        # wp_xyz_list = list(self.nx_graph_topology.nodes())

    def find_inner_topology_rsl(self, p1, p2):
        """
        Find road inner waypoints that don't belong to the nodes WPs;
        This function operates only waypoint information - NOTE: doesn't work in the current implementation
        as self.map_topology info is required for matching.
        :param:
            p1 : carla waypoint object - first wp
            p2 : carla waypoint object - second wp
        :return:
            inner_p_list : list - inner waypoint object list devoid the p1 and p2
        """
        inner_p_list = []

        def is_same_rsl(pa, pb):
            if pa.road_id == pb.road_id and pa.section_id == pb.section_id and pa.lane_id == pb.lane_id:
                return True
            else: return False

        if not is_same_rsl(p1, p2): return inner_p_list

        for wp in self.map_all_waypoints:
            if is_same_rsl(p1, wp):
                inner_p_list.append(wp)

        return inner_p_list

    def find_inner_topology_rsl_bysegm(self, p1, p2):
        """
        Find road inner waypoints that don't belong to the nodes WPs;
        This function operates the initial information about the segment to match the inner WPs
        :param:
            p1 : carla waypoint object - first wp
            p2 : carla waypoint object - second wp
        :return:
            inner_p_list : list - inner waypoint object list devoid the p1 and p2
        """
        inner_p_list = []

        dist = np.inf
        p_found = False
        p1_order = False
        p2_order = False
        d1 = np.inf

        def distance(v_a, v_b):
            return math.sqrt(
                (v_a.transform.location.x - v_b.transform.location.x) ** 2 +
                (v_a.transform.location.y - v_b.transform.location.y) ** 2)  # z is not taken into account

        def is_same_rsl(pa, pb):
            if pa.road_id == pb.road_id and pa.section_id == pb.section_id and pa.lane_id == pb.lane_id:
                return True
            else:
                return False

        for segment in self.map_topology:
            if segment[0].id == p1.id and segment[1].id == p2.id:
                if not is_same_rsl(segment[0], p1):
                    continue

                # always take rsl of the waypoint idx 0 of the segment
                for wp in self.map_all_waypoints:
                    if is_same_rsl(p1, wp):

                        # skip adding wp same as p1 or p2, as the node items will be treated outside.
                        #if wp.id == p1.id or wp.id == p2.id:
                        #    continue

                        inner_p_list.append(wp)
            ##### so far only this part runs
            elif segment[1].id == p1.id and segment[0].id == p2.id:
                if not is_same_rsl(segment[0], p2):
                    return inner_p_list

                #dist = np.inf

                # always take rsl of the waypoint idx 0 of the segment
                for wp in self.map_all_waypoints:
                    if is_same_rsl(p2, wp):
                        # p2 is always end of segment where to drive
                        # p1 is always a beginning of segment to drive from
                        # skip adding same wp as initial wp if found (unlikely)
                        if wp.id == p2.id:
                            continue
                        #inner_p_list.append(wp)
                        #if p_found is False:
                        #    d1 = distance(wp, p1)
                        #
                        inner_p_list.append(wp)
                        """
                        
                        if len(inner_p_list) == 1:
                            inner_p_list.append(wp)
                            dist = distance(wp, p2)
                        elif len(inner_p_list) >= 2:
                            index = len(inner_p_list)
                            for i in range(len(inner_p_list)):
                                d = dist(inner_p_list[i], wp)
                                if d < dist:
                                    dist = d
                                    index = i
                        """
                #inner_p_list.insert(0, p2)  # always append last initial point first
                # inner_p_list contains all the points except p2 itself
                if inner_p_list: # if the list is not empty check the order of points
                    d1 = distance(inner_p_list[0], p2)
                    d2 = distance(inner_p_list[len(inner_p_list)-1], p2)
                    if d1 < d2:
                        inner_p_list.reverse()
                        inner_p_list.append(p2)
                    else:
                        inner_p_list.append(p2)
                else:  # if the list is empty just add the p2
                    inner_p_list.append(p2)
        return inner_p_list

    def find_inner_topology(self, p1, p2):
        """
        Find road inner waypoints that don't belong to the nodes WPs;
        This function operates only waypoint information - NOTE: this method will not work as the map_all_waypoints
        doesn't store segmented lines (e.g. it is not sorted in accordance with each segment waypoints).
        :param:
            p1 : carla waypoint object - first wp
            p2 : carla waypoint object - second wp
        :return:
            inner_p_list : list - inner waypoint object list devoid the p1 and p2
        """
        inner_p_list = []
        temp_p_list = []
        len_inner = np.inf
        len_t = 0

        for idx in range(len(self.map_all_waypoints)):
            #temp_p_list.clear()
            #len_t = 0
            #len_inner = np.inf

            if self.map_all_waypoints[idx].id == p1.id:
                for idx_t in range(idx+1, len(self.map_all_waypoints)):
                    temp_p_list.append(self.map_all_waypoints[idx_t])
                    len_t += 1
                    if self.map_all_waypoints[idx_t].id == p2.id:
                        if len_inner >= len_t:
                            inner_p_list = temp_p_list.copy()
                            len_inner = len_t
                            len_t = 0
                            temp_p_list.clear()
                            break

                    if idx_t == len(self.map_all_waypoints) - 1:
                        len_t = 0
                        temp_p_list.clear()

            if self.map_all_waypoints[idx].id == p2.id:
                for idx_t in range(idx+1, len(self.map_all_waypoints)):
                    temp_p_list.append(self.map_all_waypoints[idx_t])
                    len_t += 1
                    if self.map_all_waypoints[idx_t].id == p1.id:
                        if len_inner >= len_t:
                            inner_p_list = temp_p_list.copy()
                            len_inner = len_t
                            len_t = 0
                            temp_p_list.clear()
                            break

                    if idx_t == len(self.map_all_waypoints) - 1:
                        len_t = 0
                        temp_p_list.clear()

        return inner_p_list

    def build_route(self):
        """
        Builds top level route from user-chosen A and B points. Uses three steps:
        -Astar on the originally annotated waypoints.
        -Coarse linear filling of the route sections with more WPs
        -Fine 3rd order spline fitting
        Top level max speeds are assigned for each segment.
        :param: None
        :return: None
        """
        A = None  # location
        B = None  # location
        dist_a = np.inf
        dist_b = np.inf

        nx_node_a = None  # nx vertex
        nx_node_b = None  # nx vertex

        interp_granul = 0.5 # 0.4 # 4 # interpolation granularity
        spline_accur = lambda dist: int(dist / interp_granul)
        cum_dist = 0  # cumulative distance between global waypoints. Used to estimate interpolation granularity
        wp_mindist_thr = 4 # 8  # min threshold above which more points will be linearly inserted

        c_theta = 0  # road yaw angle
        d_theta = 0  # road yaw angle delta segment-wise
        d_t_thr_j = 0.26  # Road yaw delta threshold 15°
        d_t_thr = 0.08  # Road yaw delta threshold 5°

        mxspd_c = 8#11  # max common city speed 40 km/h # NOTE: don't set to greater speed
        mxspd_c_t = 6#8  # max common city speed 29 km/h for turns
        mxspd_c_s_j = 7#10 # max common straight junction speed 36 km/h
        mxspd_c_t_j = 5 # max common turn junction speed 18 km/h

        # get closest points for A and B
        for vertex_id in self.nx_graph_topology.nodes():

            def distance(wp_xy, v_xyz): return math.sqrt(
                (wp_xy[0] - v_xyz.x) ** 2 + (wp_xy[1] - v_xyz.y) ** 2)  # z is not taken into account

            # use reverse order for right lane driving
            da = distance(self.nav_points_map[1], self.node_id_map_topology[vertex_id].transform.location)
            db = distance(self.nav_points_map[0], self.node_id_map_topology[vertex_id].transform.location)

            if da < dist_a:
                dist_a = da
                A = self.node_id_map_topology[vertex_id].transform.location
                nx_node_a = vertex_id

            if db < dist_b:
                dist_b = db
                B = self.node_id_map_topology[vertex_id].transform.location
                nx_node_b = vertex_id

        # store vertexes as simple waypoint objects
        self.waypoint_a.x = A.x
        self.waypoint_a.y = A.y
        self.waypoint_b.x = B.x
        self.waypoint_b.y = B.y

        try:
            self.nx_path = nx.algorithms.astar_path(self.nx_graph_topology, nx_node_a, nx_node_b, weight='length')

            #self.nx_path = nx.algorithms.shortest_path(self.nx_graph_topology, nx_node_a, nx_node_b, weight='length',
            #                                          method='dijkstra')

            self.nx_path_exist = True
            self.nx_path_built = True
        except nx.NetworkXNoPath:
            print('No path created from NX node graph')
            self.nx_path_exist = False

        # for wp in self.nx_path:
        #    self.waypoints.append(Waypoint(wp.transform.location.x, wp.transform.location.y))

        for i in range(len(self.nx_path)):

            def distance(v_xyz_a, v_xyz_b): return math.sqrt(
                (v_xyz_a.x - v_xyz_b.x) ** 2 + (v_xyz_a.y - v_xyz_b.y) ** 2)  # z is not taken into accoint

            def theta(v_xyz_a, v_xyz_b): return np.arctan2(-v_xyz_a.y + v_xyz_b.y, -v_xyz_a.x + v_xyz_b.x)

            last_segment = False

            if i == len(self.nx_path)-2:
                # change speed settings for last segment
                last_segment = True

            if i == len(self.nx_path)-1:
                # append the last WP with the 0 enclosing speed and stop
                plast = self.node_id_map_topology[self.nx_path[i]].transform.location
                self.waypoints.append(cn.Waypoint(plast.x, plast.y, 0))
                break

            # TODO: reminder - NOT USED
            # inner_points = self.find_inner_topology(self.node_id_map_topology[self.nx_path[i]],
            #                                         self.node_id_map_topology[self.nx_path[i+1]])

            #inner_points = self.find_inner_topology_rsl(self.node_id_map_topology[self.nx_path[i]],
            #                                            self.node_id_map_topology[self.nx_path[i+1]])

            inner_points = self.find_inner_topology_rsl_bysegm(self.node_id_map_topology[self.nx_path[i]],
                                                               self.node_id_map_topology[self.nx_path[i+1]])
            #ptest = self.node_id_map_topology[self.nx_path[i]]
            #wp = cn.Waypoint(ptest.transform.location.x, ptest.transform.location.y, 0)
            #self.waypoints.append(wp)

            if len(inner_points) == 1:
                p = self.node_id_map_topology[self.nx_path[i]].transform.location
                p_next = self.node_id_map_topology[self.nx_path[i+1]].transform.location
                dist_to_next = distance(p, p_next)
                cum_dist += dist_to_next
                n_theta = theta(p, p_next)
                d_theta = np.abs(n_theta - c_theta)
                plocal_speed = 0

                if self.node_id_map_topology[self.nx_path[i]].is_junction:
                    # if junction
                    if d_theta > d_t_thr_j:  # current point has significant yaw delta
                        plocal_speed = mxspd_c_t_j
                    else:  # current point has insignificant yaw delta
                        plocal_speed = mxspd_c_s_j
                else:
                    # no junction
                    if d_theta > d_t_thr:  # insignificant expected curvature
                        plocal_speed = mxspd_c_t
                    elif d_theta > d_t_thr_j:  # current point of road has significant curvature
                        plocal_speed = mxspd_c_t_j
                    else:  # current point is simple straight
                        plocal_speed = mxspd_c

                wp = cn.Waypoint(p.x, p.y, plocal_speed)  # current point is simple straight but reduced speed
                self.waypoints.append(wp)
                print(p.x, " ", p.y, " ", plocal_speed)
            else:
                for iwp_idx in range(len(inner_points)-1):

                    p = inner_points[iwp_idx].transform.location
                    p_next = inner_points[iwp_idx+1].transform.location
                    dist_to_next = distance(p, p_next)
                    cum_dist += dist_to_next

                    n_theta = theta(p, p_next)
                    d_theta = np.abs(n_theta - c_theta)

                    plocal_speed = 0

                    if inner_points[iwp_idx].is_junction:
                        # if junction
                        if d_theta > d_t_thr_j: # current point has significant yaw delta
                            plocal_speed = mxspd_c_t_j
                        else: # current point has insignificant yaw delta
                            plocal_speed = mxspd_c_s_j
                    else:
                        # no junction
                        if d_theta > d_t_thr:  # insignificant expected curvature
                            plocal_speed = mxspd_c_t
                        elif d_theta > d_t_thr_j: # current point of road has significant curvature
                            plocal_speed = mxspd_c_t_j
                        else: # current point is simple straight
                            plocal_speed = mxspd_c

                    #wp = cn.Waypoint(p.x, p.y, plocal_speed)
                    #self.waypoints.insert(0, wp)
                    #self.waypoints.append(wp)
                    #print(p.x, " ", p.y, " ", plocal_speed)
                    # check if requires to insert more points linearly between two global path neighbour points
                    # if the distance between two WPs is greater than the min threshold, add more points in between

                    if dist_to_next > wp_mindist_thr:
                        d_list = np.linspace(0, dist_to_next, num=int(dist_to_next/wp_mindist_thr), endpoint=False)
                        n_segm = len(d_list)
                        for j in range(0, n_segm):
                            if last_segment is False:
                                wp = cn.Waypoint(p.x + np.cos(n_theta) * d_list[j],
                                              p.y + np.sin(n_theta) * d_list[j],
                                              plocal_speed)  # current point is simple straight
                            else:
                                t = j/n_segm
                                div = 0.15*t/(0.15-t+1)+1  # TODO: too fast converged
                                wp = cn.Waypoint(p.x + np.cos(n_theta) * d_list[j],
                                              p.y + np.sin(n_theta) * d_list[j],
                                              plocal_speed/div)  # current point is simple straight but reduced speed
           
                            self.waypoints.append(wp)
                            print(wp.x, " ", wp.y, " ", wp.s)

                    c_theta = n_theta



        # apply cubic spline interpolation with the wp-granularity based on the cumulative route distance
        cubic_spline_int = cubic_spline.CubicSplineParam(self.waypoints, spline_accur(cum_dist))
        X,Y,S = cubic_spline_int.interpolate_wp()

        for i in range(len(X)):
            # append from left as the A and B points are submitted in a reverse order to the NX graph
            self.extended_wp.insert(0, cn.Waypoint(X[i], Y[i], S[i]))

        """
        splines = cubic_spline.Cubicspline(self.waypoints)
        self.extended_wp.append(self.waypoints[0])

        for i in range(0, len(self.nx_path)-1, 1):
            k = 0
            while (self.waypoints[i+1].x - self.waypoints[i].x - k) > 1:
                k += 5
                nwx = self.waypoints[i].x + k
                nwy = splines.get_y(nwx, i)
                self.extended_wp.append(cn.Waypoint(nwx, nwy))

            self.extended_wp.append(self.waypoints[i])
        """
        #self.extended_wp = list(reversed(self.extended_wp))

        return self.nx_path_built

    def draw_extended_path(self):
        """
        Visualize the road network with path formed of the initially annotated waypoints as well as
         an extended fine path.
        :param: None
        :return: None
        """
        xp_xyz_list = list()
        for vertex_id in self.nx_graph_topology.nodes():
            xp_xyz_list.append(self.node_id_map_topology[vertex_id].transform.location.x)
            xp_xyz_list.append(self.node_id_map_topology[vertex_id].transform.location.y)

        segment_list = list()

        for edge in self.nx_graph_topology.edges():
            segment_list.append(edge)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(xp_xyz_list[::2], xp_xyz_list[1::2], s=5)
        ax.set_title('Global navi waypoints with route')

        for i in range(len(segment_list)):
            ax.plot([self.node_id_map_topology[segment_list[i][0]].transform.location.x,
                     self.node_id_map_topology[segment_list[i][1]].transform.location.x],
                    [self.node_id_map_topology[segment_list[i][0]].transform.location.y,
                     self.node_id_map_topology[segment_list[i][1]].transform.location.y],
                    'k-',
                    lw=1)

        for i in range(len(self.waypoints)-1):
            ax.plot([self.waypoints[i].x, self.waypoints[i+1].x],
                    [self.waypoints[i].y, self.waypoints[i+1].y],
                    'k-',
                    color='red',
                    lw=2)

        for i in range(len(self.extended_wp)-1):
            ax.plot([self.extended_wp[i].x,
                     self.extended_wp[i+1].x],
                    [self.extended_wp[i].y,
                     self.extended_wp[i+1].y],
                    'k-',
                    color='green',
                    lw=1)
            if i%3 == 0:
                ax.text(self.extended_wp[i].x, self.extended_wp[i].y, str(int(self.extended_wp[i].s)), rotation=45, fontsize=8)

        plt.show()

    def draw_path(self):
        """
        Visualize the road network with path formed of the initially annotated waypoints.
        :param: None
        :return: None
        """
        xp_xyz_list = list()
        for vertex_id in self.nx_graph_topology.nodes():
            xp_xyz_list.append(self.node_id_map_topology[vertex_id].transform.location.x)
            xp_xyz_list.append(self.node_id_map_topology[vertex_id].transform.location.y)

        segment_list = list()

        for edge in self.nx_graph_topology.edges():
            segment_list.append(edge)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(xp_xyz_list[::2], xp_xyz_list[1::2], s=5)
        ax.set_title('Global navi waypoints with route')

        for i in range(len(segment_list)):
            ax.plot([self.node_id_map_topology[segment_list[i][0]].transform.location.x,
                     self.node_id_map_topology[segment_list[i][1]].transform.location.x],
                    [self.node_id_map_topology[segment_list[i][0]].transform.location.y,
                     self.node_id_map_topology[segment_list[i][1]].transform.location.y],
                    'k-',
                    lw=1)

        for i in range(len(self.waypoints)-1):
            ax.plot([self.waypoints[i].x, self.waypoints[i+1].x],
                    [self.waypoints[i].y, self.waypoints[i+1].y],
                    'k-',
                    color='red',
                    lw=2)

        plt.show()
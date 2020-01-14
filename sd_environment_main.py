"""
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Author: Konstantin Ivanchenko
# Date: December 26, 2019

This works attempts to build a complete top-down controller to solve a partial SD task for navigating a car
in a simulated environment based on common practises in autonomous vehicles.

Utilizes:
    -CARLA simulator
    -some of the excellent methods proposed in Coursera Self-Driving Cars specialization.
    -some of the standard source code from CARLA demo.
    -uses only classic algorithms without AI technics.
"""

"""
Still required minimum (TODO):
    --Add complete ego state estimation model.
    --Add obstacle grid.
    --Optimizations required.
"""


from __future__ import print_function
from __future__ import division

# System level imports
import glob
import sys
import os
import argparse
import random
import pygame
import queue
import numpy as np

import global_route as gr
import controller2D as c2d
#import configparser
from motion_planner import behavioral_planner as bpl
from motion_planner import local_planner as lpl
#import dynamic_obstacles
from perception import boundary_finder as bfd
import plotting
import common as cn

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import carla

"""
Actor global parameters
Note: strings
"""
CAM_ON_VEHICLE_FOV = '90.0'     #DEFAULT - '90.0'; cameras field of view
LIDAR_CHANNELS = '32'           #DEFAULT - '32'; number of lasers
LIDAR_RANGE = '1000'            #DEFAULT - '1000'; FIXME: for carla version >0.9.6 range is in [m]
LIDAR_POINTS_PER_SEC = '5600'   #DEFAULT - '56000'; All lasers per second
LIDAR_UPPER_LIMIT_DEG = '10.0'  #DEFAULT - '10.0'; degrees
LIDAR_LOWER_LIMIT_DEG = '-30.0' #DEFAULT - '-30.0'; degrees
LIDAR_SENSOR_TICKS = '0.0'      #DEFAULT - '0.0'; seconds, ticks between sensor captures

EGO_SPAWN_INDEX = 5000          #Arbitrary:
                                #Good spawn locations: Town03 - 1000; Town03 - 5000

"""
World global parameters
"""
CLIENT_TIMEOUT = 10.0       # DEFAULT - 10.0;  of the client instance on start
NUM_VEHICLES = 2            # DEFAULT - 2; TODO: dynamic obstacles
NUM_PEDESTRIANS = 0         # DEFAULT - 0; TODO: dynamic obstacles
SEED_VEHICLES = 1.0         # DEFAULT - 1.0; TODO: dynamic obstacles
SEED_PEDESTRIANS = 1.0      # DEFAULT - 1.0; TODO: dynamic obstacles
GEN_POINTS_SPAWN_DIST = 0.1     # 0.1 - DEFAULT
SIMWEATHER = 0                  # 0 - DEFAULT

"""
Controller update rate factor.
"""
UPDATE_CONTROL_RATE = 4         # 4 - DEFAULT; faster is lower; int; min = 1

"""
Plotting parameters
"""
ENABLE_PLOT_TRACK = True
ENABLE_PLOT_CONTROL = False
ENABLE_PLOT_LOCAL_PATH = True
ENABLE_PLOTTING = True

class World(object):
    """
    Representation class of the simulation environment.
    :param:
        carla_world : carla.World object
            References initialized carla.world
        carla_client : carla.Client object
            References initialized carla.Client
        carla_vehicle_ctl : carla.VehicleControl object
            References initialized carla.VehicleControl
        carla_timestamp : carla.Timestamp object
            References initialized carla.Timestamp
        args : {string} - argparser arguments
    :return: None
    """
    def __init__(self, carla_world, carla_client, carla_vehicle_ctl, carla_timestamp, args):
        self.world = carla_world
        self.client = carla_client
        self.vcl_ctl = carla_vehicle_ctl
        self.timestamp = carla_timestamp

        self.map = carla_world.get_map()
        self.ego_vehicle = None
        self.ego_cam = None
        self.ego_camdep = None
        self.ego_camsemseg = None
        self.update_world_settings(args)
        self.ego_spawn_point = self.init_ego_objects(args.timeperiod)


    def init_ego_objects(self, frame_period):
        """
        Initializes ego related carla object blueprints and spawns them.
        To be extended with further carla objects if necessary
        :param:
            frame_period : float - simulation frame rate in seconds; used for artificial adjustment of the lidar rotation.
            Usually adjusts frequency for having a full spin at one simulation step.
        :return:
            spawn_point : carla.Transform object - ego spawn point as carla.Transform object
        """
        # was 1000 on map Town3
        spawn_point_idx = 5000  # use here an arbitrary index

        ego_blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.ford.mustang'))
        ego_blueprint.set_attribute('role_name', 'ego_vehicle')

        all_waypoints = self.map.generate_waypoints(GEN_POINTS_SPAWN_DIST)
        spawn_point_move = all_waypoints[spawn_point_idx].transform
        # First spawn at predefined location and then move the ego to necessary location.
        # Helps to avoid glitches on route start

        # display camera blueprint
        ego_cam_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        ego_cam_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')

        # ego camera depth blueprint
        ego_camdep_transform = carla.Transform(carla.Location(x=1.5, z=1.5), carla.Rotation(pitch=0))
        ego_camdep_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
        ego_camdep_blueprint.set_attribute('fov', CAM_ON_VEHICLE_FOV)  # field of view

        # ego camera semantic segmentation blueprint
        ego_camsemseg_transform = carla.Transform(carla.Location(x=1.5, z=1.5), carla.Rotation(pitch=0))
        ego_camsemseg_blueprint = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        ego_camsemseg_blueprint.set_attribute('fov', CAM_ON_VEHICLE_FOV)  # field of view

        # lidar sensor blueprint
        ego_lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        ego_lidar_blueprint.set_attribute('channels', LIDAR_CHANNELS)
        ego_lidar_blueprint.set_attribute('range', LIDAR_RANGE)  # Range in centimeters
        ego_lidar_blueprint.set_attribute('points_per_second', LIDAR_POINTS_PER_SEC)
        # FIXME: may require adjustment
        ego_lidar_blueprint.set_attribute('rotation_frequency', str(2*np.pi/frame_period))
        ego_lidar_blueprint.set_attribute('upper_fov', LIDAR_UPPER_LIMIT_DEG)  # degrees of the upper most laser
        ego_lidar_blueprint.set_attribute('lower_fov', LIDAR_LOWER_LIMIT_DEG)  # degrees of the lower most laser
        # FIXME: may require adjustment
        ego_lidar_blueprint.set_attribute('sensor_tick', LIDAR_SENSOR_TICKS)  # seconds between sensor captures
        ego_lidar_transform = carla.Transform(carla.Location(x=0.5, z=2.5), carla.Rotation(pitch=0))

        if self.ego_vehicle is None:
            spawn_point_idx = 0  # default, arbitrary # was 3
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[spawn_point_idx]
            # random.choice(spawn_points) if spawn_points else carla.Transform()
            self.ego_vehicle = self.world.try_spawn_actor(ego_blueprint, spawn_point)
            self.ego_cam = self.world.try_spawn_actor(ego_cam_blueprint, ego_cam_transform,
                                                      attach_to=self.ego_vehicle)
            self.ego_camdep = self.world.try_spawn_actor(ego_camdep_blueprint, ego_camdep_transform,
                                                         attach_to=self.ego_vehicle)
            self.ego_camsemseg = self.world.try_spawn_actor(ego_camsemseg_blueprint, ego_camsemseg_transform,
                                                            attach_to=self.ego_vehicle)
            self.ego_lidar = self.world.try_spawn_actor(ego_lidar_blueprint, ego_lidar_transform,
                                                        attach_to=self.ego_vehicle)

            self.ego_vehicle.set_transform(spawn_point_move)

            # wait_for_tick is required after spawning actors. Method proposed at
            # https://github.com/carla-simulator/carla/issues/1424
            # self.world.wait_for_tick()
            # time.sleep(2)

        return spawn_point_move

    def update_world_settings(self, args):
        """
        Updates carla.world.settings with user defined config.
        :param:
            args : {} - argparser arguments
        :return:
            settings : carla.world.settings object
        """

        settings = self.world.get_settings()

        if NUM_VEHICLES > 0 or NUM_PEDESTRIANS > 0:
            get_non_player_agents_info = True
        else:
            get_non_player_agents_info = False

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.timeperiod  # True ## FIXME: set here a time variable
        settings.SendNonPlayerAgentsInfo = get_non_player_agents_info
        settings.NumberOfVehicles = NUM_VEHICLES
        settings.NumberOfPedestrians = NUM_PEDESTRIANS
        settings.SeedVehicles = SEED_VEHICLES
        settings.SeedPedestrians = SEED_PEDESTRIANS
        settings.WeatherId = SIMWEATHER
        settings.QualityLevel = args.quality

        self.world.apply_settings(settings)

        return settings

    def get_ego_state(self):
        """
        Get ego state as vector [x,y,z,yaw,s,a] + timestamp
        :param: None
        :return: Egostate object
        """
        t, v, a, ts = self.read_ego_gt()
        x = t.location.x
        y = t.location.y
        z = t.location.z
        yaw = t.rotation.yaw*np.pi/180.0  # t.rotation.yaw is in degrees
        # alpha_v = np.arctan2(v.y, v.x)
        # alpha_a = np.arctan2(a.y, a.x)

        #TODO:simplified
        vel = np.sqrt(v.x**2 + v.y**2)
        acc = np.sqrt(a.x**2 + a.y**2)

        # vel = np.sqrt(v.x**2 + v.y**2)*np.cos(yaw - alpha_v) # TODO: check this
        # acc = np.sqrt(a.x**2 + a.y**2)*np.cos(yaw - alpha_a) # TODO: check this
        # vel = v.x*np.cos(yaw)+v.y*np.sin(yaw)  # longitudinal velocity
        # acc = a.x * np.cos(yaw) + a.y * np.sin(yaw)  # longitudinal acceleration

        return cn.Egostate(x, y, z, yaw, vel, acc, ts/1000.0)

    def read_ego_gt(self):
        """
        Receive world's ground truth state of ego vehicle with current timestamp.
        :param: None
        :return: carla.Transform object, (), (), float
        """
        sn = self.world.get_snapshot()
        return self.ego_vehicle.get_transform(), \
               self.ego_vehicle.get_velocity(), \
               self.ego_vehicle.get_acceleration(), \
               sn.timestamp.platform_timestamp  / 1000.0  # platform_timestamp is in seconds

    def write_ego_control(self, steer=0.0, brake=0.5, throttle=0.0, hand_brake=False, reverse=False):
        """
        Send control command to the world while running in synchronous mode.
        :param:
            steer : float - steer command [-1.0, 1.0]
            brake : float - brake command [0.0, 1.0]
            throttle : float - throttle command [0.0, 1.0]
            hand_brake : bool - hand brake command yes/no
            reverse : bool - reverse command yes/np
        :return: None
        """
        self.vcl_ctl.steer = np.fmax(np.fmin(steer, 1.0), -1.0)
        self.vcl_ctl.brake = np.fmax(np.fmin(brake, 1.0), 0.0)
        self.vcl_ctl.throttle = np.fmax(np.fmin(throttle, 1.0), 0.0)
        self.vcl_ctl.hand_brake = hand_brake
        self.vcl_ctl.reverse = reverse
        self.ego_vehicle.apply_control(self.vcl_ctl)

    def get_average_timedelta(self, frames=10):
        """
        Get average time delta between frames.
        Note: actual time delta is larger during full control stack simulation. The returned is only used
        for adaptation of gain parameters of the low level controller.
        :param:
            frames : int - amount of frames to run averaging
        :return:
            timedelta : float - time delta in milliseconds
        """
        timedelta = 0.0

        if frames <= 0:
            frames = 10
        for i in range(frames):
            sn = self.world.get_snapshot()
            timedelta += sn.timestamp.delta_seconds / 2.0#2000.0
            self.write_ego_control()
        return timedelta

    def get_timedelta(self):
        """
        Get timedelta elapsed since previous frame
        :param: None
        :return:
            timedelta : float - time delta in milliseconds
        """
        sn = self.world.get_snapshot()
        return sn.timestamp.delta_seconds #/ 1000.0


class SensorCapture:
    """
    Capture class for attached sensors
    :param:
    cam_sensors : list of sensor.camera objects - spawned camera sensor objects to be queued
            Example: [world.ego_cam, world.ego_camdep, world.ego_camsemseg]
    lidar_sensor : sensor.lidar object - spawned lidar sensor object to be queued
            Example: world.ego_lidar
    :return: None
    """
    def __init__(self, cam_sensors=None, lidar_sensor=None):
        pygame.init()
        self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        #weak_self = weakref.ref(self)

        self.data_queues = []

        if cam_sensors is not None:
            for item in cam_sensors:
                #self.data_queues.append(queue.Queue())
                self.register_sensor_listen_method(item.listen)

        if lidar_sensor is not None:
            #self.data_queues.append(queue.Queue())
            self.register_sensor_listen_method(lidar_sensor.listen)

        #cam_sensors.listen(lambda image: self.on_cam_retrieve(weak_self, image))
        #self.data_available = False

    #@staticmethod
    #def on_event_insert_data(image):
        #new_queue = queue.Queue()

    def register_sensor_listen_method(self, listen):
        """
        Registers a callback for sensor listen method. Will be call upon each sensor data capture.
        Used to queue the incoming data of each sensor.
        :param
            listen: reference to method - callback to be registered
        :return: None
        """
        new_queue = queue.Queue()
        listen(new_queue.put)
        self.data_queues.append(new_queue)

    def capture_periodic(self):
        """
        Retrieves all sensor data. Called periodically from the mina loop.
        :param: None
        :return:
            data : list - list of carla.sensor.__ data objects
        """
        data = [self.display_data_retrieve(q) for q in self.data_queues]
        return data

    def display_data_retrieve(self, data_queue):
        """
        Retrieves sensor data object. Called periodically from the mina loop.
        :param:
            data_queue: Queue object - to be retrieved from.
        :return:
            data_sample : carla.Sensor.__ - sensor object.
        """
        while True:
            data_sample = data_queue.get()
            return data_sample

    def draw_image(self, image, blend=False):
        """
        Visualizes an image from any camera of the scene. Used for visualization of ego's behavior.
        :param
            image: carla.Image object - image to be drawn
            blend: boolean - if blend then alpha channel may be used.
        :return: None
        """
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        self.display.blit(image_surface, (0, 0))
"""
class DisplaySim:
    def __init__(self, cam_sensor):
        pygame.init()
        self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        weak_self = weakref.ref(self)
        cam_sensor.listen(lambda image: self.on_cam_retrieve(weak_self, image))
        #self.data_available = False
        self.data_queue = queue.Queue()

    #TODO: implement a data queue. Register QUEUE out as a callback function for camera.listen

    def cam_data_captured(self):
        self.data_available = False

    @staticmethod
    def on_cam_retrieve(weak_self, data):
        #data_available = True
        self = weak_self()
        draw_image(self.display, data)
        pygame.display.flip()

    #def cam_retrieve_on_tick(self, cam_sensor, timeout, frame_id):
    #    data = cam_sensor.listen()#timeout=timeout)
    #    if data.frame == frame_id:
    #        return data
"""


def exec_autopilot_nav(args):
    """
    Main thread executing ego-car navigation along the user-selected route
    :param args: {} - argparser arguments
    :return: None
    """
    ca_client = carla.Client(args.host, args.port)
    ca_client.set_timeout(CLIENT_TIMEOUT)
    ca_world = ca_client.load_world('Town03')
    # Carla vehicle control instance
    ca_vcl_control = carla.VehicleControl(manual_gear_shift=False)
    # Carla timestamp instance
    ca_timestamp = carla.Timestamp()

    world = World(ca_world, ca_client, ca_vcl_control, ca_timestamp, args)
    route = gr.Route(world.map, world.ego_spawn_point.location)
    # FIXME: uncomment for lidar sensing
    #sensor_capture = SensorCapture(cam_sensors=[world.ego_cam], lidar_sensor=world.ego_lidar)
    sensor_capture = SensorCapture(cam_sensors=[world.ego_cam, world.ego_camdep, world.ego_camsemseg])

    # ego control command send as dummy. Make sure that control is initialized on server side
    world.write_ego_control()
    ego_st_vect = world.get_ego_state()
    ts_prev = ego_st_vect.ts  # update previous timestemp

    ## perception: boundary finder
    bof = bfd.MarkedGrid()
    # used only for visualization of semantically segmented image
    bof.lf.set_palette(carla.ColorConverter.CityScapesPalette)

    ## control: behavioral planner
    bp = None

    ## control: velocity planner
    vp = None

    ## control: low_level controller
    ll_controller = None

    ## debug: motion plotter
    mt_plotter = None

    while True:
        frame_id = world.world.tick() # Initial tick is required here.
        # amount of items returned below shall correspond to the amount of sensors attached
        # the first one is the camera frame to display
        # FIXME: uncomment for lidar sensing
        # frame, lidar_data = display_sim.display_periodic()
        # print("n points: ", len(lidar_data))

        # get here set of frame: standard camera visual frame, the depth frame and semantically segmented frame
        frame, frame_dep, frame_sems = sensor_capture.capture_periodic()
        sensor_capture.draw_image(frame)
        pygame.display.flip()

        # update perception
        bof.find_lines(frame_dep, frame_sems)

        # perform check for the availability of route points A and B
        # if path is not built yet, then proceed
        if len(route.nav_points_map) == 2 and route.nx_path_built is False:
            # build route
            route.build_route()
            # if path found
            if route.nx_path_exist:
                # route.draw_path()
                # visualize path augmented with inner points
                route.draw_extended_path()
                # initialize local planner
                lp = lpl.LocalPlanner()
                # initialize behavioral planner
                bp = bpl.BehaviouralPlanner()
                # initialize controller with extended (augmented) set of waypoints
                ll_controller = c2d.Controller2D(route.extended_wp)
                # TODO: measure frame/server period
                ave_frame_rate_server = world.get_average_timedelta()
                print("Average simulation frame rate: {} ms ".format(ave_frame_rate_server))

                ## Motion Plotter
                if ENABLE_PLOTTING:
                    mt_plotter = plotting.MotionPlotter(route.extended_wp, lp.get_num_path())
                    mt_plotter.refresh_plots()

                ## Simulation step counter
                simulation_step = 0
        # if path is built
        # execute the main control loop
        if route.nx_path_built is True:
                # update ego state from the world data. Note: it is ground truth
                # TODO: implemente ego state estimation model.
                ego_st_vect = world.get_ego_state()

                # camera location in global frame
                cam_st = world.ego_camdep.get_transform()
                cam_x = cam_st.location.x  # camera coordinates already in global frame
                cam_y = cam_st.location.y  # camera coordinates already in global frame
                cam_yaw = ego_st_vect.yaw  # camera yaw equals ego yaw

                if simulation_step % UPDATE_CONTROL_RATE == 0:
                    ### Planning loop
                    ego_vel_planned = lp._velocity_planner.get_open_loop_speed(ego_st_vect.ts - ts_prev)

                    # TODO: behavioral planner
                    bp.set_lookahead(ego_st_vect.vel)
                    bp.transition_state(route.extended_wp, ego_st_vect)
                    # TODO: ~behavioral planner

                    # TODO: check for lead car position###
                    ######

                    # TODO: local planner
                    # Get a number of possible local goal states
                    local_goal_states = lp.get_goal_state_set(bp.get_goal_state_ix(), route.extended_wp, ego_st_vect)
                    # Compute local paths (local frame) corresponding to the possible local goal states
                    paths, path_validity = lp.plan_paths(local_goal_states)
                    # transform paths to the global frame
                    paths = lp.transform_paths(paths, ego_st_vect)
                    # goal state
                    goal_st = bp.get_goal_state()
                    # local yaw change identifies the turn direction
                    goal_direction = lp.get_goal_direction()

                    # TODO: perform collision checking against occupancy grid
                    # Use dummy obstacle points during no_dynamic_obstacles_mode
                    dummyObstaclePoints = []
                    for i in range(4):
                        dummyObstaclePoints.append([0, 0])

                    collision_check_array = lp._collision_checker.collision_check(paths, [dummyObstaclePoints]) #[parkedcar_box_pts])

                    bof.set_lines_global(cam_x, cam_y, cam_yaw)
                    left_lines_glob, right_lines_glob = bof.get_lines()

                    ## While turning left check right line boarders
                    ## While turning right check left line boarders
                    ## Goal directs = 0 yields no direction check leaving collision_check_array with dynamic obstacles
                    if goal_direction == -1 and right_lines_glob is not None:
                        collision_check_array = lp._collision_checker.lane_boundary_check(paths,
                                                                                        collision_check_array,
                                                                                        right_lines_glob)
                    if goal_direction == 1 and left_lines_glob is not None:
                        collision_check_array = lp._collision_checker.lane_boundary_check(paths,
                                                                                          collision_check_array,
                                                                                          left_lines_glob)

                    best_path_index = lp._collision_checker.select_best_path_index(paths,
                                                                                   collision_check_array,
                                                                                   goal_st)

                    if best_path_index == None and lp._prev_best_path == None:
                        best_path_index = 0
                        best_path = paths[best_path_index]

                    # TODO CHECK
                    elif best_path_index == None:
                        best_path = lp._prev_best_path
                    # ~TODO
                    else:
                        best_path = paths[best_path_index]
                        lp._prev_best_path = best_path

                    # update desired speed
                    # FIXME: change to the speed after velocity planner
                    # Used in the velocity profile open vs. close loop. See mt_plotter
                    ds = goal_st.s

                    # update lead car state
                    # TODO: update lead car state here
                    #
                    decelerate_to_stop = False
                    follow_lead_vehicle = False
                    lead_car_state = [0,0,0]#[lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
                    local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, ego_st_vect.vel,
                                                                                    ego_vel_planned,
                                                                                    decelerate_to_stop, lead_car_state,
                                                                                    follow_lead_vehicle)

                    # Optional: apply fine interpolation method for local wps
                    # additional smoothness of control
                    wp_local_interp = ll_controller.fine_interpolation(local_waypoints)

                    # Update low level controller with a new local path
                    ll_controller.update_waypoints(wp_local_interp)
                    # ~TODO: local planner
                    ### End of the planning loop

                if wp_local_interp != None and wp_local_interp != []:
                    # update low level controller
                    ll_controller.update_values(ego_st_vect)
                    # calculate a new low level controls
                    ll_controller.update_controls()
                    #ds = ll_controller.get_desired_speed()
                    st, b, t = ll_controller.get_commands()
                else:
                    st = 0
                    b = 0
                    t = 0

                # send a low level control commands
                world.write_ego_control(st, b, t)

                ## Plotting updates and refresh
                if ENABLE_PLOTTING:
                    if ENABLE_PLOT_TRACK:
                        mt_plotter.tr_fig.roll(mt_plotter.fig_names["trajectory"], ego_st_vect.x, ego_st_vect.y)
                        mt_plotter.tr_fig.roll(mt_plotter.fig_names["ego"], ego_st_vect.x, ego_st_vect.y)

                    if ENABLE_PLOT_CONTROL:
                        mt_plotter.ego_speed_fig.roll(mt_plotter.fig_names["speed"], simulation_step, ego_st_vect.vel)
                        mt_plotter.ego_speed_fig.roll(mt_plotter.fig_names["des_speed"], simulation_step, ds)
                        mt_plotter.throttle_fig.roll(mt_plotter.fig_names["throttle"], simulation_step, t)
                        mt_plotter.brake_fig.roll(mt_plotter.fig_names["brake"], simulation_step, b)
                        mt_plotter.steer_fig.roll(mt_plotter.fig_names["steer"], simulation_step, st)

                    if left_lines_glob is not None:
                        for i in range(len(left_lines_glob)):
                            x1 = left_lines_glob[i][0]
                            y1 = left_lines_glob[i][1]
                            x2 = left_lines_glob[i][2]
                            y2 = left_lines_glob[i][3]
                            mt_plotter.tr_fig.update(mt_plotter.fig_names["llines"] + str(i), [x1,x2], [y1,y2], 'b')

                        for i in range(len(left_lines_glob), 20):
                            mt_plotter.tr_fig.update(mt_plotter.fig_names["llines"] + str(i), [0, 0], [0, 0], 'b')

                    if right_lines_glob is not None:
                        for i in range(len(right_lines_glob)):
                            x1 = right_lines_glob[i][0]
                            y1 = right_lines_glob[i][1]
                            x2 = right_lines_glob[i][2]
                            y2 = right_lines_glob[i][3]
                            mt_plotter.tr_fig.update(mt_plotter.fig_names["rlines"] + str(i), [x1,x2], [y1,y2], 'g')

                        for i in range(len(right_lines_glob), 20):
                            mt_plotter.tr_fig.update(mt_plotter.fig_names["rlines"] + str(i), [0, 0], [0, 0], 'g')

                    # Local path plotter update
                    if ENABLE_PLOT_LOCAL_PATH and simulation_step % UPDATE_CONTROL_RATE == 0:
                        path_counter = 0
                        for i in range(lp.get_num_path()):
                            # If a path was invalid in the set, there is no path to plot.
                            if path_validity[i]:
                                # Colour paths according to collision checking.
                                if not collision_check_array[path_counter]:
                                    colour = 'r'
                                elif i == best_path_index:
                                    colour = 'k'
                                else:
                                    colour = 'b'
                                mt_plotter.tr_fig.update(mt_plotter.fig_names["paths"] + str(i), paths[path_counter][0],
                                                      paths[path_counter][1], colour)
                                path_counter += 1
                            else:
                                mt_plotter.tr_fig.update(mt_plotter.fig_names["paths"]+ str(i), [ego_st_vect.x], [ego_st_vect.y], 'r')

                        mt_plotter.refresh_plots()

                ## Update the simulation step and timestep
                simulation_step += 1
                ts_prev = ego_st_vect.ts


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-q', '--quality',
        metavar='Q',
        default='low',
        help='world graphics quality')
    argparser.add_argument(
        '-t', '--timeperiod',
        metavar='T',
        default=0.02,
        type=float,
        help='server sim time period in [ms]')

    args = argparser.parse_args()

    # Execute when server connection is established
    while True:
        #try:
        exec_autopilot_nav(args)
        print('Done.')
        # return
        # except TCPConnectionError as error:
        #    logging.error(error)
        #    time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
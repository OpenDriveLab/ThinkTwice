#!/usr/bin/env python

# Copyright (c) 2021 IBISC Laborartory, Pelvoux, France

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

##From https://github.com/AbanobSoliman/IBISCape

import random
import glob
import os
import sys

import csv


####Please Change it to your path of Carla **built from source**
egg_path = None
print("Egg Path:", egg_path, "(Please Change it to your path of Carla **built from source**!!!)")

sys.path.append(glob.glob(egg_path)[0])

import cv2
import carla
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_k
    from pygame.locals import K_j
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


import time
import subprocess


client = carla.Client('localhost', 2000)
client.set_timeout(20.0)


# ==============================================================================
# -- Creating the World --------------------------------------------------------
# ==============================================================================

global world
global m
world = client.get_world()
m = world.get_map()
self = world
self.recording = False

# ==============================================================================
# -- Output files Handling------------------------------------------------------
# ==============================================================================
try:
    parent_dir = os.getcwd()
    path1 = os.path.join(parent_dir, "RGBD_Calib_i_%s" % (m.name))
    os.mkdir(path1)
    path4 = os.path.join(path1, "rgb")
    os.mkdir(path4)
    path8 = os.path.join(path4, "frames")
    os.mkdir(path8)
    path12 = os.path.join(path1, "depth")
    os.mkdir(path12)
    path15 = os.path.join(path12, "frames")
    os.mkdir(path15)
    path10 = os.path.join(path1, "other_sensors")
    os.mkdir(path10)
    path11 = os.path.join(path1, "vehicle_gt")
    os.mkdir(path11)
except OSError:
    print("Creation of the directory %s failed" % path1)
    print("Creation of the directory %s failed" % path4)
    print("Creation of the directory %s failed" % path8)
    print("Creation of the directory %s failed" % path10)
    print("Creation of the directory %s failed" % path11)
    print("Creation of the directory %s failed" % path12)
    print("Creation of the directory %s failed" % path15)
else:
    print("Successfully created the directory %s " % path1)
    print("Successfully created the directory %s " % path4)
    print("Successfully created the directory %s " % path8)
    print("Successfully created the directory %s " % path10)
    print("Successfully created the directory %s " % path11)
    print("Successfully created the directory %s " % path12)
    print("Successfully created the directory %s " % path15)

file_rgb = open(os.path.join(path4, 'timestamps.csv'), "a")
file_depth = open(os.path.join(path12, 'timestamps.csv'), "a")
file_groundtruth_sync = open(os.path.join(path11, 'groundtruth_sync.csv'), "a")
file_GNSS = open(os.path.join(path10, 'gnss.csv'), "a")

writer0 = csv.writer(file_rgb, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer0.writerow(['#timestamp [ns]', 'filename'])
writer8 = csv.writer(file_depth, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer8.writerow(['#timestamp [ns]', 'filename'])
writer4 = csv.writer(file_groundtruth_sync, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer4.writerow(['#timestamp [ns]', 'p_RS_R_x_m_', 'p_RS_R_y_m_', 'p_RS_R_z_m_', 'v_RS_R_x_mS__1_', 'v_RS_R_y_mS__1_', 'v_RS_R_z_mS__1_', 'a_RS_S_x_mS__2_', 'a_RS_S_y_mS__2_', 'a_RS_S_z_mS__2_', 'q_RS_w__', 'q_RS_x__', 'q_RS_y__', 'q_RS_z__', 'w_RS_S_x_radS__1_', 'w_RS_S_y_radS__1_', 'w_RS_S_z_radS__1_'])
writer5 = csv.writer(file_GNSS, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer5.writerow(['#timestamp [ns]', 'latitude', 'longitude', 'altitude'])

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================
class gnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        bp.set_attribute('noise_alt_bias', '0.0')
        bp.set_attribute('noise_lat_bias', '0.0')
        bp.set_attribute('noise_lon_bias', '0.0')
        bp.set_attribute('noise_alt_stddev', '0.0')
        bp.set_attribute('noise_lat_stddev', '0.0')
        bp.set_attribute('noise_lon_stddev', '0.0')
        bp.set_attribute('noise_seed', '0')
        bp.set_attribute('sensor_tick', '0.05')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(
            x=-0.2, y=0, z=2.8), carla.Rotation(pitch=0, roll=0, yaw=0)), attach_to=self._parent, attachment_type=carla.AttachmentType.Rigid)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: gnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude
        self.tmp = event.timestamp

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


# ==============================================================================
# -- Useful Functions-----------------------------------------------------------
# ==============================================================================
def resize_img(img):
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    quat = [qw, qx, qy, qz]
    return quat/np.linalg.norm(quat)


def draw_rgb_image(surface, image_rgb):
    array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(resize_img(array.swapaxes(0, 1)))
    # image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def draw_depth_image(surface, image_depth):
    array = np.frombuffer(image_depth.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image_depth.height, image_depth.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(resize_img(array.swapaxes(0, 1)))
    # image_surface.set_alpha(100)
    surface.blit(image_surface, (1000, 0))
    
def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def create_main_actors(start_pose, waypoint):
    blueprint_library = world.get_blueprint_library()

    vehicle = world.spawn_actor(
        blueprint_library.filter('vehicle')[1], start_pose)
    vehicle.set_simulate_physics(True)
    vehicle.set_transform(waypoint.transform)

    bp_rgb = blueprint_library.find('sensor.camera.rgb')
    fov = "150.0"
    print("Current Fov:", fov, "Please Change it to your deisred fov!")
    # Modify the basic attributes of the blueprint
    bp_rgb.set_attribute('image_size_x', '1600')
    bp_rgb.set_attribute('image_size_y', '900')
    bp_rgb.set_attribute('sensor_tick', '0.05')
    bp_rgb.set_attribute('bloom_intensity', '0.675')
    bp_rgb.set_attribute('fov', fov)
    bp_rgb.set_attribute('fstop', '1.4')
    bp_rgb.set_attribute('iso', '100.0')
    bp_rgb.set_attribute('gamma', '2.2')
    bp_rgb.set_attribute('lens_flare_intensity', '0.1')
    bp_rgb.set_attribute('shutter_speed', '200.0')
    # Modify the lens distortion attributes
    bp_rgb.set_attribute('chromatic_aberration_intensity', '0.5')
    bp_rgb.set_attribute('chromatic_aberration_offset', '0.0')
    bp_rgb.set_attribute('lens_circle_falloff', '3.0')
    bp_rgb.set_attribute('lens_circle_multiplier', '3.0')
    bp_rgb.set_attribute('lens_k', '-1.0')
    bp_rgb.set_attribute('lens_kcube', '0.0')
    bp_rgb.set_attribute('lens_x_size', '0.08')
    bp_rgb.set_attribute('lens_y_size', '0.08')
    camera_rgb = world.spawn_actor(
        bp_rgb,
        carla.Transform(carla.Location(x=1.3, y=0.0, z=2.5),
                        carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0)),
        attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        
    bp_depth = blueprint_library.find('sensor.camera.depth')
    # Modify the basic attributes of the blueprint
    bp_depth.set_attribute('image_size_x', '160')
    bp_depth.set_attribute('image_size_y', '90')
    bp_depth.set_attribute('fov', fov)
    bp_depth.set_attribute('sensor_tick', '0.05')
    # Modify the lens distortion attributes
    bp_depth.set_attribute('lens_circle_falloff', '5.0')
    bp_depth.set_attribute('lens_circle_multiplier', '0.0')
    bp_depth.set_attribute('lens_k', '-1.0')
    bp_depth.set_attribute('lens_kcube', '0.0')
    bp_depth.set_attribute('lens_x_size', '0.08')
    bp_depth.set_attribute('lens_y_size', '0.08')
    camera_depth = world.spawn_actor(
        bp_depth,
        carla.Transform(carla.Location(x=0.0, y=0.0, z=2.8),
                        carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0)),
        attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)

    Gnss_sensor = gnssSensor(vehicle)

    return vehicle, camera_rgb, camera_depth, Gnss_sensor


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

       
def _RGB_sensor_callback(sensor_data, ts, display):
    draw_rgb_image(display, sensor_data)
    if self.recording:
        sensor_data.save_to_disk('RGBD_Calib_i_%s/rgb/frames/%d.png' % (m.name, ts))
        file_rgb.write("%d,%d.png \n" % (ts, ts))
        
def _DEPTH_sensor_callback(sensor_data, ts, display):
    sensor_data.convert(cc.LogarithmicDepth)
    draw_depth_image(display, sensor_data)
    if self.recording:
        sensor_data.save_to_disk('RGBD_Calib_i_%s/depth/frames/%d.png' % (m.name, ts))
        file_depth.write("%d,%d.png \n" % (ts, ts))

    
# ==============================================================================
# -- Sensor Synchronization-----------------------------------------------------
# ==============================================================================

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
    
# ==============================================================================
# -- Main Function--------------------------------------------------------------
# ==============================================================================


def main():
    actor_list = []
    pygame.init()
    pygame.display.set_caption(
        'IBISCape: Multi-modal Data Acquisition Framework')

    display = pygame.display.set_mode(
        (1200, 450),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    iterator = 1
    rvrs = False
    hand_brake = False    

    try:
        start_Pose = m.get_spawn_points()
        start_pose = start_Pose[32]
        waypoint = m.get_waypoint(start_pose.location)

        vehicle, camera_rgb, camera_depth, Gnss_sensor = create_main_actors(start_pose, waypoint)
        file_veh_sim = open(os.path.join(path11, '%s_simulation.txt' %
                                         get_actor_display_name(vehicle, truncate=200)), "a")
        file_associate = open(os.path.join(path11, 'association.txt'),"a")                                 

        actor_list.append(vehicle)
        actor_list.append(camera_rgb)
        actor_list.append(camera_depth)
        actor_list.append(Gnss_sensor)
        
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_depth, fps=20) as sync_mode:

            while True:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == K_r:
                            self.recording = not self.recording
                        elif event.key == K_q:
                            rvrs = not rvrs
                        elif event.key == K_SPACE:
                            vehicle.disable_constant_velocity()
                            vehicle.apply_control(
                                carla.VehicleControl(hand_brake=not hand_brake))
                        elif event.key == K_w:
                            if rvrs:
                                vehicle.apply_control(
                                    carla.VehicleControl(reverse=True, brake=0.0))
                                vehicle.enable_constant_velocity(
                                    carla.Vector3D(-3, 0, 0))
                            else:
                                vehicle.apply_control(
                                    carla.VehicleControl(throttle=1.0, brake=0.0))
                                vehicle.enable_constant_velocity(
                                    carla.Vector3D(3, 0, 0))
                        elif event.key == K_s:
                            vehicle.disable_constant_velocity()
                            vehicle.apply_control(
                                carla.VehicleControl(throttle=0.0, brake=1.0))
                        elif event.key == K_a:
                            vehicle.apply_control(
                                carla.VehicleControl(steer=-1.0))
                        elif event.key == K_d:
                            vehicle.apply_control(
                                carla.VehicleControl(steer=1.0))
                        elif event.key == pygame.K_ESCAPE:
                            return
                    elif event.type == pygame.KEYUP:
                        if event.key == K_a:
                            vehicle.apply_control(
                                carla.VehicleControl(steer=0.0))
                        elif event.key == K_d:
                            vehicle.apply_control(
                                carla.VehicleControl(steer=0.0))

                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_depth = sync_mode.tick(
                    timeout=2.0)
                Pose = vehicle.get_transform()
                Accl = vehicle.get_acceleration()
                velo = vehicle.get_velocity()
                AngV = vehicle.get_angular_velocity()
                Ctrl = vehicle.get_control()
                t_world = snapshot.timestamp.elapsed_seconds*(10**9)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                _RGB_sensor_callback(image_rgb, t_world, display)
                _DEPTH_sensor_callback(image_depth, t_world, display)               
                
                if self.recording:
                    file_GNSS.write("%f,%21.21f,%21.21f,%21.21f \n" % (t_world, Gnss_sensor.lat, Gnss_sensor.lon, Gnss_sensor.alt))
                    quat = euler_to_quaternion(math.radians(Pose.rotation.yaw), math.radians(Pose.rotation.pitch), math.radians(Pose.rotation.roll))
                    file_groundtruth_sync.write("%d,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f,%21.21f \n" % (t_world, Pose.location.x, Pose.location.y, Pose.location.z, velo.x, velo.y, velo.z, Accl.x, Accl.y, Accl.z, quat[0], quat[1], quat[2], quat[3], math.radians(AngV.x), math.radians(AngV.y), math.radians(AngV.z)))
                    file_veh_sim.write("%d,%f,%f,%f \n" % (
                        t_world, Ctrl.throttle, Ctrl.steer, Ctrl.brake))
                    file_associate.write("%d rgb/%d.png %d depth/%d.png \n" %(t_world,t_world,t_world,t_world))
                    display.blit(font.render('Calibration Frame Number: %d' % (
                        iterator), True, (250, 0, 0)), (50, 208))
                    iterator += 1

                display.blit(
                    font.render('% 5d FPS (real)' %
                                clock.get_fps(), True, (250, 0, 0)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' %
                                fps, True, (250, 0, 0)),
                    (8, 28))
                display.blit(font.render('Recording %s' % (
                    'On' if self.recording else 'Off - Press R Key to Record'), True, (250, 0, 0)), (50, 46))
                display.blit(font.render('GNSS:% 24s' % ('(% 3.8f, % 3.8f, % 3.8f)' % (
                    Gnss_sensor.lat, Gnss_sensor.lon, Gnss_sensor.alt)), True, (250, 0, 0)), (50, 100))
                display.blit(font.render('x=%5.8f,y=%5.8f,z=%5.8f (m)' % (
                    Pose.location.x, Pose.location.y, Pose.location.z), True, (250, 0, 0)), (50, 118))
                display.blit(font.render('vx=%5.8f,vy=%5.8f,vz=%5.8f (m/sec)' % (
                    velo.x, velo.y, velo.z), True, (250, 0, 0)), (50, 136))
                display.blit(font.render('ax=%5.8f,ay=%5.8f,az=%5.8f (m/sec^2)' % (
                    Accl.x, Accl.y, Accl.z), True, (250, 0, 0)), (50, 154))
                display.blit(font.render('roll=%5.8f,pitch=%5.8f,yaw=%5.8f (rad)' % (
                    math.radians(Pose.rotation.roll), math.radians(Pose.rotation.pitch), math.radians(Pose.rotation.yaw)), True, (250, 0, 0)), (50, 172))
                display.blit(font.render('r_d=%5.8f,p_d=%5.8f,y_d=%5.8f (rad/sec)' % (
                    math.radians(AngV.x), math.radians(AngV.y), math.radians(AngV.z)), True, (250, 0, 0)), (50, 190))

                pygame.display.flip()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        file_rgb.close()
        file_depth.close()
        file_groundtruth_sync.close()
        file_GNSS.close()
        file_veh_sim.close()
        file_associate.close()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

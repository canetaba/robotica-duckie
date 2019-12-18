#!/usr/bin/env python3
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

####
from PIL import Image
import cv2
import math
from apriltag import Detector
import transformations as tf
####

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown')
parser.add_argument('--map-name', default='udem2')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)
def _draw_pose(overlay, camera_params, tag_size, pose, z_sign=1):

    opoints = np.array([
        -1, -1, 0,
         1, -1, 0,
         1,  1, 0,
        -1,  1, 0,
        -1, -1, -2*z_sign,
         1, -1, -2*z_sign,
         1,  1, -2*z_sign,
        -1,  1, -2*z_sign,
    ]).reshape(-1, 1, 3) * 0.5*tag_size

    edges = np.array([
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        5, 6,
        6, 7,
        7, 4
    ]).reshape(-1, 2)
        
    fx, fy, cx, cy = camera_params

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3,:3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
    ipoints = np.round(ipoints).astype(int)
    ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]
    for i, j in edges:
        cv2.line(overlay, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)


# Posicion global
def global_pose(matrix, x_ob, y_ob, angle):
    # obtiene el angulo del tag con respecto al mapa
    q1 = math.atan2(y_ob, x_ob)
    # invierte el angulo del tag segun el plano del mapa
    angle = -angle
    # Calcula la distancia del robot al tag
    z = dist(matrix)
    # Calcula la distancia del tag al mapa
    d = math.sqrt(x_ob ** 2 + y_ob ** 2)
    # Calcula el angulo del robot c/r a q1
    q2 = angle2(q1, angle, tf.euler_from_matrix(matrix))

    # Matrices de rotacion para la posicion de robot
    R1 = tf.rotation_matrix(q1, [0, 0, 1])
    T1 = tf.translation_matrix([d, 0, 0])
    R2 = tf.rotation_matrix(q2, [0, 0, 1])
    T2 = tf.translation_matrix([z, 0, 0])
    result = R1.dot(T1.dot(R2.dot(T2.dot([0, 0, 0, 1]))))

    return result


def angle2(q, angle, euler):
    return q - (angle - yaw(euler))


def l1(x, y):
    return math.sqrt(x ** 2, y ** 2)


def yaw(euler_angles):
    return euler_angles[2]


def dist(matrix):
    return np.linalg.norm([matrix[0][3], matrix[1][3], matrix[2][3]])


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)

    ### apriltags detector
    label = ""
    original = Image.fromarray(obs)
    cv_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    detector = Detector()
    gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    detections, dimg = detector.detect(gray, return_image=True)
    camera = [305.57, 308.83, 303.07, 231.88]

    pose = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    robot_pose = [0.0, 0.0]
    for detection in detections:

        pose, e0, e1 = detector.detection_pose(detection, camera, 0.18 / 2 *0.585)
        if not np.isnan(pose[0][0]):
            _draw_pose(cv_img,
                       camera,
                       0.18/ 2 *0.585,
                       pose)
            
        robot_pose = global_pose(pose, 2.08*0.585, 4.05*0.585, math.pi/2)
        
    label = 'detections = %d, dist = %.2f, pos = (%.2f, %.2f)' % (len(detections), pose[2][3], robot_pose[0], robot_pose[1])
    
    cv2.imshow('win', cv_img)
    cv2.waitKey(5)

    #if done:
    #   print('done!')
    #    env.reset()
    #    env.render()
    extra_label = pyglet.text.Label(
        font_name="Arial",
        font_size=14,
        x=5,
        y=600 - 19*2
    )
    env.render(mode="top_down")
    extra_label.text = label
    extra_label.draw()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
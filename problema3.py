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

MAP_ORIGIN = [0,0]
UP = [0.44, 0.0]
DOWN = [-0.44, 0]
RIGHT = [0.35, -1]
LEFT =[0.35, +1]

X = 3.5
y = 5.9
angle = (np.pi/2)
tag_size = 0.18
tile_size = 0.585



# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown')
parser.add_argument('--map-name', default='udem1')
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


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array(MAP_ORIGIN)

    if key_handler[key.UP]:
        action = np.array(UP)
    if key_handler[key.DOWN]:
        action = np.array(DOWN)
    if key_handler[key.LEFT]:
        action = np.array(LEFT)
    if key_handler[key.RIGHT]:
        action = np.array(RIGHT)
    if key_handler[key.SPACE]:
        action = np.array(MAP_ORIGIN)

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)

    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        #env.reset()
        #env.render()

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



# Calcula la pose del apriltag
def process_april_tag(pose):
    ## Aquí jugar
    ## pose es tag con respecto al robot
    ## T_a es la transformación del april tag con respecto al mapa
    T_a_m = tf.translation_matrix([-X * tile_size, -tag_size * 3/4, y*tile_size])
    R_a = tf.euler_matrix(0, angle, 0)


    ## Aquí dando vuelta el robot, por cuenta del cambio de los angulos
    T_r_a = np.dot(pose, tf.euler_matrix(0, np.pi, 0))
    T_a_r = np.linalg.inv(T_r_a)

    # Posicion del robot con respecto al apriltag
    T_a_r = np.dot(T_a_r, pose)
    print(T_a_r)


def global_pose(matrix,x_ob,y_ob,angle):
    #obtiene el angulo del tag con respecto al mapa
    q1 = math.atan2(y_ob,x_ob)
    # invierte el angulo del tag segun el plano del mapa
    angle = -angle
    # Calcula la distancia del robot al tag
    z = dist(matrix)
    # Calcula la distancia del tag al mapa
    d = math.sqrt(x_ob**2 + y_ob**2)
    # Calcula el angulo del robot c/r a q1
    q2 = angle2(q1,angle,tf.euler_from_matrix(matrix))

    # Matrices para el apriltag
    R1 = tf.rotation_matrix(q1,[0,0,1])
    T1 = tf.translation_matrix([d,0,0])

    # Matriz para el robot
    R2 = tf.rotation_matrix(q2,[0,0,1])
    T2 = tf.translation_matrix([z,0,0])
    result = R1.dot(T1.dot(R2.dot(T2.dot([0,0,0,1]))))
    
    return result
    

def angle2(q,angle,euler):
    return q-(angle-yaw(euler))

def l1(x,y):
    return math.sqrt(x**2,y**2)

def yaw(euler_angles):
    return euler_angles[2]

def dist(matrix):
    return np.linalg.norm([matrix[0][3],matrix[1][3],matrix[2][3]])

# Calcula la matriz inversa
def inverse_matriz(matrix):
    return tf.transformations.inverse_matrix(matrix)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array(UP)
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
    camera = [305.57, 308.83, 303.07, 231.88]
    tag_size = 0.18 / 2

    # Referencia mapa
    pose = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    robot_pose = MAP_ORIGIN
    robot_pose_apriltag = MAP_ORIGIN
    label = ""
    label2 = ""
    original = Image.fromarray(obs)
    cv_img = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    detector = Detector()
    tag_x = 2.08
    tag_y = 4.05
    tile_size = 0.585
    detections, dimg =detector.detect(gray, return_image=True)


    for detection in detections:
        pose, e0, e1 = detector.detection_pose(detection, camera, tag_size/2 )
        if not np.isnan(pose[0][0]):
            _draw_pose(cv_img,camera, tag_size/2,pose)

        robot_pose = global_pose(pose,2.08*0.585, 4.05*0.585, math.pi/2 )
        label2 = "Coordenadas robot" % (robot_pose)

        # Necesitamos saber la posicion del robot con respecto al apriltag
        #robot_pose_apriltag = inverse_matriz(robot_pose)
        #print(robot_pose_apriltag)



    label = 'detections = %d, dist = %.2f, pos = (%.2f, %.2f)' % (len(detections), pose[2][3], robot_pose[0], robot_pose[1])
    #label = 'robot_pose_apriltag = %d, dist = %.2f, posRobAp = (%.2f, %.2f)' % (len(detections), robot_pose_apriltag[2][3], robot_pose_apriltag[0], robot_pose_apriltag[1])
    
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
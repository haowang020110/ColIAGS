from typing import NamedTuple
import numpy as np
import os
import math
import json
import torch
from scene.cameras import Camera 
import copy
import matplotlib.pyplot as plt
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    depth_path: str
    depth_name: str
    width: int
    height: int

def normalize(x):
    return x / np.linalg.norm(x)

# cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                         image_path=image_path, image_name=image_name, 
#                         depth=depth, depth_name=depth_name, depth_path=depth_path,
#                         width=width, height=height)
# cam_infos.append(cam_info)
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w

def set_axes_equal(ax):
    """Set equal scaling for 3D axes."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    mid_x = (x_limits[1] + x_limits[0]) / 2
    mid_y = (y_limits[1] + y_limits[0]) / 2
    mid_z = (z_limits[1] + z_limits[0]) / 2

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

def visualize_camera_pose(c2ws, c2ws_spiral):
    print('Visualizing camera poses')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original c2ws (solid lines)
    for i in range(c2ws.shape[0]):
        c2w = c2ws[i]
        x_axis, y_axis, z_axis, pos = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3]
        ax.quiver(*pos, *x_axis, color='pink')
        ax.quiver(*pos, *y_axis, color='lightgreen')
        ax.quiver(*pos, *z_axis, color='lightblue')
    # plot original c2ws' mean c2w_avg
    # c2w_avg = poses_avg(c2ws)
    # x_axis, y_axis, z_axis, pos = c2w_avg[:3, 0], c2w_avg[:3, 1], c2w_avg[:3, 2], c2w_avg[:3, 3]
    # ax.quiver(*pos, *x_axis, color='black', label='c2ws_avg')
    # ax.quiver(*pos, *y_axis, color='yellow')
    # ax.quiver(*pos, *z_axis, color='magenta')
    # Plot c2ws_spiral (dashed and lighter colors)
    for i in range(c2ws_spiral.shape[0]):
        c2w = c2ws_spiral[i]
        x_axis, y_axis, z_axis, pos = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3]
        ax.quiver(*pos, *x_axis, color='r', alpha=0.5, label='c2ws_spiral' if i == 0 else "")
        ax.quiver(*pos, *y_axis, color='g', alpha=0.5)
        ax.quiver(*pos, *z_axis, color='b', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.legend()
    plt.show()

def visualize_camera_pose_seperate(c2ws):
    print('visualize_camera_pose')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(c2ws.shape[0]):
        c2w = c2ws[i]
        x_axis = c2w[:3, 0]
        y_axis = c2w[:3, 1]
        z_axis = c2w[:3, 2]
        pos = c2w[:3, 3]
        # plot camera pose
        ax.quiver(*pos, *(x_axis), color='r')
        ax.quiver(*pos, *(y_axis), color='g')
        ax.quiver(*pos, *(z_axis), color='b')
        ax.text(*(pos + x_axis), 'x', color='r', fontsize=12)
        ax.text(*(pos + y_axis), 'y', color='g', fontsize=12)
        ax.text(*(pos + z_axis), 'z', color='b', fontsize=12)
    c2w_avg = poses_avg(c2ws)
    x_axis = normalize(c2w_avg[:3, 0])
    y_axis = normalize(c2w_avg[:3, 1])
    z_axis = normalize(c2w_avg[:3, 2])
    pos = c2w_avg[:3, 3]
    ax.quiver(*pos, *(x_axis), color='k')
    ax.quiver(*pos, *(y_axis), color='y')
    ax.quiver(*pos, *(z_axis), color='m')
        # plot label
    ax.quiver([], [], [], [], [], [], color='r', label='x')
    ax.quiver([], [], [], [], [], [], color='g', label='y')
    ax.quiver([], [], [], [], [], [], color='b', label='z')
    ax.legend()
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    set_axes_equal(ax)
    ax.set_title('3D Point Cloud Visualization')
    plt.show()


def camera_path_spiral(c2ws, focal=1, zrate=.5, rots=3, N=300):
    c2w = poses_avg(c2ws)
    # add visualize func to see the c2w and c2ws
    print('c2w', c2w.shape)
    print('c2ws', c2ws.shape)
    c2w1 = c2ws[0]
    
    # visualize_camera_pose(c2ws)
    up = normalize(c2ws[:, :3, 1].sum(0))
    tt = c2ws[:,:3,3]
    rads = np.percentile(np.abs(tt), 90, 0)
    rads[:] = rads.max() * .01
    print('rads', rads)
    
    render_poses = []
    rads = np.array(list(rads) + [1.])
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        # z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))

        # 因为我们用的Z轴朝前
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), np.sin(theta), np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(np.dot(c2w[:3,:4], np.array([0,0, focal, 1.])) - c)

        render_poses.append(viewmatrix(z, up, c))
    render_poses = np.stack(render_poses, axis=0)
    render_poses = np.concatenate([render_poses, np.zeros_like(render_poses[..., :1, :])], axis=1)
    render_poses[..., 3, 3] = 1
    render_poses = np.array(render_poses, dtype=np.float32)

    visualize_camera_pose(c2ws, render_poses)
    # visualize_camera_pose_seperate(render_poses)
    return render_poses

def spiral_cam_info(views, focal=10):
    cam_num = len(views)
    print(f'spiral num {cam_num}')
    c2ws = np.zeros((cam_num, 4, 4))
    for i in range(cam_num):
        w2c_tmp = np.eye(4)
        # w2c_tmp[3,3]=1
        # print('R', views[i].R)
        # print('R_transpose()', np.transpose(views[i].R))
        w2c_tmp[:3, :3] = np.transpose(views[i].R)
        w2c_tmp[:3, 3] = views[i].T
        c2ws[i] = np.linalg.inv(w2c_tmp)

    
    width = views[i].image_width
    height = views[i].image_height
    fovx = views[i].FoVx
    fovy = views[i].FoVy
    
    render_poses = camera_path_spiral(c2ws, focal=focal, N=300, rots=3)
    # print('render_poses', render_poses[].shape)
    cam_infos = []

    for i, pose in enumerate(render_poses):
        # print('pose', pose.shape, pose)
        # if i >= cam_num:
        #     continue
        view_new = copy.deepcopy(views[0])
        # w2c = pose

        w2c = np.linalg.inv(pose)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        #GSCamera(colmap_id=idx, R=R, T=T, FoVx=fovx, FoVy=fovy, image=torch.zeros((3, height, width)), depth=torch.zeros((3, height, width)), gt_alpha_mask=None, image_name ='fake', uid=0)
        view_new.modify_camera(R, T)
        cam_infos.append(view_new)
    print('cam_info', len(cam_infos))
    return cam_infos

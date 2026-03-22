from typing import NamedTuple
from PIL import Image
from scipy.spatial.transform import Rotation as R
from typing import List
import numpy as np
import os
import sys
import math
import json
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import focal2fov, fov2focal
from scene.colmap_loader import Camera
import cv2
from tqdm import tqdm

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
    depth_params: dict
    is_test: bool
    specular_mask: np.array
    intrinsic: np.array


def read_C3VD_c2w(extrinsics_file):
    c2w = np.loadtxt(extrinsics_file, delimiter=',').reshape(-1,4,4).transpose(0,2,1)
    return c2w

def read_C3VD_intrinsics(intrinsics_file):
    with open(intrinsics_file, 'r') as file:
        intrinsics = json.load(file)
    K =  np.array([[intrinsics['fx'], 0, intrinsics['cx']],
                   [0, intrinsics['fy'], intrinsics['cy']],
                    [0,0,1]])
    intrinsics = {'K':K, 'h':intrinsics['h'], 'w':intrinsics['w']}
    return intrinsics

def readC3VD(extrinsics_file, intrinsics_file, images_folder, depths_folder):
    # all image files that are png or jpg in images_folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.png') or f.endswith('.jpg')]
    depth_files = [f for f in os.listdir(depths_folder) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.tiff')]

    # if image files are of the format '1_color.png'  # now use this
    if len(image_files[0].split("_")) == 2:
        image_files.sort(key=lambda f: int(f.split("_")[0]))
        depth_files.sort(key=lambda f: int(f.split("_")[0]))
    # if image files are of the format 'frame018250.jpg' or 'frame018251.jpg' 
    elif len(image_files[0].split("frame")) == 2:
        image_files.sort(key=lambda f: int(f.split("frame")[1].split(".")[0]))
        depth_files.sort(key=lambda f: int(f.split(".")[0]))
    # if image files are of the format '1.png'
    elif len(image_files[0].split(".")) == 2:
        image_files.sort(key=lambda f: int(f.split(".")[0]))
        depth_files.sort(key=lambda f: int(f.split(".")[0]))
    # if image files are of the format '1305031102.175304.png'
    elif len(image_files[0].split(".")) == 3:
        image_files.sort(key=sort_key)
        depth_files.sort(key=sort_key)
    else:
        assert False, "Image file format not recognized!"

    try:
        cam_c2w = read_C3VD_c2w(extrinsics_file)
        cam_intrinsics = read_C3VD_intrinsics(intrinsics_file)
    except Exception as e:
        print('Error reading extrinsics file: {}'.format(e))
        return None
    
    cam_infos = []
    for idx in range(cam_c2w.shape[0]): 
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{} for C3VD data".format(idx+1, len(cam_c2w)))
        sys.stdout.flush()

        c2w = cam_c2w[idx]
        w2c = np.linalg.inv(c2w)
        intr = cam_intrinsics
        height = intr['h']
        width = intr['w']

        uid = idx
        # also check format opencv or opengl
        R = np.transpose(w2c[:3,:3]) # check this # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3,3]

        focal_length_y = intr['K'][1,1]
        focal_length_x = intr['K'][0,0]
        
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        

        image_path = os.path.join(images_folder, image_files[idx])
        assert int(image_files[idx].split('_')[0]) == idx, "Image file name does not match the index"
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # load depth 
        depth_path = os.path.join(depths_folder, depth_files[idx])
        depth_name = os.path.basename(depth_path).split(".")[0]


        depth = np.array(cv2.imread(str(depth_path),-1))
        depth = depth.astype(np.float32)
        depth = ((depth)/(2**16-1))
        depth[depth==0]=np.nan
        depth = (depth * 100).astype(np.float32)
        depth = Image.fromarray(depth) 


        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, depth=depth, depth_params=None,
                              depth_path=depth_path, depth_name=depth_name, width=width, height=height, is_test=False, 
                              specular_mask=None, intrinsic=intr['K'])
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def init_pts_C3VD(train_cam_infos: List[CameraInfo], sampling='random'):
    pts_total, colors_total = [], []
    for idx in tqdm(range(len(train_cam_infos)), desc='init pts'):
        color, depth, mask = np.array(train_cam_infos[idx].image), np.array(train_cam_infos[idx].depth), None
        assert color.shape[-1] == 3, f"Color image should have 3 channels, but got {color.shape}"
        focal_x, focal_y = fov2focal(train_cam_infos[idx].FovX, depth.shape[1]), fov2focal(train_cam_infos[idx].FovY, depth.shape[0])
        pts, colors, _ = get_pts_cam(depth, mask, color, focal_x, focal_y, disable_mask=True)
        R = train_cam_infos[idx].R.transpose()
        T = train_cam_infos[idx].T
        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3,3] = T
        c2w = np.linalg.inv(w2c)

        pts = get_pts_wld(pts, c2w)
        pts_total.append(pts)
        colors_total.append(colors)
        
        num_pts = pts.shape[0]
        if sampling == 'fps':
            sel_idxs = fpsample.bucket_fps_kdline_sampling(pts, int(0.1*num_pts), h=3)
        elif sampling == 'random':
            sel_idxs = np.random.choice(num_pts, int(0.1*num_pts), replace=False)
        else:
            raise ValueError(f'{sampling} sampling has not been implemented yet.')
        
        pts_sel, colors_sel = pts[sel_idxs], colors[sel_idxs]
        pts_total.append(pts_sel)
        colors_total.append(colors_sel)
    
    pts_total = np.concatenate(pts_total)
    colors_total = np.concatenate(colors_total)
    sel_idxs = np.random.choice(pts_total.shape[0], 200000, replace=True)
    pts, colors = pts_total[sel_idxs], colors_total[sel_idxs]
    normals = np.zeros((pts.shape[0], 3))
    
    return pts, colors, normals

def get_pts_cam(depth, mask, color, focal_x, focal_y, disable_mask=False):
        W, H = depth.shape[1], depth.shape[0]
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        X_Z = (i-W/2) / focal_x
        Y_Z = (j-H/2) / focal_y
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = color.reshape(-1, 3)
        
        if not disable_mask:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam[mask, :]
            color_valid = color[mask, :]
        else:
            pts_valid = pts_cam
            color_valid = color
                    
        return pts_valid, color_valid, mask

def get_pts_wld(pts, pose):
        c2w = pose
        pts_cam_homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(c2w @ np.transpose(pts_cam_homo))
        pts_wld = pts_wld[:, :3]
        return pts_wld
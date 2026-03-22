#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import cv2
import numpy as np
# def l1_loss(network_output, gt):
#     return torch.abs((network_output - gt)).mean()
def l1_loss(network_output, gt, mask=None):
    loss = torch.abs((network_output - gt))
    # print('mask',mask.shape)
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        elif mask.ndim == 3:
            mask = mask.repeat(network_output.shape[0], 1, 1)
        else:
            raise ValueError('the dimension of mask should be either 3 or 4')
    
        try:
            loss = loss[mask!=0]
        except:
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
    return loss.mean()
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, mask=None, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if mask is not None:
        mask = mask.cpu().numpy().astype(np.uint8)
        # print(np.sum(mask.astype(np.uint8)), mask.shape)
        contours, _ = cv2.findContours(mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        # print(x, y, w, h)
        img1 = img1[:, y:y + h, x:x + w]
        img2 = img2[:, y:y + h, x:x + w]
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out

def predicted_normal_loss(normal, normal_ref, alpha=None, threshold=0.05):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: ( H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()
        weight[weight < threshold] = 0.0
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device)
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3)
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss

def specular_normal_alignment_loss(normal_map, specular_mask, intrinsic, depth_map):
    """
    Args:
        normal_map: (3, H, W), torch.float32, normalized normals in camera coordinate
        specular_mask: (H, W), torch.bool or binary mask, 1 for specular highlights
        intrinsic: (3, 3), camera intrinsics
    Returns:
        scalar loss: mean(1 - dot(normal, view_dir)) over specular region
    """

    # Step 1: 生成像素坐标 (u,v)
    _, H, W = normal_map.shape
    device = normal_map.device
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    ones = torch.ones_like(x)
    pix_coords = torch.stack((x, y, ones), dim=-1).float()  # (H, W, 3)

    # Step 2: 转换为相机坐标方向（从像素指向相机中心）
    K_inv = torch.inverse(intrinsic.to(device))  # (3,3)
    view_dirs = pix_coords @ K_inv.T  # (H, W, 3)
    view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-6)  # normalize

    # Step 3: 提取对应位置法线和视线方向
    normals = normal_map.permute(1,2,0)  # (H, W, 3)
    mask = specular_mask.bool()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)  # no specular region

    normals = normals[mask]             # (N, 3)
    views = view_dirs[mask]             # (N, 3)

    # Step 4: dot product（希望越接近1越好）
    dot = (normals * views).sum(dim=-1)
    loss = 1 - dot  # 越接近1越好

    return loss.mean()

def depth_to_point_map(depth, intrinsic):
    """
    Args:
        depth: (H, W) depth map in camera coordinates
        intrinsic: (3, 3) camera intrinsics
    Returns:
        point_map: (3, H, W), xyz in camera coordinates
    """
    H, W = depth.shape
    device = depth.device

    # 构造像素坐标网格 (u,v)
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    ones = torch.ones_like(x)
    pixel_coords = torch.stack((x, y, ones), dim=0).float()  # (3, H, W)

    # 相机坐标下点 = K^-1 @ (u,v,1) * depth
    K_inv = torch.inverse(intrinsic.to(device))
    cam_coords = K_inv @ pixel_coords.view(3, -1)  # (3, H*W)
    cam_coords = cam_coords.view(3, H, W)
    point_map = cam_coords * depth.unsqueeze(0)  # 广播 depth 到 (3, H, W)

    return point_map  # shape: (3, H, W)

def specular_normal_alignment_loss(normal_map, specular_mask, depth, intrinsic):
    """
    Args:
        normal_map: (3, H, W), normalized normals (camera coordinate)
        specular_mask: (H, W), binary mask of specular highlight (1 for specular)
        depth: (H, W), rendered depth in camera space
        intrinsic: (3, 3), camera intrinsics
    Returns:
        scalar loss: mean(1 - cosθ) over specular region
    """
    device = depth.device
    point_map = depth_to_point_map(depth, intrinsic)  # (3, H, W)
    view_dirs = -point_map
    view_dirs = torch.nn.functional.normalize(view_dirs, p=2, dim=0)

    mask = specular_mask.bool()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    normals = normal_map[:, mask]  # (3, N)
    views = view_dirs[:, mask]     # (3, N)

    dot = (normals * views).sum(dim=0).clamp(-1.0, 1.0)  # 余弦值
    loss = 1 - dot  # cos 趋近 1 越好
    return loss.mean()

def edge_aware_normal_loss(I, D):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device)/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device)/4

    dD_dx = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_x, padding=1) for i in range(D.shape[0])])
    dD_dy = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_y, padding=1) for i in range(D.shape[0])])
    
    dI_dx = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_x, padding=1) for i in range(I.shape[0])])
    dI_dx = torch.mean(torch.abs(dI_dx), 0, keepdim=True)
    dI_dy = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_y, padding=1) for i in range(I.shape[0])])
    dI_dy = torch.mean(torch.abs(dI_dy), 0, keepdim=True)

    weights_x = (dI_dx-1)**500
    weights_y = (dI_dy-1)**500

    loss_x = abs(dD_dx) * weights_x
    loss_y = abs(dD_dy) * weights_y
    loss = (loss_x + loss_y).norm(dim=0, keepdim=True)
    return loss.mean()

def spec_aware_reg_loss(spec_mask, D):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(D.device)/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(D.device)/4

    dD_dx = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_x, padding=1) for i in range(D.shape[0])])
    dD_dy = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_y, padding=1) for i in range(D.shape[0])])


def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co
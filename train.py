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

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import scaffold_render, network_gui, scaffold_prefilter_voxel
import sys
import torchvision.transforms.functional as tf
from scene import Scaffold_Scene, Scaffold_GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import Scaffold_ModelParams, Scaffold_OptimizationParams, PipelineParams
import json
import torchvision
import time
import shutil, pathlib
from pathlib import Path
import torch.nn.functional as F
from lpipsPyTorch import lpips
from metrics import readImages
import cv2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, logger=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = Scaffold_GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, 
                                dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scaffold_Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    depth_type = opt.depth_type
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # use_c2f = opt.use_c2f
    # smooth_term = get_linear_noise_func(lr_init=opt.c2f_init_factor, lr_final=1.0, lr_delay_mult=0.01,
    #                                     max_steps=opt.c2f_until_iter)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = scaffold_render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
            # gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        voxel_visible_mask = None
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        # down_sampling = smooth_term(iteration) if use_c2f else 1.0
        render_pkg = scaffold_render(viewpoint_cam, gaussians, pipe, bg, visible_mask=voxel_visible_mask,
                                   retain_grad=retain_grad)

        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg[
            "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], \
            render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        mse = 0
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            
            if depth_type == "l1":
                depth = invDepth.clone()
                depth[depth!=0] = 1 / depth[depth!=0]
                depth[depth==0] = viewpoint_cam.zfar
                mono_depth = mono_invdepth.clone()
                mono_depth[mono_depth!=0] = 1 / mono_depth[mono_depth!=0]
                mono_depth[mono_depth==0] = viewpoint_cam.zfar
                Ll1depth_pure = torch.abs((depth  - mono_depth) * depth_mask).mean()
            elif depth_type == "inverse_l1":
                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            else:
                raise ValueError("Unknown depth loss type")
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            

            loss += Ll1depth
            Ll1depth = Ll1depth.item()
            mse = ((1 / invDepth - 1 / mono_invdepth) ** 2 * depth_mask).mean().item()
        else:
            Ll1depth = 0
        scaling_reg = scaling.prod(dim=1).mean()
        # add loss used in scaffold gs to regularize the scaling
        loss += 0.01 * scaling_reg
        loss.backward()

        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "MSE": f"{mse:.{7}}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, scaffold_render, (pipe, background), dataset.train_test_exp, logger)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # anchor densification
            if iteration < opt.update_until and iteration > opt.start_stat:

                viewspace_point_tensor_densify = viewspace_point_tensor # densify no in graph
                gaussians.training_statis(viewspace_point_tensor_densify, opacity, visibility_filter, offset_selection_mask,
                                          voxel_visible_mask)

                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold,
                                            grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scaffold_Scene, renderFunc, renderArgs, train_test_exp, logger=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = None
                    # voxel_visible_mask = scaffold_prefilter_voxel(view, gaussians, pipeline, background)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type="vgg").mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                           
                logger.info(
                "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test,
                                                                                   psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(error_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(gt_depth_path, exist_ok=True)
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # voxel_visible_mask = scaffold_prefilter_voxel(view, gaussians, pipeline, background)
        voxel_visible_mask = None
        render_pkg = scaffold_render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)
        depth_rendering = render_pkg["depth"]
        depth_rendering = 1 / depth_rendering
        depth_rendering = depth_rendering / 150 * (2**16 - 1)
        depth_rendering = depth_rendering.permute(1, 2, 0).cpu().numpy().astype(np.uint16)
        # gts
        gt = view.original_image[0:3, :, :]
        gt_depth = 1 / view.invdepthmap
        gt_depth = gt_depth / 100 * (2**16 - 1)
        gt_depth = gt_depth.permute(1, 2, 0).cpu().numpy().astype(np.uint16)
        
        # error maps
        errormap = (rendering - gt).abs()

        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(depth_rendering, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), depth_rendering)
        cv2.imwrite(os.path.join(gt_depth_path, '{0:05d}'.format(idx) + ".png"), gt_depth)
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    for idx, view in enumerate(tqdm(views, desc="FPS test progress")):
        torch.cuda.synchronize();
        t_start = time.time()

        # voxel_visible_mask = scaffold_prefilter_voxel(view, gaussians, pipeline, background)
        voxel_visible_mask = None
        render_pkg = scaffold_render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();
        t_end = time.time()

        t_list.append(t_end - t_start)

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list        
def render_sets(dataset: Scaffold_ModelParams, iteration: int, pipeline: PipelineParams, skip_train=True, skip_test=False,
                tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = Scaffold_GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, 
                                    dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                    dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scaffold_Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            t_train_list, _ = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                                         gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter,
                                                    scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)

    return visible_count

def readDepths(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if fname.endswith(".png"): 
            # render = Image.open(renders_dir / fname).convert("L")
            # gt = Image.open(gt_dir / fname).convert("L")
            render = np.array(cv2.imread(renders_dir / fname,-1))
            gt = np.array(cv2.imread(gt_dir / fname,-1))
            # ---- testing next two line ----
            # make sure both are between 0-100 
            render = render / (2**16 - 1) * 150.0
            gt = gt / (2**16 - 1) * 100.0
            render_tensor = tf.to_tensor(render)
            gt_tensor = tf.to_tensor(gt)
            # render_tensor = normalize_tensor_to_range(render_tensor, 0, 1)
            
            renders.append(render_tensor.unsqueeze(0).cuda())
            gts.append(gt_tensor.unsqueeze(0).cuda())
            image_names.append(fname)
        
    return renders, gts, image_names

def evaluate(model_paths, tb_writer=None, dataset_name=None, logger=None):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)
        depth_dir = method_dir / "depth"
        gt_depth_dir = method_dir / "gt_depth"
        depths, gt_depths, _ = readDepths(depth_dir, gt_depth_dir)
        ssims = []
        psnrs = []
        lpipss = []
        mse_depths = []
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg").detach())
            mse_depths.append((torch.mean((depths[idx] - gt_depths[idx]) ** 2)).item())
        print('len(mse_depths)', len(mse_depths))
        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("  MSE Depth: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(mse_depths).mean(), ".5"))
        print("")

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/MSE Depth', torch.tensor(mse_depths).mean().item(), 0)

        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                             "PSNR": torch.tensor(psnrs).mean().item(),
                                             "LPIPS": torch.tensor(lpipss).mean().item(),
                                             "Depth_MSE": torch.tensor(mse_depths).mean().item()})
        per_view_dict[scene_dir][method].update(
            {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
             "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
             "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
             "Depth_MSE": {name: mse for mse, name in zip(torch.tensor(mse_depths).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = Scaffold_ModelParams(parser)
    op = Scaffold_OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 15_000] + list(range(20000, 30001, 1000)))
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 15_000] )
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # enable logging
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
              args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger=logger)

    # All done
    print("\nTraining complete.")
    # rendering
    logger.info(f'\nStarting Rendering~')
    for iteration in args.save_iterations:
        visible_count = render_sets(lp.extract(args), iteration, pp.extract(args), logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\nStarting evaluation...")
    evaluate(args.model_path, logger=logger)
    logger.info("\nEvaluating complete.")

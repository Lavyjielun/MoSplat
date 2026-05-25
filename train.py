
import os
import torch
import numpy as np
from random import randint
from gaussian_renderer import render_functions
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from arguments.config_handler import ConfigHandler
import PIL.Image as Image
from utils.general_utils import generate_heatmaps
from scene.dataset_readers import DataLoader
import hydra
from omegaconf import DictConfig
import sys
import logging
from utils import losses, early_stopping_strategy, consistency_losses
from utils.general_utils import unpack_covariance, OptEarlyStopping
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import random
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from itertools import combinations

#暂且是让上肢部分靠着身体17号关节链接，下肢部分靠着18号盆骨链接
PANOPTIC_JOINT_GROUPS = {
    "left_arm":[6,8,10],
    "right_arm":[5,9,7],
    "body":[17,0,1,2,3,4,18],
    "left_leg":[11,13,15],
    "right_leg":[12,14,16] 
}
# 如果是H36M (17 joints):
#脊椎有3个节点部分内容，传统派
H36M_JOINT_GROUPS = {
    "right_arm": [14, 16, 15],   # shoulder, elbow, wrist
    "left_arm": [12, 11, 13],    # shoulder, elbow, wrist
    "body": [0,8,7,9,10],           # hip, spine (作为根) 8连接双手0连接双腿
    "right_leg": [1, 2, 3],   # hip, knee, ankle
    "left_leg": [4, 6, 5], # hip, knee, ankle
}
FOSHAN_JOINT_GROUPS = {
    "right_arm": [10, 9, 8],   # shoulder, elbow, wrist
    "left_arm": [12, 11, 13],    # shoulder, elbow, wrist
    "body": [0,7],           # hip, spine (作为根) 8连接双手0连接双腿
    "right_leg": [4, 6, 5], # hip, knee, ankle
    "left_leg": [1, 2, 3],   # hip, knee, ankle

}
def training(dataset, model, opt, pipe, debug, training, dataset_loader, output_dir, log):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    #主损失函数和一致性损失函数
    opt_criterion = losses[training.loss_function]
    consistency_criterion = consistency_losses[training.consistency_loss]
    #开启3DGS渲染
    #动态获取渲染函数
    render = render_functions[pipe.rendering]
    #动态获取早停策略
    early_stopping = early_stopping_strategy[training.early_stopping]()
    aligned_traj = collect_aligned_joint_trajectories( 
        dataset, 
        dataset_loader, 
        model, opt, 
        output_dir, 
        num_frames=30)

    # ==================================================
    # Collect aligned j    oint trajectories
    # ==================================================
    #构建组内父子关系以及组间父子关系以及保存组间连接点
    joint_structure = infer_joint_structure_from_trajectories(
        aligned_traj,
        dataset.data_root
    )

    logging.info(
        f"Joint structure built: "
        f"{len(joint_structure['groups'])} groups, "
        f"{len(joint_structure['inter_group_connections'])} inter-group links"
        
    )
    inter_group_connections = joint_structure["inter_group_connections"]

    tb_writer = prepare_output_and_logger(output_dir)
    
    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    log.info(f"Training on {len(dataset_loader)} scenes")


    for scene_id, scene_data in dataset_loader:
        
        pose_3d, pose_3d_gt, poses_2d, cameras, scene_name = scene_data 
        pose_3d_gt = np.asarray(pose_3d_gt, dtype=np.float32)
        pose_3d_gt = torch.tensor(pose_3d_gt, dtype=torch.float32, device="cuda")

        if training.std_dev_noise > 0.0:
            log.info(f"Adding Gaussian noise with std. dev. {training.std_dev_noise} to 3D initial pose")
            rng = np.random.default_rng(seed=0)  # reproducible
            noise = rng.normal(loc=0.0, scale=training.std_dev_noise, size=pose_3d.shape)
            pose_3d = pose_3d + noise

        first_iter = 0
        gaussians = GaussianModel(model.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, model, gaussians, pose_3d, cameras, scene_name, output_dir)
        gaussians.training_setup(opt)

        covariance_3d = unpack_covariance(gaussians.get_covariance())
        heatmaps_cameras = generate_heatmaps(gaussians, poses_2d, scene.getTrainCameras(), covariance_3d, training.dropout, dataset.data_root, dataset.nviews)
       
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
        cam_idx_counter = 0

        # to save gt heatmaps
        if debug.save_images:
            save_heatmaps(len(viewpoint_stack), heatmaps_cameras, output_dir, name="heatmap")

        accumulated_loss_total = 0.0

        first_iter += 1  

        grads = []
        accumulated_grads = torch.zeros((len(viewpoint_stack), gaussians.get_xyz.shape[0], gaussians.get_xyz.shape[1]), device="cuda")

        # to compute errors
        errors_all = []
        errors_rel_all = []

        # early stopping
        stop = False

        for iteration in range(first_iter, opt.iterations + 1):

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            idx = viewpoint_indices[cam_idx_counter % len(viewpoint_stack)]
            viewpoint_cam = viewpoint_stack[idx]
            cam_idx_counter += 1

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=model.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if debug.save_images and iteration==1:
                save_images(scene.getTrainCameras(), gaussians, pipe, model, output_dir, name="render_1")
            
            # Loss
            c = viewpoint_cam.uid
            gt_heatmaps = heatmaps_cameras[str(c)]
            l2_loss, error = opt_criterion(image, gt_heatmaps, poses_2d[c, :, :2],training.lambda_loss_function, reduction="mean")
            loss_consistency = consistency_criterion(gaussians.get_xyz, dataset.data_root, reduction="mean") * training.lambda_consistency

            
  
            limb_group_loss = limb_3d_consistency_loss_from_group(gaussians.get_xyz, inter_group_connections)* training.lambda_consistency
            leaf_loss = leaf_anchor_error_constraint(error, inter_group_connections)
            
            loss =  l2_loss+ limb_group_loss  + leaf_loss * 0.05  
            # loss = l2_loss + loss_consistency 
            if early_stopping(loss.item()):
                stop = True

            accumulated_loss_total += loss.item()

            params = [gaussians.get_xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity]
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

            grads_xyz = grads[0]
            grads_scaling = grads[1]
            grads_rotation = grads[2]
            grads_opacity = grads[3]

            # grad = torch.autograd.grad(loss, gaussians.get_xyz, create_graph=True, retain_graph=True)[0]
            if gaussians.get_xyz.grad is None:
                gaussians.get_xyz.grad = torch.zeros_like(gaussians.get_xyz)
                gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
                gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)
                gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
            
            accumulated_grads[idx, ...] = grads_xyz

            gaussians._scaling.grad = grads_scaling
            gaussians._rotation.grad = grads_rotation
            gaussians._opacity.grad = grads_opacity

            iter_end.record()
            if iteration % training.accumulation_steps == 0 or stop:

                with torch.no_grad():
                    # error computation
                    if "h36m" in dataset.data_root or "occlusion-person" in dataset.data_root:
                        subject, activity, step = scene.scene_name.split("_")
                    elif "panoptic" in dataset.data_root:
                        subject = scene.scene_name.split("_")[0]
                        step = scene.scene_name.split("_")[-1]
                        activity = scene.scene_name.split("_")[1] + "_" + scene.scene_name.split("_")[2]
                    elif "foshan" in dataset.data_root:
                        subject, activity, step = scene.scene_name.split("_")
                    if subject == 'S9' and activity in ['SittingDown 1', 'Waiting 1', 'Greeting']:
                        error = torch.tensor([0.0], device="cuda")
                    else:
                        pred = gaussians.get_xyz.clone()
                        gt = pose_3d_gt
                        error = torch.norm(pred - gt, dim=1)
                        # log.info("Opt - Absolute error: " + str(error))
                        errors_all.append(error)

                    pred_rel = pred - pred[0, ...]
                    gt_rel = gt - gt[0, ...]
                    error_rel = torch.norm(pred_rel - gt_rel, dim=1)
                    errors_rel_all.append(error_rel)

                    torch.cuda.synchronize()
                    training_report(
                        tb_writer, iteration,
                        accumulated_loss_total / training.accumulation_steps,  # averaged loss
                        iter_start.elapsed_time(iter_end),
                        scene, error, error_rel
                    )

                gradients = accumulated_grads
                gradients = gradients.to(gaussians.get_xyz.dtype)
                gradients = gradients.mean(dim=0)
                gaussians.get_xyz.grad = gradients

                with torch.no_grad():
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            # Reset accumulated losses
            accumulated_loss_total = 0.0

            if iteration in debug.save_iterations or stop:
                print(f"Saving iteration {iteration} for scene {scene_name}")
                scene.save_h36m(iteration, scene_name)

            if stop:
                log.info(f"Stopping training for scene {scene_name} at iteration {iteration}")
                break

        # to render on all cameras and save images
        if debug.save_images:
            save_images(scene.getTrainCameras(), gaussians, pipe, model, output_dir, name="render")

        log.info("Absolute error: " + str(error))
        log.info("Relative error: " + str(error_rel))
        log.info("Mean absolute error: " + str(error.mean()))
        log.info("Mean relative error: " + str(error_rel.mean()))

    print("Training completed.")
def validate_limb_anchors_with_body(inter_group_connections, hand_anchor, leg_anchor):
    """
    验证四肢的锚点是否与身体的锚点一致
    inter_group_connections: 来自 infer_joint_structure_from_trajectories 的输出，包含关节连接信息
    hand_anchor: 手部锚点
    leg_anchor: 腿部锚点
    """
    # 获取四肢锚点和身体锚点的连接信息
    body_anchor = inter_group_connections["body"]["body_anchor"]

    # 检查手部和腿部锚点是否与身体的锚点一致
    hand_is_correct = (inter_group_connections["left_arm"]["body_anchor"] == hand_anchor or 
                       inter_group_connections["right_arm"]["body_anchor"] == hand_anchor)
    leg_is_correct = (inter_group_connections["left_leg"]["body_anchor"] == leg_anchor or
                      inter_group_connections["right_leg"]["body_anchor"] == leg_anchor)

    # 输出检查结果
    logging.info(f"Body anchor: {body_anchor}")
    logging.info(f"Hand anchor matches body: {hand_is_correct}")
    logging.info(f"Leg anchor matches body: {leg_is_correct}")

    # 输出最终结果
    if hand_is_correct and leg_is_correct:
        logging.info("Both hand and leg anchors are correctly connected to the body anchor.")
    else:
        logging.error("Mismatch detected: one or more limb anchors are not correctly connected to the body anchor.")
    
    return hand_is_correct, leg_is_correct
def compare_limb_order(pred_edges, gt_edges, limb_name, correct_order=True):
    """
    比较预测的四肢连接顺序与真实连接顺序是否一致
    pred_edges: 预测的边
    gt_edges: 真实的边
    limb_name: 四肢名称
    correct_order: 是否检查顺序是否完全一致
    """
    # 从四肢的连接边中提取
    pred_limb_edges = [edge for edge in pred_edges if limb_name in edge]
    gt_limb_edges = [edge for edge in gt_edges if limb_name in edge]

    if correct_order:
        # 完全匹配顺序的情况下，比较每个连接的顺序是否完全一致
        return pred_limb_edges == gt_limb_edges
    else:
        # 只要求顺序不一致的情况下匹配
        return set(pred_limb_edges) == set(gt_limb_edges)
def compare_topology(pred, gt_groups, traj, body_anchor, hand_anchor, leg_anchor):

    def normalize_edges(edges):
        return set(tuple(sorted(e)) for e in edges)

    # -------------------------
    # edges
    # -------------------------
    pred_edges = []
    for gname, gdata in pred.items():
        if isinstance(gdata, dict) and "edges" in gdata:
            pred_edges += gdata["edges"]

    gt_edges = []
    for gname, edges in gt_groups.items():
        gt_edges += edges

    pred_edges = normalize_edges(pred_edges)
    gt_edges = normalize_edges(gt_edges)

    edge_ok = all(edge in gt_edges for edge in pred_edges)

    # -------------------------
    # anchor
    # -------------------------
    hand_ok, leg_ok = validate_limb_anchors_with_body(
        pred,
        hand_anchor,
        leg_anchor
    )

    # -------------------------
    # limb order accuracy
    # -------------------------
    limb_total = 0
    limb_correct = 0

    for g in ["left_arm", "right_arm", "left_leg", "right_leg"]:
        if g in pred and g in gt_groups:
            limb_total += 1

            pred_set = set(tuple(sorted(e)) for e in pred[g]["edges"])
            gt_set = set(tuple(sorted(e)) for e in gt_groups[g])

            if pred_set == gt_set:
                limb_correct += 1

    limb_acc = limb_correct / max(limb_total, 1)

    # -------------------------
    # body accuracy
    # -------------------------
    pred_body = set(tuple(sorted(e)) for e in pred["body"]["edges"])
    gt_body = set(tuple(sorted(e)) for e in gt_groups["body"])

    body_acc = 1.0 if pred_body == gt_body else 0.0

    return {
        "edge_ok": edge_ok,
        "hand_ok": hand_ok,
        "leg_ok": leg_ok,
        "limb_acc": limb_acc,
        "body_acc": body_acc
    }
def leaf_anchor_error_constraint(error, inter_group_connections):
    """
    抑制 distal joint 误差爆炸

    error: (J,2) or (J,3)
    inter_group_connections: 包含 edges + limb_anchor
    """

    device = error.device
    losses = []

    for gname, group in inter_group_connections.items():

        if gname == "body":
            continue

        edges = group["edges"]
        anchor = group["limb_anchor"]

        if len(edges) == 0:
            continue

        # 1️⃣ 找叶子节点（度为1且不是 anchor）
        degree = {}
        for i, j in edges:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1

        leaf_nodes = [
            idx for idx, deg in degree.items()
            if deg == 1 and idx != anchor
        ]

        if len(leaf_nodes) == 0:
            continue

        anchor_error = torch.norm(error[anchor])

        for leaf in leaf_nodes:
            leaf_error = torch.norm(error[leaf])

            # 🔥 关键：只惩罚超过 anchor 的部分
            losses.append(
                torch.relu(leaf_error - anchor_error)
            )

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()
    
def limb_3d_consistency_loss_from_group(xyz, inter_group_connections):
    """
    xyz: (J, 3) 关节坐标
    inter_group_connections: dict, 每条 limb 信息
        {
            "left_arm": {"edges": [(i,j), ...], "limb_anchor": k, "body_anchor": b},
            ...
        }
    return: scalar loss
    核心：
    - 组内：保持 limb 内部边长度一致
    - 组间：左右 limb 平均长度和 anchor 距离差
    - 保持 torch.norm 形式，不归一化、不 detach
    """
    device = xyz.device
    limb_stats = {}  # 保存每条 limb 的 mean_len 和 anchor_dist

    # -------------------------
    # 1️⃣ 组内：计算每条 limb 的平均长度和锚点距离
    # -------------------------
    loss_intra = []

    for gname, group in inter_group_connections.items():
        if gname == "body":
            continue  # body 不参与 limb 内部

        edges = group["edges"]
        limb_anchor = group["limb_anchor"]
        body_anchor = group["body_anchor"]

        if len(edges) == 0:
            continue

        # 组内边长度一致性
        lens = torch.stack([torch.norm(xyz[i]-xyz[j]) for i,j in edges])
        if len(lens) >= 2:
            loss_intra.append(torch.norm(lens - lens.mean()))
        mean_len = lens.mean()

        # limb_anchor → body_anchor 距离（组间约束）
        anchor_dist = torch.norm(xyz[limb_anchor] - xyz[body_anchor])

        # 保存统计量给左右对称性用
        limb_stats[gname] = {
            "mean_len": mean_len,
            "anchor_dist": anchor_dist,
            "body_anchor": body_anchor
        }

    # -------------------------
    # 2️⃣ 左右 limb 对称性（绝对差）
    # -------------------------
    loss_sym = []

    def add_sym_pair(left, right):
        if left in limb_stats and right in limb_stats:
            L = limb_stats[left]
            R = limb_stats[right]

            # 左右 limb 平均长度差
            loss_sym.append(torch.norm(L["mean_len"] - R["mean_len"]))

            # 左右 limb → body_anchor 距离差
            if L["body_anchor"] == R["body_anchor"]:
                loss_sym.append(torch.norm(L["anchor_dist"] - R["anchor_dist"]))

    add_sym_pair("left_arm", "right_arm")
    add_sym_pair("left_leg", "right_leg")

    # -------------------------
    # 3️⃣ 汇总
    # -------------------------
    total_loss = []
    total_loss.extend(loss_intra)
    total_loss.extend(loss_sym)

    if len(total_loss) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(total_loss).mean()

def prepare_output_and_logger(output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_dir + "/tb")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, loss, elapsed, scene : Scene, error, rel_error):
    torch.cuda.synchronize()
    if "h36m" in scene.scene_type or "foshan"in scene.scene_type:
        subject, activity, step = scene.scene_name.split("_")
    elif "panoptic" in scene.scene_type:
        subject = scene.scene_name.split("_")[0]
        step = scene.scene_name.split("_")[-1]
        activity = scene.scene_name.split("_")[1] + "_" + scene.scene_name.split("_")[2]
    tb_string = f"Subject_{subject}_Activity_{activity}/Step_{step}"
    
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar(tb_string + "/absolute_error", error.mean(), iteration)
        tb_writer.add_scalar(tb_string + "/relative_error", rel_error.mean(), iteration)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def save_images(train_cameras, gaussians, pipe, model, output_dir, name="image"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    render = render_functions[pipe.rendering]
    
    for i_camera in range(len(train_cameras)):
        viewpoint_cam = train_cameras[i_camera]
        render_pkg = render(viewpoint_cam, gaussians, pipe, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), use_trained_exp=model.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        im = torch.sum(image, dim=0)
        # ===== 小修正，防止佛山武术集出现 NaN/Inf =====
        im_min = torch.min(im)
        im_max = torch.max(im)
        denom = im_max - im_min
        if denom == 0:  # 全零或全同值
            im_norm = torch.zeros_like(im)
        else:
            im_norm = (im - im_min) / denom
        im_uint8 = (im_norm * 255).clamp(0, 255).to(torch.uint8)
        im_pil = Image.fromarray(im_uint8.detach().cpu().numpy())
        im_pil.save(f"{output_dir}/images/{name}_{i_camera}.png")

def save_heatmaps(nviews, heatmaps_cameras, output_dir, name="heatmap"):

    os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)

    for i_camera in range(nviews):
        heatmap = heatmaps_cameras[str(i_camera)]
        # logging.info(f"{torch.std(heatmap[0])}")
        # logging.info(f"single joint >0 ratio: {(single > 1e-6).float().mean().item()}")
        # logging.info(f"init min/max: {heatmap.min().item()}, {heatmap.max().item()}")
        # logging.info(f"very small ratio: {(heatmap < 1e-3).float().mean().item()}")
        # logging.info(f"{(heatmap[0] > 0.1).float().mean()}")
        # heatmap = heatmap ** 6
        # im = heatmap[0]
        # logging.info(f"{(im > 0.1).float().mean()}")
        im = torch.sum(heatmap, dim=0)
        # logging.info(f"{(im > 0.1).float().mean()}")
        
        # logging.info(f"zero ratio: {(im < 1e-6).float().mean().item()}")
        # logging.info(f"before norm: {im.min().item()}, {im.max().item()}")
        im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))
        im = (im * 255).detach().cpu().numpy().astype(np.uint8)
        im = Image.fromarray(im)
        im.save(f"{output_dir}/heatmaps/{name}_{i_camera}.png")
def collect_aligned_joint_trajectories(
    dataset,
    dataset_loader,
    model,
    opt,
    output_dir,
    num_frames
):
    traj = []

    for i, (scene_id, scene_data) in enumerate(dataset_loader):
        if i >= num_frames:
            break

        pose_3d, pose_3d_gt, poses_2d, cameras, scene_name = scene_data

        gaussians = GaussianModel(model.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, model, gaussians, pose_3d, cameras, scene_name, output_dir)
        gaussians.training_setup(opt)

        joints = gaussians.get_xyz.detach().cpu()
        traj.append(joints)


    return np.stack(traj, axis=0)  # (F, J, 3)  
# 原版    
def stage4_split_head_spine(
    traj,
    body_joints,
    alpha=1.0,
    beta=1.0
):
    import numpy as np
    import logging
    from itertools import combinations

    T = traj.shape[0]
    logging.info(f"[BODY] body_joints = {body_joints}")

    # ============================================================
    # 1. 刚体分解（你原来的逻辑，完全保留）
    # ============================================================
    edge_std = {}
    for i, j in combinations(body_joints, 2):
        d = np.linalg.norm(traj[:, i] - traj[:, j], axis=1)
        std = np.std(d)
        edge_std[(i, j)] = std
        edge_std[(j, i)] = std

    std_vals = np.array(list(set(edge_std.values())))
    rigidity_th = np.percentile(std_vals, 25)

    parent = {j: j for j in body_joints}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j), std in edge_std.items():
        if i < j and std <= rigidity_th:
            union(i, j)

    components = {}
    for j in body_joints:
        r = find(j)
        components.setdefault(r, []).append(j)

    rigid_components = list(components.values())
    logging.info(f"[BODY] rigid_components = {rigid_components}")

    # ============================================================
    # 2. 选“头部候选刚体”（最稳定的一坨）
    # ============================================================
    def stable_edge_count(comp):
        return sum(
            edge_std[(i, j)] <= rigidity_th
            for i, j in combinations(comp, 2)
        )

    head_candidate_comp = max(rigid_components, key=stable_edge_count)
    logging.info(f"[BODY] head_candidate_comp = {head_candidate_comp}")

    head_cluster = head_candidate_comp
    spine_nodes = [j for j in body_joints if j not in head_cluster]


    logging.info(f"[BODY] head_cluster = {head_cluster}")
    logging.info(f"[BODY] spine_nodes = {spine_nodes}")

    return head_cluster, spine_nodes
def build_head_tree_rigid_gs_arc(traj, head_cluster, alpha=1.0, beta=1.0):
    """
    基于惯性残差 + 角速度一致性选择头部根节点，再按夹角分层构建树
    1. 根节点选择：惯性残差 + 角速度一致性最小
    2. 其他节点按相对根节点夹角分层
    3. 构建跨层树（根节点最多连2个）
    """
    import numpy as np
    import logging
    if len(head_cluster) == 1:
        head_root = head_cluster[0]
        logging.info(f"[HEAD] single-node head, root = {head_root}")
        return head_root, []

    T = traj.shape[0]
    head_cluster = list(head_cluster)

    # -----------------------------
    # Step 1: 计算每个关节的“力代理分数”（jerk）
    # -----------------------------
    # 局部旋转稳定性：节点位置相对头部中心的方差
    head_mean = traj[:, head_cluster, :].mean(axis=1, keepdims=True)  # (T,1,3)
    rotation_stability = {}
    jerk_consistency = {}
    
    # 三阶差分 jerk
    jerk = traj[3:] - 3*traj[2:-1] + 3*traj[1:-2] - traj[:-3]
    
    for j in head_cluster:
        # 局部旋转偏差
        local_disp = traj[:, j, :] - head_mean[:, 0, :]
        rotation_stability[j] = np.var(np.linalg.norm(local_disp, axis=1))

        # jerk 与其他头部节点差异
        diffs = []
        for k in head_cluster:
            if k == j:
                continue
            diff = jerk[:, j, :] - jerk[:, k, :]
            diffs.append(np.linalg.norm(diff, axis=1))
        diffs = np.stack(diffs, axis=1)
        jerk_consistency[j] = np.mean(np.var(diffs, axis=0))

    # -----------------------------
    # Step 2: 根节点选择（力分数最小）
    # -----------------------------
    # 综合评分
    alpha, beta = 0.9, 0.1
    scores = {j: alpha*rotation_stability[j] + beta*jerk_consistency[j] for j in head_cluster}
    
    # 最小分数为 root
    head_root = min(scores, key=lambda k: scores[k])
    logging.info(f"[HEAD_ROOT] candidate scores: {scores}")
    logging.info(f"[HEAD] root selected = {head_root}")
    #--------------------
    # Step 3: 计算剩余节点相对根节点的平均夹角（保留原逻辑）
    # -----------------------------
    remaining = set(head_cluster)
    remaining.remove(head_root)
    root_pos = traj[:, head_root, :].mean(axis=0)

    rel_vecs = {}
    angles_to_root = {}
    for j in remaining:
        rel_vec = traj[:, j, :].mean(axis=0) - root_pos
        if np.linalg.norm(rel_vec) > 1e-8:
            rel_vec /= np.linalg.norm(rel_vec)
        rel_vecs[j] = rel_vec

        # 逐帧夹角
        dot_products = []
        for t in range(T):
            v1 = traj[t, j, :] - traj[t, head_root, :]
            if np.linalg.norm(v1) > 1e-8:
                v1 /= np.linalg.norm(v1)
                dot_products.append(np.dot(v1, rel_vec))
        mean_dot = np.mean(dot_products) if dot_products else 1.0
        angle = np.arccos(np.clip(mean_dot, -1, 1))
        angles_to_root[j] = angle
        logging.info(f"[HEAD_ANGLE] joint={j} | angle_to_root={angle:.4f} rad")

    # -----------------------------
    # Step 4: 按夹角分层
    # -----------------------------
    sorted_joints = sorted(angles_to_root.items(), key=lambda x: x[1])
    if len(sorted_joints) >= 2:
        first_layer = [sorted_joints[0][0], sorted_joints[1][0]]
        second_layer = list(remaining - set(first_layer))
    else:
        first_layer = [sorted_joints[0][0]]
        second_layer = list(remaining - set(first_layer))

    head_layers = [[head_root], first_layer, second_layer]
    logging.info(f"[HEAD_LAYERS] {head_layers}")

    # -----------------------------
    # Step 5: 构建跨层树（严格邻层连接）
    # 根节点只能连第二层
    # 第二层节点每个最多只能有一个第三层子节点
    # -----------------------------
    head_edges = []
    
    root = head_layers[0][0]
    second_layer = head_layers[1]
    third_layer = head_layers[2]
    
    # -----------------------------
    # 1. 根节点连接第二层节点（按平均距离最短顺序）
    # -----------------------------
    for j in second_layer:
        d = np.mean(np.linalg.norm(traj[:, j] - traj[:, root], axis=1))
        head_edges.append((root, j))
        logging.info(f"[HEAD_TREE] {root} -> {j} | score={d:.4f}")
    
    # -----------------------------
    # 2. 第二层节点连接第三层节点（每个最多1个）
    # -----------------------------
    assigned_third = set()  # 已经分配的第三层节点
    
    for p in second_layer:
        # 找未分配的第三层节点中最近的一个
        candidates = [j for j in third_layer if j not in assigned_third]
        if not candidates:
            continue
    
        best_j, best_d = None, np.inf
        for j in candidates:
            d = np.mean(np.linalg.norm(traj[:, j] - traj[:, p], axis=1))
            if d < best_d:
                best_d, best_j = d, j
    
        if best_j is not None:
            head_edges.append((p, best_j))
            assigned_third.add(best_j)
            logging.info(f"[HEAD_TREE] {p} -> {best_j} | score={best_d:.4f}")

    return head_root, head_edges
def build_limb_chain_with_terminal_rule(
    joint_ids,
    anchor,
    traj,
    mark,   # "CMU" or "H36M"
):
    """
    输入:
        joint_ids: limb 的 3 个关节
        anchor:
            - CMU: 身体连接点
            - H36M: 肢体中点
        traj: (F, J, 3)

    输出:
        [(root, mid), (mid, end)]
    """
    import numpy as np
    import logging

    assert len(joint_ids) == 3

    mean_pos = traj.mean(axis=0)
    motion_energy = np.linalg.norm(
        traj - mean_pos[None, :, :], axis=2
    ).mean(axis=0)

    others = [j for j in joint_ids if j != anchor]
    j1, j2 = others

    me_anchor = motion_energy[anchor]
    me1 = motion_energy[j1]
    me2 = motion_energy[j2]

    # -------------------------------------------------
    # CMU：anchor 已是 root
    # -------------------------------------------------
    if mark == "CMU" :
        # 末端 = motion energy 最大
        if me1 > me2:
            mid, end = j2, j1
        else:
            mid, end = j1, j2

        root = anchor

    # -------------------------------------------------
    # H36M：anchor 实际是 mid
    # -------------------------------------------------
    elif mark == "H36M" or mark == "FOSHAN":
        mid = anchor
        # 在剩余两个中：
        # 更稳定的 → root（靠身体）
        # 更不稳定的 → end
        if me1 < me2:
            root, end = j1, j2
        else:
            root, end = j2, j1

    else:
        raise ValueError(f"Unknown mark: {mark}")

    logging.info(
        f"[LIMB ORDER][{mark}] "
        f"root={root}, mid={mid}, end={end} | "
        f"me: root={motion_energy[root]:.2f}, "
        f"mid={motion_energy[mid]:.2f}, "
        f"end={motion_energy[end]:.2f}"
    )

    return [(root, mid), (mid, end)]

def fix_leg_chain_direction(limb_chain, body_anchor, traj):
    """
    输入:
        limb_chain: [(a,b),(b,c)] 无向腿链
        body_anchor: pelvis
        traj: (F,J,3)

    输出:
        new_limb_chain: [(hip,knee),(knee,foot)]
        【只返回 edges，绝不返回 int】
    """

    import numpy as np
    from collections import Counter

    # --------------------------------------------------
    # 0. 拿到三个唯一关节
    # --------------------------------------------------
    joints = []
    for u, v in limb_chain:
        joints.append(u)
        joints.append(v)
    joints = list(set(joints))
    assert len(joints) == 3, f"leg must have 3 joints, got {joints}"

    # --------------------------------------------------
    # 1. 找 knee（出现 2 次的）
    # --------------------------------------------------
    cnt = Counter()
    for u, v in limb_chain:
        cnt[u] += 1
        cnt[v] += 1
    knee = [j for j, c in cnt.items() if c == 2][0]
    ends = [j for j in joints if j != knee]  # 两个端点

    # --------------------------------------------------
    # 2. 用三角形面积变化稳定性判断 hip/foot
    # --------------------------------------------------
    def triangle_area_variance(a, b, c):
        # a,b,c = joint indices
        F = traj.shape[0]
        areas = []
        for f in range(F):
            p0 = traj[f, a]
            p1 = traj[f, b]
            p2 = traj[f, c]
            # 向量叉积法计算面积
            vec1 = p1 - p0
            vec2 = p2 - p0
            area = 0.5 * np.linalg.norm(np.cross(vec1, vec2))
            areas.append(area)
        return np.var(areas)

    # 上半段三角形：body_anchor - end - knee
    var0 = triangle_area_variance(body_anchor, ends[0], knee)
    var1 = triangle_area_variance(body_anchor, ends[1], knee)

    # 越稳定（方差小） → hip
    if var0 < var1:
        hip, foot = ends[0], ends[1]
    else:
        hip, foot = ends[1], ends[0]

    # --------------------------------------------------
    # 3. 构造有向链
    # --------------------------------------------------
    edges = [(hip, knee), (knee, foot)]

    import logging
    logging.info(
        f"[LEG FIX AREA] triangle vars: {ends[0]}:{var0:.4e}, {ends[1]}:{var1:.4e}"
    )
    logging.info(
        f"[LEG STRUCT AREA] hip={hip}, knee={knee}, foot={foot}, edges={edges}"
    )

    return edges
def build_hypergraph_for_limb_connection(spine_order, limb_chains, traj, mark="CMU"):
    """
    构建 limb ↔ body 超图（改进版）

    核心策略：
    1. 腿 anchor 固定为 spine_order[0]（默认 pelvis）
    2. 腿 anchor 不参与其他 limb 的 anchor 竞争
    3. 其他 limb 在剩余 spine 上做 energy-based 选择
    4. 保留 H36M 腿方向校正
    """

    import numpy as np
    import logging

    F, J, _ = traj.shape
    logging.info("====== BUILD HYPERGRAPH (LIMB ↔ BODY, NEW) ======")

    # ------------------------
    # 能量函数：相对稳定性
    # ------------------------
    def relative_stability_energy(body_j, limb_j):
        rel = traj[:, limb_j] - traj[:, body_j]
        return np.var(np.linalg.norm(rel, axis=1))

    # ------------------------
    # 腿方向校正（仅 H36M）
    # ------------------------
    def enforce_leg_chain_direction(limb_chain, body_anchor):
        (u0, u1), (_, u2) = limb_chain

        v01 = traj[:, u1] - traj[:, u0]
        v12 = traj[:, u2] - traj[:, u1]

        t01 = traj[:, u1] - traj[:, body_anchor]
        t12 = traj[:, u2] - traj[:, body_anchor]

        dot1 = np.sum(v01 * t01, axis=1)
        dot2 = np.sum(v12 * t12, axis=1)

        if (np.mean(dot1 < 0) > 0.5) or (np.mean(dot2 < 0) > 0.5):
            logging.info("    >>> LEG CHAIN REVERSED")
            return [(u2, u1), (u1, u0)]

        return limb_chain

    # ------------------------
    # Step 1: 固定腿 anchor（pelvis）
    # ------------------------
    if len(spine_order) == 0:
        raise ValueError("spine_order 为空，无法构建超图")

    leg_anchor = spine_order[0]
    excluded_anchors = {leg_anchor}

    logging.info(f"[FIXED LEG ANCHOR] pelvis={leg_anchor}")

    hyperedges = {}

    # ------------------------
    # Step 2: 遍历所有 limb
    # ------------------------
    for gname, limb_chain in limb_chains.items():

        logging.info(f"\n[LIMB] {gname}")
        logging.info(f"  limb_chain: {limb_chain}")

        limb_root = limb_chain[0][0]

        # ------------------------
        # 腿：直接绑定 pelvis
        # ------------------------
        if "leg" in gname:

            body_anchor = leg_anchor

            if mark.upper() == "H36M":
                limb_chain = enforce_leg_chain_direction(limb_chain, body_anchor)

            limb_root = limb_chain[0][0]

            logging.info(f"  >>> LEG FIXED anchor={body_anchor}, root={limb_root}")

            hyperedges[gname] = {
                "body_anchor": body_anchor,
                "limb_root": limb_root,
                "edges": limb_chain,
            }

            continue

        # ------------------------
        # 其他 limb：排除 pelvis 后做 energy 搜索
        # ------------------------
        candidates = [b for b in spine_order if b not in excluded_anchors]

        best_E = float("inf")
        best_body = None

        for b in candidates:
            E = relative_stability_energy(b, limb_root)

            if E < best_E:
                best_E = E
                best_body = b

        logging.info(f"  >>> SELECTED anchor={best_body}, root={limb_root}, E={best_E:.6f}")

        hyperedges[gname] = {
            "body_anchor": best_body,
            "limb_root": limb_root,
            "edges": limb_chain,
        }

    logging.info("====== HYPERGRAPH DONE ======\n")

    return hyperedges    
def find_chain_spine_order(traj, body_joints, limb_chain):
    import numpy as np
    import logging

    logging.info("====== FIND SPINE ORDER (GEOMETRY VERSION) ======")

    # ============================================================
    # 1️⃣ score（jerk + rot）
    # ============================================================
    jerk = traj[3:] - 3*traj[2:-1] + 3*traj[1:-2] - traj[:-3]
    scores = {}

    mean_pos = traj[:, body_joints, :].mean(axis=1, keepdims=True)

    for j in body_joints:
        local_disp = traj[:, j] - mean_pos[:, 0]
        rot_stab = np.var(np.linalg.norm(local_disp, axis=1))

        diffs = [
            np.linalg.norm(jerk[:, j] - jerk[:, k], axis=1)
            for k in body_joints if k != j
        ]
        jerk_dev = 0.0 if len(diffs) == 0 else np.mean(
            np.var(np.stack(diffs, axis=1), axis=0)
        )

        scores[j] = rot_stab + jerk_dev
        logging.info(f"{j}, score={scores[j]}")

    # ============================================================
    # 2️⃣ 找关键点（end + center）
    # ============================================================
    sorted_joints = sorted(scores, key=lambda j: scores[j], reverse=True)

    end1, end2 = sorted_joints[:2]   # 两端（最大两个）

    if len(body_joints) % 2 == 1:
        center = min(body_joints, key=lambda j: scores[j])
        logging.info(f"[CENTER] {center}")
    else:
        center = None

    # ============================================================
    # 3️⃣ 构建 spine 主轴（end1 → end2）
    # ============================================================
    axis = np.mean(traj[:, end2] - traj[:, end1], axis=0)
    axis /= (np.linalg.norm(axis) + 1e-8)

    # ============================================================
    # 4️⃣ 投影排序（🔥核心）
    # ============================================================
    proj = {}
    for j in body_joints:
        v = traj[:, j] - traj[:, end1]
        proj[j] = np.mean(np.dot(v, axis))

    spine = sorted(body_joints, key=lambda j: proj[j])

    logging.info(f"[CHAIN] geometric spine = {spine}")

    # ============================================================
    # 5️⃣ limb 决定方向（pelvis → head）
    # ============================================================
    leg_anchors = []
    arm_anchors = []

    for name, chain in limb_chain.items():
        root = chain[0][0]
        if "leg" in name:
            leg_anchors.append(root)
        elif "arm" in name:
            arm_anchors.append(root)

    # 如果没有 limb 信息，直接返回
    if len(leg_anchors) == 0 or len(arm_anchors) == 0:
        return spine

    def direction_score(spine):
        s0, sN = spine[0], spine[-1]
        axis = np.mean(traj[:, sN] - traj[:, s0], axis=0)

        score = 0.0

        # 腿应该靠 pelvis（起点）
        for leg in leg_anchors:
            v = np.mean(traj[:, s0] - traj[:, leg], axis=0)
            score += np.dot(v, axis)

        # 手应该靠 head（终点）
        for arm in arm_anchors:
            v = np.mean(traj[:, arm] - traj[:, sN], axis=0)
            score += np.dot(v, axis)

        return score

    if direction_score(spine[::-1]) > direction_score(spine):
        spine = spine[::-1]

    logging.info(f"[FINAL SPINE] {spine}")

    return spine
def infer_joint_structure_from_trajectories(
    aligned_traj,      # (F, J, 3)
    dataset_root,
):
    import numpy as np
    import logging
    from sklearn.neighbors import NearestNeighbors

    # ------------------------------------------------------------
    # Stage 0: joint groups
    # ------------------------------------------------------------
    
    if "h36m" in dataset_root.lower():
        groups = H36M_JOINT_GROUPS
        mark = "H36M"
    elif "panoptic" in dataset_root.lower():
        groups = PANOPTIC_JOINT_GROUPS
        mark = "CMU"
    else:
        groups = FOSHAN_JOINT_GROUPS
        mark = "FOSHAN"
    F, J, _ = aligned_traj.shape
    traj = aligned_traj
    mean_pos = traj.mean(axis=0)

    logging.info(f"Using joint groups: {list(groups.keys())}")

    # ------------------------------------------------------------
    # 提前定义 body_joints（⚠️关键修复）
    # ------------------------------------------------------------
    body_joints = [j for j in groups["body"] if j < J]
    
    # ------------------------------------------------------------
    # Stage 1: 发现关节组连接关系
    # ------------------------------------------------------------
    limb_anchors = {}
    body_edges = []
    intra_group_connections = {}
    inter_group_connections = {}
    inter_group_edges = []
    #找参考点
    for gname in ["left_arm", "right_arm", "left_leg", "right_leg"]:
        joint_ids = [j for j in groups[gname] if j < J]
        if not joint_ids:
            continue
    # # limb internal center
        center = mean_pos[joint_ids].mean(axis=0)
    
        # select limb root by minimal distance to limb center
        limb_anchor = min(
            joint_ids,
            key=lambda j: np.linalg.norm(mean_pos[j] - center)
        )
    
        limb_anchors[gname] = limb_anchor
    
        logging.info(
            f"[LIMB-ANCHOR] {gname}: root={limb_anchor}"
        )
    #四肢内部的连接可能
    for gname in ["left_arm", "right_arm", "left_leg", "right_leg"]:
        joint_ids = [j for j in groups[gname] if j < J]
    
        if len(joint_ids) <= 1:
            intra_group_connections[gname] = []
            continue
    
        anchor = limb_anchors[gname]
    
        edges = build_limb_chain_with_terminal_rule(
            joint_ids,
            anchor,
            traj,
            mark
        )
        intra_group_connections[gname] = edges
        logging.info(
            f"[{gname}] anchor={anchor}, intra-group edges={edges}"
        )    
        
    # ------------------------------------------------------------
    # Stage 2: 开始构造身体关节关系
    # ------------------------------------------------------------        
    logging.info("====== STAGE 4: BODY ======")
    # 立体结构需要拆除头部结构
    # 其拆完后进行链子脊椎构建
    if mark=="CMU" :
        # 头脊拆分
        logging.info(f"拆分头身")
        head_cluster, spine_candidates = stage4_split_head_spine(
            traj,
            body_joints,
        )
        # 头部构建
        logging.info(f"头部构建")
        head_root, head_edges = build_head_tree_rigid_gs_arc(
            traj,
            head_cluster
        )
        body_joints = spine_candidates
    # 准备好锚点用        
    limb_chains = {
        k: v for k, v in intra_group_connections.items()
        if k in ["left_arm", "right_arm", "left_leg", "right_leg"]
    }        
    # 进行脊椎链子的推敲 ,连接顺序固定从低到高
    spine_order = find_chain_spine_order(
        traj,
        body_joints,
        limb_chains
    )
    logging.info("====== STAGE 5 ======")
    # ------------------------------------------------------------
    # Stage 5: 四肢内部（稳定性 → 单向链）
    # ------------------------------------------------------------
    links = build_hypergraph_for_limb_connection(
        spine_order,
        limb_chains,
        traj,
        mark
    )
    logging.info(f"links after processing: {links}")    
    # 正常的大骨架链子
    body_edges = [(spine_order[i], spine_order[i+1]) 
                   for i in range(len(spine_order)-1)]    
    #组装身体
    if mark == "CMU":
        spine_top = spine_order[-1]
        head_spine_link = (spine_top,head_root)
        body_edges = body_edges + [head_spine_link] +head_edges

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    # 先把四肢内部边存进 inter_group_connections
    inter_group_connections = intra_group_connections.copy()
    
    # 添加四肢锚点信息


    for gname, limb_data in links.items():
        inter_group_connections[gname] = {
            "edges": limb_data["edges"],        # ✅ build_hypergraph_for_limb_connection 的最终链
            "limb_anchor": limb_data["limb_root"],  # 或 hyperedges[gname]["limb_root"]
            "body_anchor": limb_data["body_anchor"]  # 或 hyperedges[gname]["body_anchor"]
        }
    inter_group_connections["body"] = {
        "edges": body_edges,
        "limb_anchor": 0,# 身体没有 limb_anchor，可以置 None
        "body_anchor": 0,  # 比如盆骨节点作为锚点

    }   
    logging.info(f"[INTER_GROUP_CONNECTIONS] {inter_group_connections}")
    return {
        "groups": groups,
        "parents": [-1] * J,
        "inter_group_connections": inter_group_connections,
    }

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    config = ConfigHandler(cfg)

    output_dir = config.hydra_out
    dataset = cfg.dataset
    train = cfg.training
    debug = cfg.debug
    model = cfg.model
    opt = cfg.optimization
    pipe = cfg.pipeline

    print(output_dir)

    log = logging.getLogger(__name__)

    if train.dropout:
        print("Dropping out some gt joints during training")

    initial_guess_path = os.path.join(dataset.data_root, "initial_guess", dataset.initial_guess)
    poses_2d_path = os.path.join(dataset.data_root, "2d_" + dataset.poses_2d)

    debug.save_iterations.append(opt.iterations)
    dataset_loader = DataLoader(dataset.data_root, initial_guess_path, poses_2d_path,
                                frame_step=dataset.frame_step, start_id=dataset.start_scene_id, 
                                end_id=dataset.end_scene_id, nviews=dataset.nviews)
    

    # Initialize system state (RNG)
    safe_state(train.quiet)
    training(dataset, model, opt, pipe, debug, train, dataset_loader, output_dir, log)

if __name__ == "__main__":
    main()

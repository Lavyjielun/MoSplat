import os
import numpy as np
import torch
import smplx

OPENPOSE_TO_15= [
    1,  # Neck
    2,  # RShoulder
    3,  # RElbow
    4,  # RWrist
    5,  # LShoulder
    6,  # LElbow
    7,  # LWrist
    8,  # MidHip
    9,  # RHip
    10, # RKnee
    11, # RAnkle
    12, # LHip
    13, # LKnee
    14  # LAnkle
]
SMPL_TO_15 = [0,2,5,8,1,4,7,15,16,18,20,17,19,21]
def process_camera(src_cam, dst2d_cam, smpl):

    kp_path = os.path.join(src_cam, "keypoints.npy")
    smpl_path = os.path.join(src_cam, "poses_optimized.npz")

    if not os.path.exists(kp_path):
        return None

    print("processing:", src_cam)

    os.makedirs(dst2d_cam, exist_ok=True)

    # ---------- 2D ----------
    kp = np.load(kp_path)  # (F,25,3)

    poses2d = kp[:, OPENPOSE_TO_15, :2]

    np.savez(
        os.path.join(dst2d_cam, "poses.npz"),
        poses=poses2d
    )

    # ---------- 3D ----------
    if not os.path.exists(smpl_path):
        return None

    data = np.load(smpl_path)

    body_pose = torch.tensor(data["body_pose"]).float()
    global_orient = torch.tensor(data["global_orient"]).float()
    transl = torch.tensor(data["transl"]).float()
    betas = torch.tensor(data["betas"]).float()

    F = body_pose.shape[0]
    betas = betas.unsqueeze(0).repeat(F,1)

    out = smpl(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        betas=betas
    )

    joints = out.joints.detach().cpu().numpy()

    poses3d = joints[:, SMPL_TO_15, :]

    poses3d = poses3d - poses3d[:,0:1,:]
    poses3d = poses3d * 1000

    return poses3d


def walk(src_root, dst2d_root, dst3d_root, smpl):

    for subject in os.listdir(src_root):

        subject_path = os.path.join(src_root, subject)

        for action in os.listdir(subject_path):

            action_path = os.path.join(subject_path, action)

            poses3d_saved = False
            poses3d_data = None

            for cam in os.listdir(action_path):

                cam_path = os.path.join(action_path, cam)

                if not os.path.isdir(cam_path):
                    continue

                dst2d_cam = os.path.join(dst2d_root, subject, action, cam)

                poses3d = process_camera(cam_path, dst2d_cam, smpl)

                # 只保存一次3D
                if poses3d is not None and not poses3d_saved:
                    poses3d_data = poses3d
                    poses3d_saved = True

            # ---------- 保存3D ----------
            if poses3d_saved:

                dst3d_action = os.path.join(dst3d_root, subject, action)
                os.makedirs(dst3d_action, exist_ok=True)

                np.savez(
                    os.path.join(dst3d_action, "poses.npz"),
                    poses=poses3d_data
                )


if __name__ == "__main__":

    src_root = "origin"

    dst2d_root = "2d_gt"
    dst3d_root = "3d_gt"

    smpl_model = smplx.create(
        "smpl_model",
        model_type="smpl",
        gender="neutral"
    )

    walk(src_root, dst2d_root, dst3d_root, smpl_model)
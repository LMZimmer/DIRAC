import glob
import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def refine_displacement(
    fixed,
    moving,
    disp0,
    num_iter=300,
    lr=1e-2,
    lambda_reg=0.5,
    align_corners=True,
):
    """
    Minimal ConvexAdam-style instance optimization.

    fixed, moving : (1,1,H,W,D)
    disp0         : (1,3,H,W,D) in voxel units with channel order (x, y, z)
    returns refined displacement (same shape)
    """

    device = fixed.device
    _, _, H, W, D = fixed.shape

    # Identity grid in normalized coords [-1,1], order expected by grid_sample is (x, y, z).
    xs = torch.linspace(-1, 1, D, device=device)
    ys = torch.linspace(-1, 1, W, device=device)
    zs = torch.linspace(-1, 1, H, device=device)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    grid0 = torch.stack((xx, yy, zz), dim=-1)[None]  # (1,H,W,D,3)

    disp = disp0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([disp], lr=lr)

    if align_corners:
        sx = 2.0 / (D - 1) if D > 1 else 0.0
        sy = 2.0 / (W - 1) if W > 1 else 0.0
        sz = 2.0 / (H - 1) if H > 1 else 0.0
    else:
        sx = 2.0 / D
        sy = 2.0 / W
        sz = 2.0 / H

    for _ in range(num_iter):
        disp_norm = torch.stack(
            (
                disp[:, 0] * sx,
                disp[:, 1] * sy,
                disp[:, 2] * sz,
            ),
            dim=-1,
        )

        grid = grid0 + disp_norm

        warped = F.grid_sample(
            moving,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=align_corners,
        )

        loss_sim = ((fixed - warped) ** 2).mean()

        dx = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
        dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
        dz = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
        loss_reg = dx.pow(2).mean() + dy.pow(2).mean() + dz.pow(2).mean()

        loss = loss_sim + lambda_reg * loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return disp.detach()


def warp_with_voxel_displacement(moving, disp, align_corners=True):
    """Warp moving image (1,1,H,W,D) with voxel-space displacement (1,3,H,W,D)."""
    device = moving.device
    _, _, H, W, D = moving.shape

    xs = torch.linspace(-1, 1, D, device=device)
    ys = torch.linspace(-1, 1, W, device=device)
    zs = torch.linspace(-1, 1, H, device=device)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    grid0 = torch.stack((xx, yy, zz), dim=-1)[None]

    if align_corners:
        sx = 2.0 / (D - 1) if D > 1 else 0.0
        sy = 2.0 / (W - 1) if W > 1 else 0.0
        sz = 2.0 / (H - 1) if H > 1 else 0.0
    else:
        sx = 2.0 / D
        sy = 2.0 / W
        sz = 2.0 / H

    disp_norm = torch.stack(
        (
            disp[:, 0] * sx,
            disp[:, 1] * sy,
            disp[:, 2] * sz,
        ),
        dim=-1,
    )
    grid = grid0 + disp_norm
    return F.grid_sample(moving, grid, mode="bilinear", padding_mode="border", align_corners=align_corners)


def optimize_patient(patient_dir, args, device):
    patient_id = os.path.basename(os.path.normpath(patient_dir))

    fixed_path = os.path.join(patient_dir, "t1c_bet_normalized.nii.gz")
    moving_path = os.path.join(patient_dir, "t1c_bet_normalized_followup.nii.gz")
    disp_path = os.path.join(patient_dir, f"{patient_id}_followup_to_preop_disp_voxel.nii.gz")

    required = [fixed_path, moving_path, disp_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"[SKIP] {patient_id}: missing required files -> {missing}")
        return False

    fixed_nifti = nib.load(fixed_path)
    moving_nifti = nib.load(moving_path)
    disp_nifti = nib.load(disp_path)

    fixed_np = fixed_nifti.get_fdata().astype(np.float32)
    moving_np = moving_nifti.get_fdata().astype(np.float32)
    disp_np = disp_nifti.get_fdata().astype(np.float32)

    if disp_np.shape != (*fixed_np.shape, 3):
        print(
            f"[SKIP] {patient_id}: displacement shape mismatch. "
            f"Expected {(fixed_np.shape[0], fixed_np.shape[1], fixed_np.shape[2], 3)}, got {disp_np.shape}"
        )
        return False

    fixed = torch.from_numpy(fixed_np)[None, None].to(device)
    moving = torch.from_numpy(moving_np)[None, None].to(device)
    disp0 = torch.from_numpy(np.moveaxis(disp_np, -1, 0))[None].to(device)

    with torch.enable_grad():
        refined_disp = refine_displacement(
            fixed=fixed,
            moving=moving,
            disp0=disp0,
            num_iter=args.num_iter,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
            align_corners=True,
        )

    warped = warp_with_voxel_displacement(moving, refined_disp, align_corners=True)

    refined_disp_np = refined_disp[0].detach().cpu().numpy().transpose(1, 2, 3, 0)
    warped_np = warped[0, 0].detach().cpu().numpy()

    disp_out_path = os.path.join(patient_dir, f"{patient_id}_followup_to_preop_disp_voxel_refined.nii.gz")
    warped_out_path = os.path.join(patient_dir, f"{patient_id}_Y_X_refined.nii.gz")

    nib.save(nib.Nifti1Image(refined_disp_np, disp_nifti.affine, disp_nifti.header), disp_out_path)
    nib.save(nib.Nifti1Image(warped_np, fixed_nifti.affine, fixed_nifti.header), warped_out_path)

    print(f"[OK] {patient_id}: saved {os.path.basename(disp_out_path)} and {os.path.basename(warped_out_path)}")
    return True


def main():
    parser = ArgumentParser(description="Run per-patient instance optimization from saved DIRAC displacement fields.")
    parser.add_argument(
        "--datapath",
        type=str,
        default="../Dataset/predict_gbm",
        help="Path containing patient folders (same format as BRATS_infer_DIRAC.py).",
    )
    parser.add_argument("--num_iter", type=int, default=300, help="Number of optimization iterations per patient.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for instance optimization.")
    parser.add_argument("--lambda_reg", type=float, default=0.5, help="Diffusion regularization weight.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")

    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    patient_dirs = sorted([p for p in glob.glob(os.path.join(args.datapath, "*")) if os.path.isdir(p)])
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories found in datapath: {args.datapath}")

    print(f"Found {len(patient_dirs)} patient folders in {args.datapath}")

    n_ok = 0
    for patient_dir in patient_dirs:
        n_ok += int(optimize_patient(patient_dir, args, device))

    print(f"Done. Optimized {n_ok}/{len(patient_dirs)} patients.")


if __name__ == "__main__":
    main()

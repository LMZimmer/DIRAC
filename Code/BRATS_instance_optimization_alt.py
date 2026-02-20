import glob
import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from Functions import save_flow


def make_identity_grid(D, H, W, device, dtype):
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)[None]


def voxel_disp_to_norm(disp, D, H, W):
    dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]
    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)
    sz = 2.0 / max(D - 1, 1)
    return torch.stack((dx * sx, dy * sy, dz * sz), dim=-1)


def smooth3x(disp):
    out = disp
    for _ in range(3):
        out = F.avg_pool3d(out, 3, stride=1, padding=1)
    return out


def optimize_single_direction(
    fixed,
    moving,
    disp_init,
    fixed_mask=None,
    iterations=40,
    lr=0.02,
    lambda_weight=0.6,
    grid_sp=3,
):
    _, _, D, H, W = fixed.shape
    Dg, Hg, Wg = max(D // grid_sp, 2), max(H // grid_sp, 2), max(W // grid_sp, 2)

    fixed_ds = F.interpolate(fixed, size=(Dg, Hg, Wg), mode="trilinear", align_corners=False)
    moving_ds = F.interpolate(moving, size=(Dg, Hg, Wg), mode="trilinear", align_corners=False)

    if fixed_mask is None:
        fixed_mask_ds = torch.zeros_like(fixed_ds)
    else:
        fixed_mask_ds = F.interpolate(fixed_mask, size=(Dg, Hg, Wg), mode="nearest")

    cp_init = F.interpolate(disp_init, size=(Dg, Hg, Wg), mode="trilinear", align_corners=False) / float(grid_sp)
    cp = cp_init.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([cp], lr=lr)
    grid0 = make_identity_grid(Dg, Hg, Wg, fixed.device, fixed.dtype)

    for _ in range(iterations):
        optimizer.zero_grad()

        disp_sample = smooth3x(cp)
        reg_loss = lambda_weight * (
            (disp_sample[:, :, :, :, 1:] - disp_sample[:, :, :, :, :-1]).pow(2).mean()
            + (disp_sample[:, :, :, 1:, :] - disp_sample[:, :, :, :-1, :]).pow(2).mean()
            + (disp_sample[:, :, 1:, :, :] - disp_sample[:, :, :-1, :, :]).pow(2).mean()
        )

        grid_disp = grid0 + voxel_disp_to_norm(disp_sample, Dg, Hg, Wg)
        moving_warped = F.grid_sample(
            moving_ds,
            grid_disp,
            align_corners=False,
            mode="bilinear",
            padding_mode="border",
        )

        sampled_cost = (moving_warped - fixed_ds).pow(2) * 12.0
        sampled_cost = sampled_cost * (1.0 - fixed_mask_ds.to(sampled_cost.dtype))
        loss = sampled_cost.mean()

        (loss + reg_loss).backward()
        optimizer.step()

    fitted_grid = smooth3x(cp).detach()
    disp_hr = F.interpolate(fitted_grid * float(grid_sp), size=(D, H, W), mode="trilinear", align_corners=False)
    return smooth3x(disp_hr)


def dirac_instance_optimization_alt(
    B,
    Fup,
    disp_fb_init,
    disp_bf_init,
    m_fb_fixed=None,
    m_bf_fixed=None,
    iterations=40,
    lr=0.02,
    lambda_weight=0.6,
    grid_sp=3,
):
    if m_fb_fixed is None:
        m_fb_fixed = torch.zeros_like(B)
    if m_bf_fixed is None:
        m_bf_fixed = torch.zeros_like(B)

    disp_fb = optimize_single_direction(
        fixed=B,
        moving=Fup,
        disp_init=disp_fb_init,
        fixed_mask=m_fb_fixed,
        iterations=iterations,
        lr=lr,
        lambda_weight=lambda_weight,
        grid_sp=grid_sp,
    )

    disp_bf = optimize_single_direction(
        fixed=Fup,
        moving=B,
        disp_init=disp_bf_init,
        fixed_mask=m_bf_fixed,
        iterations=iterations,
        lr=lr,
        lambda_weight=lambda_weight,
        grid_sp=grid_sp,
    )

    return disp_fb.detach(), disp_bf.detach(), m_fb_fixed.detach(), m_bf_fixed.detach()


def load_image_for_grid_sample(path, device):
    img = nib.load(path).get_fdata().astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    return img.to(device)


def load_mask_for_grid_sample(path, device):
    mask = nib.load(path).get_fdata().astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)
    mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    return mask.to(device)


def load_dirac_voxel_disp_for_grid_sample(path, device):
    disp = nib.load(path).get_fdata().astype(np.float32)

    comp_axis0 = torch.from_numpy(disp[..., 0]).permute(2, 0, 1)
    comp_axis1 = torch.from_numpy(disp[..., 1]).permute(2, 0, 1)
    comp_axis2 = torch.from_numpy(disp[..., 2]).permute(2, 0, 1)

    disp_grid = torch.stack((comp_axis1, comp_axis0, comp_axis2), dim=0).unsqueeze(0)
    return disp_grid.to(device)


def grid_sample_disp_to_dirac_voxel(disp):
    dx = disp[0, 0].permute(1, 2, 0).cpu().numpy()
    dy = disp[0, 1].permute(1, 2, 0).cpu().numpy()
    dz = disp[0, 2].permute(1, 2, 0).cpu().numpy()
    return np.stack((dy, dx, dz), axis=-1).astype(np.float32)


def run_patient(patient_dir, device, args):
    patient_id = os.path.basename(patient_dir.rstrip("/"))
    preop_path = os.path.join(patient_dir, "t1c_bet_normalized.nii.gz")
    followup_path = os.path.join(patient_dir, "t1c_bet_normalized_followup.nii.gz")

    fwd_candidates = sorted(glob.glob(os.path.join(patient_dir, "*_followup_to_preop_disp_voxel.nii.gz")))
    if len(fwd_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one followup->preop voxel field in {patient_dir}, found: {fwd_candidates}"
        )
    disp_fb_path = fwd_candidates[0]

    if not os.path.exists(preop_path) or not os.path.exists(followup_path) or not os.path.exists(disp_fb_path):
        print(f"[skip] {patient_id}: missing required inputs")
        return

    preop_mask_candidates = sorted(glob.glob(os.path.join(patient_dir, "*_yx_seg.nii.gz")))
    if len(preop_mask_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one preop-space mask (*_yx_seg.nii.gz) in {patient_dir}, found: {preop_mask_candidates}"
        )
    preop_mask_path = preop_mask_candidates[0]

    followup_mask_candidates = sorted(glob.glob(os.path.join(patient_dir, "*_xy_seg.nii.gz")))
    if len(followup_mask_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one followup-space mask (*_xy_seg.nii.gz) in {patient_dir}, found: {followup_mask_candidates}"
        )
    followup_mask_path = followup_mask_candidates[0]

    if not os.path.exists(preop_mask_path) or not os.path.exists(followup_mask_path):
        print(f"[skip] {patient_id}: missing required masks")
        return

    bwd_candidates = sorted(glob.glob(os.path.join(patient_dir, "*_preop_to_followup_disp_voxel.nii.gz")))
    if len(bwd_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one preop->followup voxel field (*_preop_to_followup_disp_voxel.nii.gz) in {patient_dir}, found: {bwd_candidates}"
        )
    disp_bf_path = bwd_candidates[0]

    preop = load_image_for_grid_sample(preop_path, device)
    followup = load_image_for_grid_sample(followup_path, device)
    preop_mask = load_mask_for_grid_sample(preop_mask_path, device)
    followup_mask = load_mask_for_grid_sample(followup_mask_path, device)
    disp_fb = load_dirac_voxel_disp_for_grid_sample(disp_fb_path, device)
    disp_bf = load_dirac_voxel_disp_for_grid_sample(disp_bf_path, device)

    disp_fb_opt, disp_bf_opt, m_fb, m_bf = dirac_instance_optimization_alt(
        B=preop,
        Fup=followup,
        disp_fb_init=disp_fb,
        disp_bf_init=disp_bf,
        m_fb_fixed=preop_mask,
        m_bf_fixed=followup_mask,
        iterations=args.iters,
        lr=args.lr,
        lambda_weight=args.lambda_weight,
        grid_sp=args.grid_sp,
    )

    ref = nib.load(preop_path)
    header, affine = ref.header, ref.affine

    fb_voxel = grid_sample_disp_to_dirac_voxel(disp_fb_opt)
    bf_voxel = grid_sample_disp_to_dirac_voxel(disp_bf_opt)

    save_flow(
        fb_voxel,
        os.path.join(patient_dir, f"{patient_id}_followup_to_preop_disp_voxel_optimized_alt.nii.gz"),
        header=header,
        affine=affine,
    )
    save_flow(
        bf_voxel,
        os.path.join(patient_dir, f"{patient_id}_preop_to_followup_disp_voxel_optimized_alt.nii.gz"),
        header=header,
        affine=affine,
    )

    if args.save_masks:
        save_flow(
            m_fb[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            os.path.join(patient_dir, f"{patient_id}_followup_to_preop_mask_optimized_alt.nii.gz"),
            header=header,
            affine=affine,
        )
        save_flow(
            m_bf[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            os.path.join(patient_dir, f"{patient_id}_preop_to_followup_mask_optimized_alt.nii.gz"),
            header=header,
            affine=affine,
        )

    print(f"[ok] {patient_id}: optimized alt fields saved")


def main():
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="../Dataset/test", help="Path containing patient folders")
    parser.add_argument("--iters", type=int, default=40, help="Adam iterations")
    parser.add_argument("--lr", type=float, default=0.02, help="Adam learning rate")
    parser.add_argument("--lambda_weight", type=float, default=0.6, help="Diffusion regularization weight")
    parser.add_argument("--grid_sp", type=int, default=3, help="Low-resolution control grid spacing")
    parser.add_argument("--save_masks", action="store_true", help="Save optimized mask volumes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run only the first patient folder found in datapath",
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    patient_dirs = sorted([d for d in glob.glob(os.path.join(args.datapath, "*")) if os.path.isdir(d)])
    print(f"Found {len(patient_dirs)} patient folders in {args.datapath}")

    if args.test_run and patient_dirs:
        patient_dirs = patient_dirs[:1]
        print(f"Test run enabled: processing only {os.path.basename(patient_dirs[0])}")

    for patient_dir in patient_dirs:
        run_patient(patient_dir, device, args)

    print("Done.")


if __name__ == "__main__":
    main()


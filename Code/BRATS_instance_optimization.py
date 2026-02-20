import glob
import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from Functions import save_flow


# ---------- Warping utilities ----------

def make_identity_grid(D, H, W, device, dtype):
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)[None]  # (1,D,H,W,3)


def voxel_disp_to_norm(disp, D, H, W):
    dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]
    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)
    sz = 2.0 / max(D - 1, 1)
    return torch.stack((dx * sx, dy * sy, dz * sz), dim=-1)


def warp(img, disp):
    _, _, D, H, W = img.shape
    grid0 = make_identity_grid(D, H, W, img.device, img.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, D, H, W)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def warp_field(field, disp):
    _, _, D, H, W = field.shape
    grid0 = make_identity_grid(D, H, W, field.device, field.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, D, H, W)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


def resize_disp_voxel(disp, size):
    """Resize displacement field and preserve voxel-space magnitude semantics."""
    _, _, D0, H0, W0 = disp.shape
    D1, H1, W1 = size
    resized = F.interpolate(disp, size=size, mode="trilinear", align_corners=True)

    sx = (W1 - 1) / max(W0 - 1, 1)
    sy = (H1 - 1) / max(H0 - 1, 1)
    sz = (D1 - 1) / max(D0 - 1, 1)

    resized[:, 0] *= sx
    resized[:, 1] *= sy
    resized[:, 2] *= sz
    return resized


# ---------- NCC similarity ----------

def ncc_loss(I, J, mask=None, win=3, eps=1e-5):
    pad = win // 2
    filt = torch.ones((1, 1, win, win, win), device=I.device, dtype=I.dtype)

    def conv(x):
        return F.conv3d(x, filt, padding=pad)

    if mask is None:
        mask = torch.ones_like(I)

    mask = mask.to(dtype=I.dtype)

    I2, J2, IJ = I * I, J * J, I * J

    w_sum = conv(mask)
    I_sum = conv(mask * I)
    J_sum = conv(mask * J)
    I2_sum = conv(mask * I2)
    J2_sum = conv(mask * J2)
    IJ_sum = conv(mask * IJ)

    u_I = I_sum / (w_sum + eps)
    u_J = J_sum / (w_sum + eps)

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * w_sum
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * w_sum
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * w_sum

    ncc = cross * cross / (I_var * J_var + eps)
    valid = (w_sum > 0).to(dtype=I.dtype)
    return -(ncc * valid).sum() / (valid.sum() + eps)


# ---------- Regularization ----------

def smoothness(disp):
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    return dx.pow(2).mean() + dy.pow(2).mean() + dz.pow(2).mean()


# ---------- Inverse consistency ----------

def inv_consistency(d_fwd, d_bwd, m_fwd=None, m_bwd=None):
    bwd_warped = warp_field(d_bwd, d_fwd)
    fwd_warped = warp_field(d_fwd, d_bwd)

    err_fwd = d_fwd + bwd_warped
    err_bwd = d_bwd + fwd_warped

    if m_fwd is not None:
        err_fwd = err_fwd * (1 - m_fwd)
    if m_bwd is not None:
        err_bwd = err_bwd * (1 - m_bwd)

    return torch.linalg.vector_norm(err_fwd) + torch.linalg.vector_norm(err_bwd)


# ---------- Paper-accurate instance optimization ----------

def dirac_instance_optimization(
    B,
    Fup,
    disp_fb_init,
    disp_bf_init=None,
    m_fb_fixed=None,
    m_bf_fixed=None,
    lambdas_reg=(0.25, 0.3, 0.3, 0.35, 0.35),
    lambdas_inv=(1.0, 2.0, 4.0, 8.0, 10.0),
    lrs=(1e-2, 5e-3, 5e-3, 3e-3, 3e-3),
    iters=(150, 100, 100, 100, 50),
):
    if disp_bf_init is None:
        disp_bf_init = -disp_fb_init

    if m_fb_fixed is None:
        m_fb_fixed = torch.zeros_like(B)
    if m_bf_fixed is None:
        m_bf_fixed = torch.zeros_like(B)

    # Full-resolution sizes (tensor order: D,H,W)
    _, _, D_full, H_full, W_full = B.shape

    Dmin, Hmin, Wmin = 80, 80, 80

    # Grid size bounds
    g_min, g_max = 32, 64

    n_levels = len(lrs)

    # ---- Image pyramid: Nmin → full resolution (aspect-ratio preserving) ----

    scale_min = min(
        Dmin / D_full,
        Hmin / H_full,
        Wmin / W_full,
        1.0,  # never upscale
    )

    scales = [
        scale_min + (1.0 - scale_min) * i / max(n_levels - 1, 1)
        for i in range(n_levels)
    ]

    pyr_sizes = [
        (
            max(1, int(round(D_full * s))),
            max(1, int(round(H_full * s))),
            max(1, int(round(W_full * s))),
        )
        for s in scales
    ]

    # ---- Grid sizes: 32³ → 64³ ----
    grid_sizes = [
        int(round(g_min + (g_max - g_min) * i / max(n_levels - 1, 1)))
        for i in range(n_levels)
    ]

    # Initialize dense fields from DIRAC init
    disp_fb_full = disp_fb_init.clone().detach()
    disp_bf_full = disp_bf_init.clone().detach()

    for lvl, (lr, n_iter, lam_reg, lam_inv) in enumerate(
        zip(lrs, iters, lambdas_reg, lambdas_inv)
    ):
        D, H, W = pyr_sizes[lvl]
        g = grid_sizes[lvl]

        print(f"[lvl {lvl}] image={D}x{H}x{W}, grid={g}³")

        # ---- Downsample images + masks to current pyramid level ----
        B_l   = F.interpolate(B,   size=(D, H, W), mode="trilinear", align_corners=True)
        F_l   = F.interpolate(Fup, size=(D, H, W), mode="trilinear", align_corners=True)
        mfb_l = F.interpolate(m_fb_fixed, size=(D, H, W), mode="nearest")
        mbf_l = F.interpolate(m_bf_fixed, size=(D, H, W), mode="nearest")

        # Bring current dense init to this level
        disp_fb_l = resize_disp_voxel(disp_fb_full, size=(D, H, W))
        disp_bf_l = resize_disp_voxel(disp_bf_full, size=(D, H, W))

        # ---- Control-point grid parameterization (learnable variables) ----
        cp_fb = F.interpolate(disp_fb_l, size=(g, g, g),
                              mode="trilinear", align_corners=True
                              ).detach().requires_grad_(True)

        cp_bf = F.interpolate(disp_bf_l, size=(g, g, g),
                              mode="trilinear", align_corners=True
                              ).detach().requires_grad_(True)

        opt = torch.optim.Adam([cp_fb, cp_bf], lr=lr)

        for _ in range(n_iter):
            # Upsample CP grid → dense field at current level
            disp_fb = F.interpolate(cp_fb, size=(D, H, W),
                                    mode="trilinear", align_corners=True)
            disp_bf = F.interpolate(cp_bf, size=(D, H, W),
                                    mode="trilinear", align_corners=True)

            # Warp images
            F_warp = warp(F_l, disp_fb)
            B_warp = warp(B_l, disp_bf)

            # Similarity (with masks)
            Ls = (
                ncc_loss(B_l, F_warp, mask=(1 - mfb_l))
                + ncc_loss(F_l, B_warp, mask=(1 - mbf_l))
            )

            # Regularization
            Lr = smoothness(disp_fb) + smoothness(disp_bf)

            # Inverse consistency
            Linv = inv_consistency(disp_fb, disp_bf, mfb_l, mbf_l)

            loss = (1 - lam_reg) * Ls + lam_reg * Lr + lam_inv * Linv

            opt.zero_grad()
            loss.backward()
            opt.step()

        # ---- Promote result to full resolution for next level ----
        disp_fb_full = resize_disp_voxel(
            F.interpolate(cp_fb.detach(), size=(D, H, W),
                          mode="trilinear", align_corners=True),
            size=(D_full, H_full, W_full),
        )

        disp_bf_full = resize_disp_voxel(
            F.interpolate(cp_bf.detach(), size=(D, H, W),
                          mode="trilinear", align_corners=True),
            size=(D_full, H_full, W_full),
        )

    return disp_fb_full.detach(), disp_bf_full.detach(), \
           m_fb_fixed.detach(), m_bf_fixed.detach()


# ---------- DIRAC I/O conversion ----------

def load_image_for_grid_sample(path, device):
    img = nib.load(path).get_fdata().astype(np.float32)  # (H,W,D)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    return img.to(device)


def load_mask_for_grid_sample(path, device):
    mask = nib.load(path).get_fdata().astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)
    mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    return mask.to(device)


def load_dirac_voxel_disp_for_grid_sample(path, device):
    disp = nib.load(path).get_fdata().astype(np.float32)  # (H,W,D,3)

    # DIRAC voxel output components follow image axes: [axis0(H), axis1(W), axis2(D)].
    # grid_sample expects channels [dx(W), dy(H), dz(D)] for tensor shape (1,3,D,H,W).
    comp_axis0 = torch.from_numpy(disp[..., 0]).permute(2, 0, 1)  # (D,H,W)
    comp_axis1 = torch.from_numpy(disp[..., 1]).permute(2, 0, 1)  # (D,H,W)
    comp_axis2 = torch.from_numpy(disp[..., 2]).permute(2, 0, 1)  # (D,H,W)

    disp_grid = torch.stack((comp_axis1, comp_axis0, comp_axis2), dim=0).unsqueeze(0)
    return disp_grid.to(device)


def grid_sample_disp_to_dirac_voxel(disp):
    # disp: (1,3,D,H,W) with [dx(W), dy(H), dz(D)]
    dx = disp[0, 0].permute(1, 2, 0).cpu().numpy()  # (H,W,D)
    dy = disp[0, 1].permute(1, 2, 0).cpu().numpy()  # (H,W,D)
    dz = disp[0, 2].permute(1, 2, 0).cpu().numpy()  # (H,W,D)

    out = np.stack((dy, dx, dz), axis=-1).astype(np.float32)  # DIRAC voxel component order
    return out


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

    disp_bf_path = None
    bwd_candidates = sorted(glob.glob(os.path.join(patient_dir, "*_preop_to_followup_disp_voxel.nii.gz")))
    if len(bwd_candidates) == 1:
        disp_bf_path = bwd_candidates[0]

    preop = load_image_for_grid_sample(preop_path, device)
    followup = load_image_for_grid_sample(followup_path, device)
    preop_mask = load_mask_for_grid_sample(preop_mask_path, device)
    followup_mask = load_mask_for_grid_sample(followup_mask_path, device)
    disp_fb = load_dirac_voxel_disp_for_grid_sample(disp_fb_path, device)
    disp_bf = load_dirac_voxel_disp_for_grid_sample(disp_bf_path, device) if disp_bf_path else None

    lrs = tuple(float(x) for x in args.lrs.split(","))
    iters = tuple(int(x) for x in args.iters.split(","))
    lambdas_reg = tuple(float(x) for x in args.lambdas_reg.split(","))
    lambdas_inv = tuple(float(x) for x in args.lambdas_inv.split(","))

    schedule_len = len(lrs)
    if not (len(iters) == len(lambdas_reg) == len(lambdas_inv) == schedule_len):
        raise ValueError("lrs, iters, lambdas_reg and lambdas_inv must have the same number of entries")

    disp_fb_opt, disp_bf_opt, m_fb, m_bf = dirac_instance_optimization(
        B=preop,
        Fup=followup,
        disp_fb_init=disp_fb,
        disp_bf_init=disp_bf,
        m_fb_fixed=preop_mask,
        m_bf_fixed=followup_mask,
        lrs=lrs,
        iters=iters,
        lambdas_reg=lambdas_reg,
        lambdas_inv=lambdas_inv,
    )

    ref = nib.load(preop_path)
    header, affine = ref.header, ref.affine

    fb_voxel = grid_sample_disp_to_dirac_voxel(disp_fb_opt)
    bf_voxel = grid_sample_disp_to_dirac_voxel(disp_bf_opt)

    save_flow(
        fb_voxel,
        os.path.join(patient_dir, f"{patient_id}_followup_to_preop_disp_voxel_optimized.nii.gz"),
        header=header,
        affine=affine,
    )
    save_flow(
        bf_voxel,
        os.path.join(patient_dir, f"{patient_id}_preop_to_followup_disp_voxel_optimized.nii.gz"),
        header=header,
        affine=affine,
    )

    if args.save_masks:
        save_flow(
            m_fb[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            os.path.join(patient_dir, f"{patient_id}_followup_to_preop_mask_optimized.nii.gz"),
            header=header,
            affine=affine,
        )
        save_flow(
            m_bf[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            os.path.join(patient_dir, f"{patient_id}_preop_to_followup_mask_optimized.nii.gz"),
            header=header,
            affine=affine,
        )

    print(f"[ok] {patient_id}: optimized fields saved")


def main():
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="../Dataset/test", help="Path containing patient folders")
    parser.add_argument("--lrs", type=str, default="1e-2,5e-3,5e-3,3e-3,3e-3", help="Comma-separated LR schedule")
    parser.add_argument("--iters", type=str, default="150,100,100,100,50", help="Comma-separated iteration schedule")
    parser.add_argument(
        "--lambdas_reg",
        type=str,
        default="0.25,0.3,0.3,0.35,0.35",
        help="Comma-separated regularization weights",
    )
    parser.add_argument(
        "--lambdas_inv",
        type=str,
        default="1.0,2.0,4.0,8.0,10.0",
        help="Comma-separated inverse-consistency weights",
    )
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

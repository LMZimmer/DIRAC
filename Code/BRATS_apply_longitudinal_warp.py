import glob
import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def make_identity_grid(d, h, w, device, dtype):
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    zs = torch.linspace(-1, 1, d, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)[None]  # (1,D,H,W,3)


def voxel_disp_to_norm(disp, d, h, w):
    dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]
    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)
    sz = 2.0 / max(d - 1, 1)
    return torch.stack((dx * sx, dy * sy, dz * sz), dim=-1)


def warp(img, disp, mode="bilinear"):
    _, _, d, h, w = img.shape
    grid0 = make_identity_grid(d, h, w, img.device, img.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, d, h, w)
    return F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=True)


def load_image_for_grid_sample(path, device):
    img = nib.load(path).get_fdata().astype(np.float32)  # (H,W,D)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    return img.to(device)


def load_dirac_voxel_disp_for_grid_sample(path, device):
    disp = nib.load(path).get_fdata().astype(np.float32)  # (H,W,D,3)

    # DIRAC stores displacements in [axis0(H), axis1(W), axis2(D)] component order.
    # grid_sample expects [dx(W), dy(H), dz(D)] for (1,3,D,H,W).
    comp_axis0 = torch.from_numpy(disp[..., 0]).permute(2, 0, 1)  # (D,H,W)
    comp_axis1 = torch.from_numpy(disp[..., 1]).permute(2, 0, 1)  # (D,H,W)
    comp_axis2 = torch.from_numpy(disp[..., 2]).permute(2, 0, 1)  # (D,H,W)

    disp_grid = torch.stack((comp_axis1, comp_axis0, comp_axis2), dim=0).unsqueeze(0)
    return disp_grid.to(device)


def tensor_to_hwd_numpy(tensor):
    # input: (1,1,D,H,W) -> output: (H,W,D)
    return tensor[0, 0].permute(1, 2, 0).cpu().numpy()


def run_patient(patient_dir, device, no_instanceopt=False):
    patient_id = os.path.basename(patient_dir.rstrip("/"))

    preop_path = os.path.join(patient_dir, "t1c_bet_normalized.nii.gz")
    followup_path = os.path.join(patient_dir, "t1c_bet_normalized_followup.nii.gz")
    tumor_seg_path = os.path.join(patient_dir, "tumor_seg.nii.gz")

    disp_path = None
    disp_alt_path = None
    if not no_instanceopt:
        disp_candidates = sorted(
            glob.glob(os.path.join(patient_dir, "*_followup_to_preop_disp_voxel_optimized.nii.gz"))
        )
        if len(disp_candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one optimized followup->preop field in {patient_dir}, found: {disp_candidates}"
            )
        disp_path = disp_candidates[0]

        disp_alt_candidates = sorted(
            glob.glob(os.path.join(patient_dir, "*_followup_to_preop_disp_voxel_optimized_alt.nii.gz"))
        )
        if len(disp_alt_candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one optimized_alt followup->preop field in {patient_dir}, found: {disp_alt_candidates}"
            )
        disp_alt_path = disp_alt_candidates[0]

    infer_disp_candidates = sorted(
        glob.glob(os.path.join(patient_dir, "*_followup_to_preop_disp_voxel.nii.gz"))
    )
    if len(infer_disp_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one followup->preop field from BRATS_infer_DIRAC.py in {patient_dir}, found: {infer_disp_candidates}"
        )
    infer_disp_path = infer_disp_candidates[0]

    required_paths = [preop_path, followup_path, tumor_seg_path, infer_disp_path]
    if disp_path is not None:
        required_paths.append(disp_path)
    if disp_alt_path is not None:
        required_paths.append(disp_alt_path)
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        print(f"[skip] {patient_id}: missing required files: {missing}")
        return

    tumor_seg = load_image_for_grid_sample(tumor_seg_path, device)
    disp_fb_infer = load_dirac_voxel_disp_for_grid_sample(infer_disp_path, device)

    followup_warped = None
    tumor_warped = None
    followup_warped_alt = None
    tumor_warped_alt = None
    if not no_instanceopt:
        followup = load_image_for_grid_sample(followup_path, device)
        disp_fb_opt = load_dirac_voxel_disp_for_grid_sample(disp_path, device)
        disp_fb_opt_alt = load_dirac_voxel_disp_for_grid_sample(disp_alt_path, device)
        followup_warped = warp(followup, disp_fb_opt, mode="bilinear")
        tumor_warped = warp(tumor_seg, disp_fb_opt, mode="nearest")
        followup_warped_alt = warp(followup, disp_fb_opt_alt, mode="bilinear")
        tumor_warped_alt = warp(tumor_seg, disp_fb_opt_alt, mode="nearest")

    recurrence_y_x = warp(tumor_seg, disp_fb_infer, mode="nearest")

    reference = nib.load(preop_path)
    header, affine = reference.header, reference.affine

    if followup_warped is not None and tumor_warped is not None:
        nib.save(
            nib.Nifti1Image(tensor_to_hwd_numpy(followup_warped).astype(np.float32), affine=affine, header=header),
            os.path.join(patient_dir, "t1c_warped_longitudinal.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(tensor_to_hwd_numpy(tumor_warped).astype(np.float32), affine=affine, header=header),
            os.path.join(patient_dir, "recurrence_preop.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(
                tensor_to_hwd_numpy(followup_warped_alt).astype(np.float32),
                affine=affine,
                header=header,
            ),
            os.path.join(patient_dir, "t1c_warped_longitudinal_alt.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(tensor_to_hwd_numpy(tumor_warped_alt).astype(np.float32), affine=affine, header=header),
            os.path.join(patient_dir, "recurrence_preop_alt.nii.gz"),
        )
    nib.save(
        nib.Nifti1Image(tensor_to_hwd_numpy(recurrence_y_x).astype(np.float32), affine=affine, header=header),
        os.path.join(patient_dir, f"{patient_id}_Y_X_recurrence.nii.gz"),
    )

    if no_instanceopt:
        print(f"[ok] {patient_id}: wrote {patient_id}_Y_X_recurrence.nii.gz")
    else:
        print(
            f"[ok] {patient_id}: wrote t1c_warped_longitudinal.nii.gz, recurrence_preop.nii.gz, "
            f"t1c_warped_longitudinal_alt.nii.gz, recurrence_preop_alt.nii.gz, "
            f"and {patient_id}_Y_X_recurrence.nii.gz"
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="../Dataset/test", help="Path containing patient folders")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run only the first patient folder found in datapath",
    )
    parser.add_argument(
        "--no_instanceopt",
        action="store_true",
        help=(
            "Skip optimized-warp outputs (t1c_warped_longitudinal(.nii.gz/_alt.nii.gz) and "
            "recurrence_preop(.nii.gz/_alt.nii.gz)) "
            "and only write {patient_id}_Y_X_recurrence.nii.gz"
        ),
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
        run_patient(patient_dir, device, no_instanceopt=args.no_instanceopt)

    print("Done.")


if __name__ == "__main__":
    main()

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


def run_patient(patient_dir, device):
    patient_id = os.path.basename(patient_dir.rstrip("/"))

    flair_path = os.path.join(patient_dir, "flair_bet_normalized.nii.gz")
    if not os.path.exists(flair_path):
        print(f"[skip] {patient_id}: flair_bet_normalized.nii.gz not found")
        return

    disp_candidates = sorted(
        glob.glob(os.path.join(patient_dir, "*_followup_to_preop_disp_voxel_optimized.nii.gz"))
    )
    if len(disp_candidates) != 1:
        print(
            f"[skip] {patient_id}: expected exactly one optimized followup->preop field, "
            f"found: {disp_candidates}"
        )
        return
    disp_path = disp_candidates[0]

    flair = load_image_for_grid_sample(flair_path, device)
    disp_fb_opt = load_dirac_voxel_disp_for_grid_sample(disp_path, device)
    flair_warped = warp(flair, disp_fb_opt, mode="bilinear")

    reference = nib.load(flair_path)
    header, affine = reference.header, reference.affine
    nib.save(
        nib.Nifti1Image(tensor_to_hwd_numpy(flair_warped).astype(np.float32), affine=affine, header=header),
        os.path.join(patient_dir, "flair_warped_longitudinal.nii.gz"),
    )

    print(f"[ok] {patient_id}: wrote flair_warped_longitudinal.nii.gz")


def main():
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="../Dataset/test", help="Path containing patient folders")
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
        run_patient(patient_dir, device)

    print("Done.")


if __name__ == "__main__":
    main()

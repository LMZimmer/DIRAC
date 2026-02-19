import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def warp_volume_with_voxel_displacement(volume: np.ndarray, disp_voxel: np.ndarray, order: int) -> np.ndarray:
    """Warp a 3D volume using voxel-space displacement with axis order (x, y, z)."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    if disp_voxel.shape[:3] != volume.shape or disp_voxel.shape[3] != 3:
        raise ValueError(
            "Displacement field must have shape (H, W, D, 3) and match the volume shape. "
            f"Got disp={disp_voxel.shape}, volume={volume.shape}."
        )

    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(volume.shape[0], dtype=np.float32),
        np.arange(volume.shape[1], dtype=np.float32),
        np.arange(volume.shape[2], dtype=np.float32),
        indexing="ij",
    )

    sample_coords = [
        grid_x + disp_voxel[..., 0],
        grid_y + disp_voxel[..., 1],
        grid_z + disp_voxel[..., 2],
    ]

    warped = map_coordinates(
        volume,
        sample_coords,
        order=order,
        mode="nearest",
    )
    return warped


def main() -> None:
    parser = ArgumentParser(description="Validate stored DIRAC transforms for a single patient folder.")
    parser.add_argument("patient_dir", type=str, help="Path to patient directory with MRI inputs + DIRAC outputs")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional prefix for output files. Defaults to patient directory basename.",
    )
    args = parser.parse_args()

    patient_dir = os.path.abspath(args.patient_dir)
    patient_id = args.output_prefix or os.path.basename(os.path.normpath(patient_dir))

    followup_path = os.path.join(patient_dir, "t1c_bet_normalized_followup.nii.gz")
    transformed_followup_path = os.path.join(patient_dir, f"{patient_id}_Y_X.nii.gz")
    disp_voxel_path = os.path.join(patient_dir, f"{patient_id}_followup_to_preop_disp_voxel.nii.gz")
    tumor_seg_path = os.path.join(patient_dir, "tumor_seg.nii.gz")

    required_paths = [followup_path, transformed_followup_path, disp_voxel_path, tumor_seg_path]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    followup_nifti = nib.load(followup_path)
    followup = followup_nifti.get_fdata().astype(np.float32)

    transformed_followup_nifti = nib.load(transformed_followup_path)
    transformed_followup = transformed_followup_nifti.get_fdata().astype(np.float32)
    disp_voxel = nib.load(disp_voxel_path).get_fdata().astype(np.float32)
    tumor_seg = nib.load(tumor_seg_path).get_fdata().astype(np.float32)

    recomputed_followup = warp_volume_with_voxel_displacement(followup, disp_voxel, order=1)
    warped_tumor_seg = warp_volume_with_voxel_displacement(tumor_seg, disp_voxel, order=0)

    mae = float(np.mean(np.abs(recomputed_followup - transformed_followup)))
    max_abs = float(np.max(np.abs(recomputed_followup - transformed_followup)))

    comparison = np.allclose(recomputed_followup, transformed_followup, atol=1e-4)
    print(f"Patient: {patient_id}")
    print(f"Recomputed-vs-saved follow-up match (atol=1e-4): {comparison}")
    print(f"MAE: {mae:.8f}")
    print(f"Max abs diff: {max_abs:.8f}")

    recomputed_path = os.path.join(patient_dir, f"{patient_id}_Y_X_recomputed_from_saved_disp.nii.gz")
    tumor_warped_path = os.path.join(patient_dir, f"{patient_id}_tumor_seg_in_preop_space.nii.gz")

    nib.save(nib.Nifti1Image(recomputed_followup, transformed_followup_nifti.affine, transformed_followup_nifti.header), recomputed_path)
    nib.save(nib.Nifti1Image(warped_tumor_seg, transformed_followup_nifti.affine, transformed_followup_nifti.header), tumor_warped_path)

    print(f"Saved recomputed transformed follow-up: {recomputed_path}")
    print(f"Saved transformed tumor segmentation: {tumor_warped_path}")


if __name__ == "__main__":
    main()

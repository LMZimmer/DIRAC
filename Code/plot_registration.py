#!/usr/bin/env python3
"""Generate per-patient registration comparison grids in a single PDF."""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage


COLUMN_FILES = [
    "t1c_bet_normalized.nii.gz",
    "t1c_bet_normalized_followup.nii.gz",
    "{patient_id}_Y_X.nii.gz",
    "t1c_warped_longitudinal.nii.gz",
    "t1c_warped_longitudinal_alt.nii.gz",
]

COLUMN_LABELS = [
    "Pre-op",
    "Follow-up",
    "F-up (dirac)",
    "F-up (dirac+ncc)",
    "F-up (dirac+ffd)",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a PDF with one page per patient containing axial/sagittal/coronal "
            "grids comparing registration outputs."
        )
    )
    parser.add_argument(
        "--datapath",
        type=Path,
        required=True,
        default='../Dataset/predict_gbm',
        help="Folder containing one subfolder per patient.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("longitudinal_registration.pdf"),
        help="Output PDF path.",
    )
    return parser.parse_args()



def load_volume(path: Path) -> np.ndarray:
    volume = nib.load(str(path)).get_fdata()
    return np.squeeze(volume)



def get_center_of_mass(segmentation: np.ndarray) -> Tuple[int, int, int]:
    mask = np.isin(segmentation, [1, 3])
    if not np.any(mask):
        mask = np.isin(segmentation, [1, 2, 3, 4])

    if np.any(mask):
        com = ndimage.center_of_mass(mask.astype(np.float32))
    else:
        shape = segmentation.shape
        com = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)

    coords = []
    for idx, value in enumerate(com):
        axis_max = segmentation.shape[idx] - 1
        coords.append(int(np.clip(np.round(value), 0, axis_max)))
    return tuple(coords)



def get_slice(volume: np.ndarray, row: int, center: Tuple[int, int, int]) -> np.ndarray:
    x, y, z = center
    if row == 0:  # axial
        image = volume[:, :, z]
    elif row == 1:  # sagittal
        image = volume[x, :, :]
    else:  # coronal
        image = volume[:, y, :]

    return np.rot90(np.asarray(image))



def add_patient_page(pdf: PdfPages, patient_dir: Path) -> bool:
    patient_id = patient_dir.name
    seg_path = patient_dir / "tumor_seg.nii.gz"

    if not seg_path.exists():
        print(f"[WARN] Skipping {patient_id}: missing {seg_path.name}")
        return False

    image_paths = [patient_dir / name.format(patient_id=patient_id) for name in COLUMN_FILES]
    missing = [path.name for path in image_paths if not path.exists()]
    if missing:
        print(f"[WARN] Skipping {patient_id}: missing files {missing}")
        return False

    seg = load_volume(seg_path)
    center = get_center_of_mass(seg)
    volumes = [load_volume(path) for path in image_paths]

    fig, axes = plt.subplots(3, 5, figsize=(16, 10), constrained_layout=True)
    for col, label in enumerate(COLUMN_LABELS):
        axes[0, col].set_title(label, fontsize=12)

    for row in range(3):
        for col, volume in enumerate(volumes):
            axes[row, col].imshow(get_slice(volume, row, center), cmap="gray")
            axes[row, col].axis("off")

    fig.suptitle(f"{patient_id} | COM (labels 1 & 3): {center}", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)
    return True



def main() -> None:
    args = parse_args()
    if not args.datapath.exists():
        raise FileNotFoundError(f"Data path does not exist: {args.datapath}")

    patient_dirs = sorted([d for d in args.datapath.iterdir() if d.is_dir()])
    if not patient_dirs:
        raise RuntimeError(f"No patient folders found in {args.datapath}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with PdfPages(args.output) as pdf:
        for patient_dir in patient_dirs:
            if add_patient_page(pdf, patient_dir):
                processed += 1

    print(f"Saved PDF to {args.output} with {processed} patient page(s).")


if __name__ == "__main__":
    main()

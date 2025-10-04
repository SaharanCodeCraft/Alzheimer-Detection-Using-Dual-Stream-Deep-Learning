import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from nipype.interfaces.fsl import BET

# === CONFIGURATION ===
os.environ["FSLDIR"]        = "/Users/harshakhiyani/fsl/share/fsl"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
os.environ["PATH"]         += os.pathsep + os.path.join(os.environ["FSLDIR"], "bin")

# Your data locations
root_dir   = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/REAL_ADNI/ADNI"
csv_path   = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/ADNI1_Baseline_Only.csv"
output_dir = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/Processed_224x224"
os.makedirs(output_dir, exist_ok=True)

# Processing parameters
NUM_SLICES    = 50
BRAIN_THRESH  = 1e-6
BET_FRAC_LIST = [0.50, 0.40, 0.30, 0.20]   # try these in order
MIN_VOXELS    = 500_000                   # require at least this many mask voxels

# === FUNCTIONS ===

def run_bet_with_retries(in_nii: str, out_base: str) -> str:
    """
    Run BET at several frac thresholds until the mask has >= MIN_VOXELS.
    Writes out_base.nii.gz and out_base_mask.nii.gz.
    Returns the stripped NIfTI path.
    """
    for frac in BET_FRAC_LIST:
        print(f"    ↪ BET frac={frac:.2f}")
        bet = BET(in_file=in_nii, out_file=out_base + ".nii.gz",
                  mask=True, robust=True, frac=frac)
        res = bet.run()
        stripped = res.outputs.out_file
        mask_file = res.outputs.mask_file

        mask_vol = np.sum(nib.load(mask_file).get_fdata() > 0)
        print(f"      → mask voxels: {mask_vol:,}")
        if mask_vol >= MIN_VOXELS:
            print(f"      ✓ accepted at frac={frac:.2f}")
            return stripped

        print("      ✗ too small, retrying…")

    raise RuntimeError(f"All BET attempts failed (< {MIN_VOXELS} voxels)")

def normalize_slice(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d - img2d.min()
    if img2d.max() > 0:
        img2d = img2d / img2d.max()
    return (img2d * 255).astype(np.uint8)

def extract_sagittal_slices(stripped_nii: str, png_dir: str):
    """
    Given a skull-stripped NIfTI, extract NUM_SLICES sagittal slices
    covering the brain, normalize to uint8, resize to 224×224, save PNGs.
    """
    img  = nib.load(stripped_nii)
    data = img.get_fdata()  # (X, Y, Z)

    xs = [i for i in range(data.shape[0]) if data[i].mean() > BRAIN_THRESH]
    if len(xs) < NUM_SLICES:
        raise RuntimeError(f"Only {len(xs)} brain slices in {os.path.basename(stripped_nii)}")

    idxs = np.linspace(xs[0], xs[-1], NUM_SLICES, dtype=int)
    os.makedirs(png_dir, exist_ok=True)
    for i, x in enumerate(idxs):
        sl = np.rot90(data[x, :, :])
        sl = normalize_slice(sl)
        slr = zoom(sl, (224/sl.shape[0], 224/sl.shape[1]), order=3)
        plt.imsave(os.path.join(png_dir, f"slice_{i:03d}.png"), slr, cmap="gray")


# === MAIN ===

df = pd.read_csv(csv_path)
for subj in df["PTID"].unique():
    subj_dir = os.path.join(root_dir, subj)
    if not os.path.isdir(subj_dir):
        continue

    # locate the first “Scaled” .nii
    nii_path = None
    for r, _, files in os.walk(subj_dir):
        for f in files:
            if f.endswith(".nii") and "Scaled" in f:
                nii_path = os.path.join(r, f)
                break
        if nii_path:
            break

    if not nii_path:
        print(f"[{subj}] ⛔ no Scaled .nii found, skipping")
        continue

    print(f"[{subj}] processing {os.path.basename(nii_path)}")
    try:
        # 1) rerun BET into a new subfolder of the subject
        nii_dir    = os.path.dirname(nii_path)
        rerun_dir  = os.path.join(nii_dir, "re_run_with_diff_paramter")
        os.makedirs(rerun_dir, exist_ok=True)

        base       = os.path.join(rerun_dir, "brain_stripped")
        stripped   = run_bet_with_retries(nii_path, base)

        # 2) extract slices into central output, but new folder name
        png_folder = os.path.join(
            output_dir,
            subj,
            "re_run_slices_with_diff_parameters_axial_224x224"
        )
        extract_sagittal_slices(stripped, png_folder)

        print(f"[{subj}] ✅ done\n")

    except Exception as e:
        print(f"[{subj}] ❌ failed: {e}\n")

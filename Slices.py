import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, binary_closing, binary_fill_holes, label
from nipype.interfaces.fsl import BET

# === CONFIGURATION ===
os.environ["FSLDIR"]        = "/Users/harshakhiyani/fsl/share/fsl"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
os.environ["PATH"]         += os.pathsep + os.path.join(os.environ["FSLDIR"], "bin")

root_dir     = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/REAL_ADNI/ADNI"
csv_path     = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/ADNI1_Baseline_Only.csv"
slices_root  = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/Slices_Trail_3"
os.makedirs(slices_root, exist_ok=True)

NUM_SLICES    = 50
BRAIN_THRESH  = 1e-6
BET_FRAC_LIST = [0.50, 0.40, 0.30, 0.20]
MIN_VOXELS    = 500_000

# === MASK REFINEMENT ===
def refine_mask(mask_data):
    filled = binary_fill_holes(mask_data)
    closed = binary_closing(filled, structure=np.ones((3,3,3)))
    labels, n_lab = label(closed)
    if n_lab == 0:
        return closed
    counts = np.bincount(labels.flatten())
    counts[0] = 0
    largest = np.argmax(counts)
    return (labels == largest)

# === BET + REFINE ===
def skull_strip_and_refine(in_nii, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "brain_stripped")
    for frac in BET_FRAC_LIST:
        print(f"    ↪ BET frac={frac:.2f}")
        bet = BET(
            in_file=in_nii,
            out_file=base + ".nii.gz",
            mask=True, robust=True, frac=frac
        )
        res = bet.run()
        raw_mask = res.outputs.mask_file
        mask_data = nib.load(raw_mask).get_fdata() > 0
        print(f"      → raw mask voxels: {mask_data.sum():,}")

        refined = refine_mask(mask_data)
        print(f"      → refined mask voxels: {refined.sum():,}")

        if refined.sum() < MIN_VOXELS:
            print("      ✗ too small after refinement, retrying…")
            continue

        # save refined mask
        mask_refined = os.path.join(out_dir, "brain_stripped_refined_mask.nii.gz")
        nib.save(nib.Nifti1Image(refined.astype(np.uint8),
                                 nib.load(raw_mask).affine),
                 mask_refined)

        # apply to original image
        orig = nib.load(in_nii)
        stripped_refined = orig.get_fdata() * refined
        out_stripped = os.path.join(out_dir, "brain_stripped_refined.nii.gz")
        nib.save(nib.Nifti1Image(stripped_refined, orig.affine),
                 out_stripped)

        print(f"      ✓ success at frac={frac:.2f}")
        return out_stripped

    raise RuntimeError("All BET+refine attempts failed")

# === SLICE EXTRACTION ===
def normalize_slice(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)

def extract_slices(stripped_nii, png_dir):
    img  = nib.load(stripped_nii)
    data = img.get_fdata()
    xs   = [i for i in range(data.shape[0]) if data[i].mean() > BRAIN_THRESH]
    if len(xs) < NUM_SLICES:
        raise RuntimeError(f"Only {len(xs)} brain slices in {os.path.basename(stripped_nii)}")
    idxs = np.linspace(xs[0], xs[-1], NUM_SLICES, dtype=int)
    os.makedirs(png_dir, exist_ok=True)
    for i, x in enumerate(idxs):
        sl  = np.rot90(data[x, :, :])
        sl  = normalize_slice(sl)
        slr = zoom(sl, (224/sl.shape[0], 224/sl.shape[1]), order=3)
        plt.imsave(os.path.join(png_dir, f"slice_{i:03d}.png"), slr, cmap="gray")

# === MAIN LOOP ===
df = pd.read_csv(csv_path)
for subj in df["PTID"].unique():
    subj_dir = os.path.join(root_dir, subj)
    if not os.path.isdir(subj_dir):
        continue

    # find first “Scaled” .nii
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
        # 1) Skull-strip + refine
        rerun_dir = os.path.join(os.path.dirname(nii_path), "re_run_with_diff_paramter")
        stripped = skull_strip_and_refine(nii_path, rerun_dir)

        # 2) Extract slices into fixed location
        png_out = os.path.join(slices_root, subj,
                               "re_run_slices_with_diff_parameters_axial_224x224")
        extract_slices(stripped, png_out)

        print(f"[{subj}] ✅ done\n")

    except Exception as e:
        print(f"[{subj}] ❌ failed: {e}\n")

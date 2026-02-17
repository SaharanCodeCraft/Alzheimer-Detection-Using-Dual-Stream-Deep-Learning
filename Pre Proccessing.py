import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from nipype.interfaces.fsl import BET

# === CONFIG ===
# 1) FSL environment
os.environ["FSLDIR"]       = "/Users/harshakhiyani/fsl/share/fsl"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
os.environ["PATH"]        += os.pathsep + os.path.join(os.environ["FSLDIR"], "bin")

# 2) Root folder—now with no spaces!
root_dir    = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/REAL_ADNI/ADNI"

# 3) Your CSV of PTIDs
csv_path    = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/ADNI1_Baseline_Only.csv"

# 4) Where to dump the 224×224 PNGs
output_dir  = "/Volumes/Samsung_PSSD_T7_Shield/ADNI/Processed_224x224"
os.makedirs(output_dir, exist_ok=True)

# === PARAMETERS ===
NUM_SLICES   = 50
BRAIN_THRESH = 1e-6
BET_FRAC     = 0.5
BET_ROBUST   = True

# === FUNCTIONS ===
def skull_strip(in_nii, out_nii):
    bet = BET(in_file=in_nii, out_file=out_nii, mask=True,
              robust=BET_ROBUST, frac=BET_FRAC)
    res = bet.run()
    return res.outputs.out_file

def normalize_slice(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)

def extract_slices(stripped_nii, png_dir):
    img  = nib.load(stripped_nii)
    data = img.get_fdata()  # (X,Y,Z)
    xs   = [i for i in range(data.shape[0]) if data[i].mean() > BRAIN_THRESH]
    if len(xs) < NUM_SLICES:
        raise RuntimeError(f"Only {len(xs)} brain slices in {stripped_nii}")
    idxs = np.linspace(xs[0], xs[-1], NUM_SLICES, dtype=int)
    os.makedirs(png_dir, exist_ok=True)
    for i, x in enumerate(idxs):
        sl = np.rot90(data[x, :, :])
        sl = normalize_slice(sl)
        slr = zoom(sl, (224/sl.shape[0], 224/sl.shape[1]), order=3)
        plt.imsave(f"{png_dir}/slice_{i:03d}.png", slr, cmap="gray")

# === MAIN ===
df = pd.read_csv(csv_path)
for subj in df["PTID"].unique():
    subj_dir = os.path.join(root_dir, subj)
    if not os.path.isdir(subj_dir):
        continue

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

    try:
        # 1) Skull-strip in place
        stripped = os.path.join(os.path.dirname(nii_path), "brain_stripped.nii.gz")
        print(f"[{subj}] 🧠 stripping → {stripped}")
        skull_strip(nii_path, stripped)

        # 2) Extract & save slices centrally
        png_out = os.path.join(output_dir, subj, "slices_224")
        print(f"[{subj}] 📐 extracting → {png_out}")
        extract_slices(stripped, png_out)

        print(f"[{subj}] ✅ done")

    except Exception as e:
        print(f"[{subj}] ❌ failed: {e}")

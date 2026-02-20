import os
import glob
import h5py
import numpy as np
import scipy.io as sio

# ========= USER CONFIG =========
# Where your raw fastMRI brain multicoil .h5 files live (adjust!)
RAW_TRAIN_DIR = os.curdir + "/raw/train"
RAW_TEST_DIR  = os.curdir + "/raw/test"

# Where to save the SSDU-ready files
OUT_DIR = os.curdir + "/ssdu_prepared"

# Undersampling settings (1D along k-space columns)
ACCEL = 4           # nominal acceleration R
CENTER_FRAC = 0.08  # fraction of k-space columns fully sampled in the center

# Target number of coils (MUST match args.ncoil_GLOB in parser_ops.py)
TARGET_NCOILS = 14

# Limit number of volumes (set to None to use all)
MAX_TRAIN_VOLS = 10
MAX_TEST_VOLS  = 5
# ===============================


os.makedirs(OUT_DIR, exist_ok=True)


def build_1d_var_dens_mask(ncol, accel=4, center_frac=0.08, rng=None):
    if rng is None:
        rng = np.random.RandomState(1234)

    num_center = int(round(ncol * center_frac))
    num_target = int(round(ncol / accel))

    if num_center > num_target:
        raise ValueError("CENTER_FRAC too large relative to ACCEL.")

    mask = np.zeros(ncol, dtype=np.float32)

    cstart = ncol // 2 - num_center // 2
    cend = cstart + num_center
    mask[cstart:cend] = 1.0

    num_remaining = num_target - num_center
    if num_remaining > 0:
        outer_indices = np.concatenate(
            [np.arange(0, cstart), np.arange(cend, ncol)]
        )
        prob = float(num_remaining) / float(len(outer_indices))
        rand = rng.rand(len(outer_indices))
        chosen = outer_indices[rand < prob]
        mask[chosen] = 1.0

    return mask


def ensure_target_coils(ks, target_ncoils):
    """
    ks: (ncoils_raw, nrow, ncol)
    -> (target_ncoils, nrow, ncol)
    """
    ncoils_raw, nrow, ncol = ks.shape
    ks_out = np.zeros((target_ncoils, nrow, ncol), dtype=ks.dtype)
    if ncoils_raw >= target_ncoils:
        ks_out[:, :, :] = ks[:target_ncoils]
    else:
        ks_out[:ncoils_raw, :, :] = ks
    return ks_out


def ensure_target_spatial(ks, target_nrow, target_ncol):
    """
    ks: (ncoils, nrow_raw, ncol_raw)
    -> (ncoils, target_nrow, target_ncol)
    centered crop / pad in k-space.
    """
    ncoils, nrow, ncol = ks.shape

    # rows
    if nrow > target_nrow:
        start = (nrow - target_nrow) // 2
        ks = ks[:, start:start+target_nrow, :]
    elif nrow < target_nrow:
        pad_before = (target_nrow - nrow) // 2
        pad_after = target_nrow - nrow - pad_before
        ks = np.pad(ks, ((0,0),(pad_before,pad_after),(0,0)))

    # cols
    ncoils2, nrow2, ncol2 = ks.shape
    if ncol2 > target_ncol:
        start = (ncol2 - target_ncol) // 2
        ks = ks[:, :, start:start+target_ncol]
    elif ncol2 < target_ncol:
        pad_before = (target_ncol - ncol2) // 2
        pad_after = target_ncol - ncol2 - pad_before
        ks = np.pad(ks, ((0,0),(0,0),(pad_before,pad_after)))

    return ks


def estimate_sens_maps_from_kspace(kspace_slice):
    """
    sens = coil_image / RSS_image
    kspace_slice: (ncoils, nrow, ncol)
    -> sens_maps: (ncoils, nrow, ncol)
    """
    coil_imgs = np.fft.ifft2(kspace_slice, axes=(-2, -1), norm="ortho")
    rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0)) + 1e-8
    sens_maps = coil_imgs / rss[None, ...]
    return sens_maps.astype(np.complex64)


def init_h5(path, nrow, ncol, ncoils):
    f = h5py.File(path, "w")
    dset = f.create_dataset(
        "data",
        shape=(0, nrow, ncol, ncoils),
        maxshape=(None, nrow, ncol, ncoils),
        dtype=np.complex64
    )
    return f, dset


def append_to_dset(dset, data_batch):
    n_old = dset.shape[0]
    n_new = n_old + data_batch.shape[0]
    dset.resize((n_new,) + dset.shape[1:])
    dset[n_old:n_new, ...] = data_batch


def process_split(raw_dir, max_vols, kspace_out_path, sens_out_path,
                  target_nrow, target_ncol, mask_2d):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.h5")))
    if max_vols is not None:
        files = files[:max_vols]

    if len(files) == 0:
        raise RuntimeError(f"No .h5 files found in {raw_dir}")

    print(f"Processing {len(files)} volumes from {raw_dir}")

    # HDF5 outputs with target spatial size
    f_ksp, dset_ksp = init_h5(kspace_out_path, target_nrow, target_ncol, TARGET_NCOILS)
    f_sens, dset_sens = init_h5(sens_out_path, target_nrow, target_ncol, TARGET_NCOILS)

    mask_2d = mask_2d.astype(np.float32)
    assert mask_2d.shape == (target_nrow, target_ncol)

    for fi, path in enumerate(files):
        print(f"[{fi+1}/{len(files)}] {os.path.basename(path)}")
        with h5py.File(path, "r") as f:
            kspace_full = f["kspace"][:]  # (nslices, ncoils_raw, nrow_raw, ncol_raw)

        nslices = kspace_full.shape[0]
        kspace_slices = []
        sens_slices = []

        for si in range(nslices):
            ks_raw = kspace_full[si]  # (ncoils_raw, nrow_raw, ncol_raw)

            # coils -> TARGET_NCOILS
            ks = ensure_target_coils(ks_raw, TARGET_NCOILS)

            # spatial -> (TARGET_NROW, TARGET_NCOL)
            ks = ensure_target_spatial(ks, target_nrow, target_ncol)

            # Undersample with mask Ω
            ks_und = ks * mask_2d[None, :, :]

            # Sensitivity from full ks
            sens = estimate_sens_maps_from_kspace(ks)

            # Reorder to (nrow, ncol, ncoil)
            ks_und_re = np.transpose(ks_und, (1, 2, 0))
            sens_re = np.transpose(sens, (1, 2, 0))

            kspace_slices.append(ks_und_re)
            sens_slices.append(sens_re)

        kspace_slices = np.stack(kspace_slices, axis=0)
        sens_slices = np.stack(sens_slices, axis=0)

        append_to_dset(dset_ksp, kspace_slices)
        append_to_dset(dset_sens, sens_slices)

    # Rename keys
    f_ksp["kspace"] = f_ksp["data"]
    del f_ksp["data"]
    f_sens["sens_maps"] = f_sens["data"]
    del f_sens["data"]

    f_ksp.close()
    f_sens.close()


# ===== MAIN =====

# Use FIRST TRAIN file to define target spatial size
sample_file = sorted(glob.glob(os.path.join(RAW_TRAIN_DIR, "*.h5")))[0]
with h5py.File(sample_file, "r") as f0:
    ksp0 = f0["kspace"][0]  # (ncoils_raw, nrow0, ncol0)
    _, nrow0, ncol0 = ksp0.shape

TARGET_NROW = nrow0
TARGET_NCOL = ncol0

print("TARGET_NROW:", TARGET_NROW)
print("TARGET_NCOL:", TARGET_NCOL)
print("TARGET_NCOILS:", TARGET_NCOILS)

# Global mask Ω for this target spatial size
mask_1d = build_1d_var_dens_mask(TARGET_NCOL, accel=ACCEL, center_frac=CENTER_FRAC)
mask_2d = np.tile(mask_1d[None, :], (TARGET_NROW, 1))

mask_mat_path = os.path.join(OUT_DIR, "brain_mask.mat")
sio.savemat(mask_mat_path, {"mask": mask_2d})
print(f"Saved mask to {mask_mat_path}, shape={mask_2d.shape}")

# # Train split
train_kspace_path = os.path.join(OUT_DIR, "brain_train_kspace.h5")
train_sens_path = os.path.join(OUT_DIR, "brain_train_sens_maps.h5")
process_split(RAW_TRAIN_DIR, MAX_TRAIN_VOLS, train_kspace_path, train_sens_path,
              TARGET_NROW, TARGET_NCOL, mask_2d)

# Test split
test_kspace_path = os.path.join(OUT_DIR, "brain_test_kspace.h5")
test_sens_path = os.path.join(OUT_DIR, "brain_test_sens_maps.h5")
process_split(RAW_TEST_DIR, MAX_TEST_VOLS, test_kspace_path, test_sens_path,
              TARGET_NROW, TARGET_NCOL, mask_2d)

print("Done preprocessing fastMRI brain for SSDU.")
print("Train kspace:", train_kspace_path)
print("Train sens:", train_sens_path)
print("Test kspace:", test_kspace_path)
print("Test sens:", test_sens_path)
print("Mask:", mask_mat_path)
"""Rejoin channels_A_N16384_part1.npz + _part2.npz into channels_A_N16384.npz (UE axis split)."""
import numpy as np
a = np.load("channels_A_N16384_part1.npz"); b = np.load("channels_A_N16384_part2.npz")
per_ue = ["hd", "g", "ue_xyz", "ideal_snr_db", "noris_snr_db"]
out = {k: a[k] for k in a.files}
for k in per_ue:
    out[k] = np.concatenate([a[k], b[k]], axis=0)
np.savez_compressed("channels_A_N16384.npz", **out)
print("merged", out["g"].shape)

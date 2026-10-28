"""Phase 1 (A) post-processing: pick best placement, export channels, maps, checks, summaries."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import ris_sim_A as A
import channel_io as cio

HERE = A.HERE
OUT = os.path.join(HERE, "maps")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(HERE, "results_A.csv"))

# ---------------------------------------------------------------- 6. best placement
agg = (df.groupby(["site", "az_deg", "downtilt_deg"])
         .agg(mean_cov=("coverage_pct", "mean"), mean_snr=("mean_snr_db", "mean"),
              mean_shadow_cov=("shadow_coverage_pct", "mean"))
         .reset_index().sort_values(["mean_cov", "mean_snr"], ascending=False))
agg.to_csv(os.path.join(HERE, "placement_ranking_A.csv"), index=False)
best = agg.iloc[0]
site, az, tl = best.site, int(best.az_deg), int(best.downtilt_deg)

per_N = []
for N in A.NS:
    d = df[df.N == N].sort_values(["coverage_pct", "mean_snr_db"], ascending=False).iloc[0]
    per_N.append(dict(N=N, best_site=d.site, best_az=int(d.az_deg), best_tilt=int(d.downtilt_deg),
                      coverage_pct=d.coverage_pct, mean_snr_db=d.mean_snr_db, p5_snr_db=d.p5_snr_db,
                      shadow_coverage_pct=d.shadow_coverage_pct,
                      same_as_overall_best=bool(d.site == site and int(d.az_deg) == az and int(d.downtilt_deg) == tl)))
json.dump(dict(overall_best=dict(site=site, position=A.SITES[site].tolist(), az_deg=az, downtilt_deg=tl,
                                 mean_coverage_pct_over_N=float(best.mean_cov), mean_snr_db_over_N=float(best.mean_snr)),
               best_per_N=per_N),
          open(os.path.join(HERE, "best_placement_A.json"), "w"), indent=2, default=float)
print("overall best:", site, az, tl)

# ---------------------------------------------------------------- 7. export channels (best placement, all N)
hd = A.direct_channel()
snr0 = A.snr_db_from_lin(A.PT_W * np.abs(hd) ** 2 / A.SIGMA2)
ideal = {}
for N in A.NS:
    ideal[N] = A.export_channels(site, az, tl, N, hd, snr0, HERE)
    print("exported channels_A_N%d.npz" % N, flush=True)

# ---------------------------------------------------------------- checks
rows = []
for N in A.NS:
    f = os.path.join(HERE, f"channels_A_N{N}.npz")
    d = cio.load_channels(f)
    s_file = cio.snr_db_from_file(d)  # recompute from the exported complex64 file with the shared helper
    s_sweep = df[(df.site == site) & (df.az_deg == az) & (df.downtilt_deg == tl) & (df.N == N)].iloc[0]
    # 1) RIS never worse than no RIS (small tol for complex64 rounding)
    assert np.all(s_file >= d["noris_snr_db"] - 1e-3), "RIS SNR < no-RIS SNR!"
    # 2) file-based recompute matches stored ideal SNR and the fast amplitude-sum sweep
    assert np.max(np.abs(s_file - d["ideal_snr_db"])) < 0.05, "file recompute != stored ideal"
    assert abs(s_file.mean() - s_sweep.mean_snr_db) < 0.05, "sweep mean != explicit-phase mean"
    # 3) random phases must not beat ideal
    rng = np.random.default_rng(0)
    th_r = rng.uniform(-np.pi, np.pi, size=d["g"].shape)
    s_rand = cio.snr_db_from_file(d, theta=th_r)
    assert np.all(s_rand <= s_file + 1e-3), "random phase beat ideal!"
    rows.append(dict(N=N, min_gain_db=float((s_file - d["noris_snr_db"]).min()),
                     max_abs_file_vs_sweep_mean_db=abs(s_file.mean() - s_sweep.mean_snr_db),
                     random_phase_mean_snr_db=float(s_rand.mean()), ideal_mean_snr_db=float(s_file.mean())))
chk = pd.DataFrame(rows)
chk.to_csv(os.path.join(HERE, "checks_A.csv"), index=False)
print(chk.to_string())
assert df.min_gain_db.min() >= -1e-9, "sweep: RIS lower than no-RIS somewhere"
print("ALL CHECKS PASSED: min gain over all 384 setups = %.3g dB" % df.min_gain_db.min())

# ---------------------------------------------------------------- summary table for D (best placement per N)
tbl = []
base_cov = df.noris_coverage_pct.iloc[0]
for N in A.NS:
    r = df[(df.site == site) & (df.az_deg == az) & (df.downtilt_deg == tl) & (df.N == N)].iloc[0]
    tbl.append(dict(N=N, site=site, az_deg=az, downtilt_deg=tl,
                    noris_mean_snr_db=r.noris_mean_snr_db, ris_mean_snr_db=r.mean_snr_db,
                    mean_gain_db=r.mean_gain_db, noris_coverage_pct=r.noris_coverage_pct,
                    ris_coverage_pct=r.coverage_pct, shadow_coverage_pct=r.shadow_coverage_pct,
                    noris_shadow_mean_snr_db=r.noris_shadow_mean_snr_db, ris_shadow_mean_snr_db=r.shadow_mean_snr_db,
                    p5_snr_db=r.p5_snr_db))
tbl = pd.DataFrame(tbl)
tbl.to_csv(os.path.join(HERE, "summary_best_placement_A.csv"), index=False)
print(tbl.round(2).to_string())

# ---------------------------------------------------------------- 8. maps
gs = (len(A.GX), len(A.GY))
ext = [A.GX[0] - 2.5, A.GX[-1] + 2.5, A.GY[0] - 2.5, A.GY[-1] + 2.5]


def decorate(ax, ris=None):
    for b in A.BLD:
        ax.add_patch(Rectangle((b["x"][0], b["y"][0]), b["x"][1] - b["x"][0], b["y"][1] - b["y"][0],
                               fc="0.25", ec="k", alpha=0.85, zorder=3))
    ax.plot(A.BS[0], A.BS[1], "k^", ms=9, zorder=5)
    ax.text(A.BS[0] + 1, A.BS[1] + 4, "BS", fontsize=8)
    if ris is not None:
        c = A.SITES[ris]
        ax.plot(c[0], c[1], "r*", ms=13, mec="w", zorder=5)
    ax.set_xlim(-5, ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def grid(v):  # [U] -> [Y,X] image
    return v.reshape(gs).T


vmin, vmax = 0, 50
fig, axs = plt.subplots(1, 5, figsize=(22, 4.6), constrained_layout=True)
im = axs[0].imshow(grid(snr0), origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap="viridis")
axs[0].set_title("No RIS"); decorate(axs[0])
for ax, N in zip(axs[1:], A.NS):
    s = cio.snr_db_from_file(cio.load_channels(os.path.join(HERE, f"channels_A_N{N}.npz")))
    ax.imshow(grid(s), origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(f"RIS N={N} ({site}, az {az}°, tilt {tl}°)"); decorate(ax, site)
fig.colorbar(im, ax=axs, label="SNR (dB)", shrink=0.85)
fig.savefig(os.path.join(OUT, "snr_maps_best_placement.png"), dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(22, 4.6), constrained_layout=True)
cm = matplotlib.colors.ListedColormap(["#d95f02", "#1b9e77"])
for ax, (lab, s) in zip(axs, [("No RIS", snr0)] + [(f"RIS N={N}", cio.snr_db_from_file(
        cio.load_channels(os.path.join(HERE, f"channels_A_N{N}.npz")))) for N in A.NS]):
    ax.imshow(grid((s >= A.THR_DB).astype(float)), origin="lower", extent=ext, cmap=cm, vmin=0, vmax=1)
    ax.set_title(f"{lab}: coverage {100 * (s >= A.THR_DB).mean():.1f}% (SNR≥{A.THR_DB:.0f} dB)")
    decorate(ax, site if lab != "No RIS" else None)
fig.savefig(os.path.join(OUT, "coverage_maps_best_placement.png"), dpi=160); plt.close(fig)

# gain map for the largest N
N = A.NS[-1]
s = cio.snr_db_from_file(cio.load_channels(os.path.join(HERE, f"channels_A_N{N}.npz")))
fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
im = ax.imshow(grid(s - snr0), origin="lower", extent=ext, cmap="magma")
decorate(ax, site); ax.set_title(f"SNR gain from RIS, N={N} (dB)"); fig.colorbar(im, ax=ax, label="dB")
fig.savefig(os.path.join(OUT, "snr_gain_map_N16384.png"), dpi=160); plt.close(fig)

# site x orientation heatmap (coverage averaged over N) + coverage / mean-SNR vs N per site
piv = (df.assign(orient=lambda x: x.az_deg.astype(str) + "°/" + x.downtilt_deg.astype(str) + "°")
         .groupby(["site", "orient"]).coverage_pct.mean().unstack())
order = [f"{a}°/{t}°" for a in A.AZS for t in A.TILTS]
piv = piv[order]
fig, ax = plt.subplots(figsize=(12, 3.8), constrained_layout=True)
im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=60, fontsize=8)
ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center", color="w", fontsize=7)
ax.set_xlabel("orientation (azimuth/downtilt)"); ax.set_title("Coverage (%) averaged over N, by site and orientation")
fig.colorbar(im, ax=ax, label="coverage %")
fig.savefig(os.path.join(OUT, "placement_heatmap.png"), dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for sid in A.SITES:
    sub = df[df.site == sid].groupby("N")[["coverage_pct", "mean_snr_db"]].max()
    axs[0].plot(sub.index, sub.coverage_pct, "o-", label=sid)
    axs[1].plot(sub.index, sub.mean_snr_db, "o-", label=sid)
axs[0].axhline(base_cov, color="k", ls="--", label="no RIS"); axs[1].axhline(df.noris_mean_snr_db.iloc[0], color="k", ls="--")
axs[0].set_ylabel("coverage % (best orientation)"); axs[1].set_ylabel("mean SNR dB (best orientation)")
for ax in axs:
    ax.set_xscale("log", base=2); ax.set_xlabel("N"); ax.set_xticks(A.NS); ax.set_xticklabels(A.NS); ax.grid(alpha=.3)
axs[0].legend(ncol=2, fontsize=8)
fig.savefig(os.path.join(OUT, "coverage_snr_vs_N_per_site.png"), dpi=160); plt.close(fig)
print("maps written")

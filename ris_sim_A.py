"""Phase 1 (A): ideal RIS model - channels, optimal phases, SNR/coverage sweep, export.

Run:  python ris_sim_A.py            (reads params.json, writes everything next to it)
"""
import json, itertools, time, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
C0 = 299792458.0


def db2lin(x):
    return 10 ** (x / 10)


# ----------------------------------------------------------------- setup
P = json.load(open(os.path.join(HERE, "params.json")))
F = P["rf"]["carrier_freq_hz"]
LAM = C0 / F
K = 2 * np.pi / LAM
BW = P["rf"]["bandwidth_hz"]
PT_W = db2lin(P["rf"]["tx_power_dbm"]) * 1e-3
NOISE_DBM = P["rf"]["noise_psd_dbm_per_hz"] + 10 * np.log10(BW) + P["rf"]["noise_figure_db"]
SIGMA2 = db2lin(NOISE_DBM) * 1e-3
GT = db2lin(P["rf"]["bs_antenna_gain_dbi"])
GR = db2lin(P["rf"]["ue_antenna_gain_dbi"])
GE = db2lin(P["ris"]["element_gain_dbi"])
DSP = P["ris"]["element_spacing_wavelengths"] * LAM
BLK_AMP = 10 ** (-P["scene"]["blockage_penalty_db"] / 20)
THR_DB = P["metrics"]["coverage_snr_threshold_db"]
BS = np.array(P["scene"]["bs_position"], float)
BLD = P["scene"]["buildings"]
SITES = {s["id"]: np.array(s["position"], float) for s in P["ris_candidate_sites"]}
AZS = P["ris_orientations"]["azimuth_deg_of_normal"]
TILTS = P["ris_orientations"]["downtilt_deg"]
NS = P["ris"]["N_values"]

g_ = P["scene"]["ue_grid"]
GX = np.arange(g_["x_min"], g_["x_max"] + 1e-9, g_["step_m"])
GY = np.arange(g_["y_min"], g_["y_max"] + 1e-9, g_["step_m"])
XX, YY = np.meshgrid(GX, GY, indexing="ij")
UE = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, g_["z"])], 1)
U = len(UE)


# ----------------------------------------------------------------- geometry
def blocked(p0, p1):
    """Segment p0 -> p1 (p1: [M,3]) crosses any building box? returns bool [M]. Slab method."""
    p1 = np.atleast_2d(p1)
    d = p1 - p0
    res = np.zeros(len(p1), bool)
    for b in BLD:
        lo = np.array([b["x"][0], b["y"][0], 0.0])
        hi = np.array([b["x"][1], b["y"][1], b["height_m"]])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - p0) / d
            t2 = (hi - p0) / d
        par = np.abs(d) < 1e-12
        inside = (p0 >= lo) & (p0 <= hi)
        tmin = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        tmax = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        te = np.max(tmin, axis=1)
        tl = np.min(tmax, axis=1)
        res |= (np.maximum(te, 0) <= np.minimum(tl, 1)) & (te <= tl)
    return res


def ris_frame(az_deg, tilt_deg):
    az, tl = np.radians(az_deg), np.radians(tilt_deg)
    n = np.array([np.cos(tl) * np.cos(az), np.cos(tl) * np.sin(az), -np.sin(tl)])
    u = np.array([-np.sin(az), np.cos(az), 0.0])
    v = np.cross(n, u)
    return n, u, v


def elements(center, n, u, v, N):
    M = int(round(np.sqrt(N)))
    idx = (np.arange(M) - (M - 1) / 2) * DSP
    a, b = np.meshgrid(idx, idx, indexing="ij")
    return center + a.reshape(-1, 1) * u + b.reshape(-1, 1) * v  # [N,3]


# ----------------------------------------------------------------- channels
def direct_channel():
    d = np.linalg.norm(UE - BS, axis=1)
    amp = np.sqrt(GT * GR) * LAM / (4 * np.pi * d)
    amp = amp * np.where(blocked(BS, UE), BLK_AMP, 1.0)
    return amp * np.exp(-1j * K * d)


def hr_channel(pos, n, center):
    """BS -> RIS element channel [N]."""
    vec = BS - pos
    d1 = np.linalg.norm(vec, axis=1)
    cosi = np.clip(vec @ n / d1, 0, None)
    blk = BLK_AMP if blocked(BS, center[None])[0] else 1.0
    return np.sqrt(GT) * np.sqrt(GE * cosi) * LAM / (4 * np.pi * d1) * blk * np.exp(-1j * K * d1)


def g_channel(pos, n, center, ue):
    """RIS element -> UE channel [Uc, N] (complex128)."""
    vec = ue[:, None, :] - pos[None, :, :]
    d2 = np.linalg.norm(vec, axis=2)
    coso = np.clip(np.einsum("uni,i->un", vec, n) / d2, 0, None)
    blk = np.where(blocked(center, ue), BLK_AMP, 1.0)[:, None]
    return np.sqrt(GR) * np.sqrt(GE * coso) * LAM / (4 * np.pi * d2) * blk * np.exp(-1j * K * d2)


def g_amp(pos, n, center, ue):
    """|g| only (fast path for the sweep)."""
    vec = ue[:, None, :] - pos[None, :, :]
    d2 = np.linalg.norm(vec, axis=2)
    coso = np.clip(np.einsum("uni,i->un", vec, n) / d2, 0, None)
    blk = np.where(blocked(center, ue), BLK_AMP, 1.0)[:, None]
    return np.sqrt(GR) * np.sqrt(GE * coso) * LAM / (4 * np.pi * d2) * blk


def snr_db_from_lin(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


def sweep():
    hd = direct_channel()
    abs_hd = np.abs(hd)
    snr0 = snr_db_from_lin(PT_W * abs_hd ** 2 / SIGMA2)
    shadow = blocked(BS, UE)
    rows = []
    t0 = time.time()
    for sid, c in SITES.items():
        for az, tl in itertools.product(AZS, TILTS):
            n, u, v = ris_frame(az, tl)
            for N in NS:
                pos = elements(c, n, u, v, N)
                hr_abs = np.abs(hr_channel(pos, n, c))
                acc = np.empty(U)
                for i in range(0, U, 49):
                    s = slice(i, i + 49)
                    acc[s] = (g_amp(pos, n, c, UE[s]) * hr_abs[None, :]).sum(1)
                # optimal phases make every term add in phase with hd
                snr = snr_db_from_lin(PT_W * (abs_hd + acc) ** 2 / SIGMA2)
                gain = snr - snr0
                rows.append(dict(
                    site=sid, az_deg=az, downtilt_deg=tl, N=N,
                    mean_snr_db=snr.mean(), p5_snr_db=np.percentile(snr, 5), min_snr_db=snr.min(),
                    coverage_pct=100 * (snr >= THR_DB).mean(),
                    noris_mean_snr_db=snr0.mean(), noris_coverage_pct=100 * (snr0 >= THR_DB).mean(),
                    mean_gain_db=gain.mean(), min_gain_db=gain.min(),
                    shadow_ues=int(shadow.sum()),
                    shadow_coverage_pct=100 * (snr[shadow] >= THR_DB).mean(),
                    shadow_mean_snr_db=snr[shadow].mean(),
                    noris_shadow_mean_snr_db=snr0[shadow].mean(),
                    bs_ris_blocked=bool(blocked(BS, c[None])[0]),
                ))
        print(f"  site {sid} done  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), hd, snr0, shadow


def export_channels(site, az, tl, N, hd, snr0, outdir):
    c = SITES[site]
    n, u, v = ris_frame(az, tl)
    pos = elements(c, n, u, v, N)
    hr = hr_channel(pos, n, c)
    g = g_channel(pos, n, c, UE)
    # ideal SNR via explicit phases (used for the check as well)
    th = -np.angle(hr[None, :] * g) + np.angle(hd)[:, None]
    h = hd + np.sum(hr[None, :] * np.exp(1j * th) * g, axis=1)
    ideal = snr_db_from_lin(PT_W * np.abs(h) ** 2 / SIGMA2)
    fn = os.path.join(outdir, f"channels_A_N{N}.npz")
    np.savez_compressed(
        fn, hd=hd.astype(np.complex64), hr=hr.astype(np.complex64), g=g.astype(np.complex64),
        ue_xyz=UE.astype(np.float32), elem_xyz=pos.astype(np.float32),
        ris_center=c.astype(np.float32), ris_normal=n.astype(np.float32),
        tx_power_w=PT_W, noise_power_w=SIGMA2, snr_threshold_db=THR_DB,
        ideal_snr_db=ideal.astype(np.float32), noris_snr_db=snr0.astype(np.float32),
        grid_shape=np.array([len(GX), len(GY)]), site=site, az_deg=az, downtilt_deg=tl,
        carrier_freq_hz=F)
    return ideal


if __name__ == "__main__":
    print(f"lambda={LAM*1e3:.3f} mm  noise={NOISE_DBM:.1f} dBm  Pt={PT_W:.1f} W  U={U}")
    df, hd, snr0, shadow = sweep()
    df.to_csv(os.path.join(HERE, "results_A.csv"), index=False)
    print(df.groupby("N")[["mean_snr_db", "coverage_pct"]].max())

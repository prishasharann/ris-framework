"""Shared helpers for reading/using channel files (A, B, C, D all use this).

File keys are documented in params.json -> channel_file_format.
"""
import numpy as np


def load_channels(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def ideal_phase(hd, hr, g):
    """theta[u,n] = -arg(hr[n]*g[u,n]) + arg(hd[u])  -> shape [U,N]."""
    return -np.angle(hr[None, :] * g) + np.angle(hd)[:, None]


def snr_linear(hd, hr, g, theta, tx_power_w, noise_power_w):
    """gamma[u] = Pt |hd[u] + sum_n hr[n] e^{j theta[u,n]} g[u,n]|^2 / sigma^2"""
    h = hd + np.sum(hr[None, :] * np.exp(1j * theta) * g, axis=1)
    return tx_power_w * np.abs(h) ** 2 / noise_power_w


def snr_db(hd, hr, g, theta, tx_power_w, noise_power_w, chunk=64):
    out = np.empty(hd.shape[0])
    for i in range(0, hd.shape[0], chunk):
        s = slice(i, i + chunk)
        out[s] = snr_linear(hd[s], hr, g[s], theta[s], tx_power_w, noise_power_w)
    return 10 * np.log10(out)


def snr_db_from_file(d, theta=None, chunk=64):
    """Convenience: SNR (dB) for a loaded channel dict; ideal phases if theta is None."""
    hd, hr, g = d["hd"], d["hr"], d["g"]
    Pt, s2 = float(d["tx_power_w"]), float(d["noise_power_w"])
    out = np.empty(hd.shape[0])
    for i in range(0, hd.shape[0], chunk):
        s = slice(i, i + chunk)
        th = ideal_phase(hd[s], hr, g[s]) if theta is None else theta[s]
        out[s] = 10 * np.log10(snr_linear(hd[s], hr, g[s], th, Pt, s2))
    return out

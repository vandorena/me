"""Forward model and scorer: for a grid of candidate listener positions and one aircraft track, predict the
Doppler curve each position would have heard and measure how much whitened energy lies on it.

The model is colsto's, as implemented in colsto-leak-and-mask's ridge.py: the frequency heard at receive
time t_r comes from the aircraft's radial velocity at the *emission* time t_e, where t_r = t_e + |r(t_e)-P|/c.
The ratio is c / (c + v_r). The curve is scored for the best unknown source frequency f0 (with x2 and x3
harmonics) and the same frames scored as a flat line (ratio 1) are subtracted, so stationary hum carries
no weight.

Because the visitor's clock is known to well under a second, the latency scan that dominated the webcam
runs shrinks to a few seconds here.

Every function takes the spectrogram band it scores in: propellers and rotors in LOW, jets' fan tones
in HIGH (see spectro.py). band_for() picks by aircraft kind."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .adsb import Track
from .spectro import HIGH, LOW, Band

C_AIR = 343.0  # m/s at ~20 C
MIC_AGL = 5.0  # assumed listener height above the DEM, m
MAX_SLANT = {"prop": 8000.0, "heli": 6000.0, "unknown": 7000.0, "jet": 4500.0}
MIN_CURVE_S = 4.0
MIN_COVERAGE = 0.6
MIN_SWING = 0.005
BLOCK_FRAMES = 256  # _shifted_sum works this many frames at a time, to bound memory


def band_for(kind: str) -> tuple[Band, ...]:
    """Where an aircraft's tone lives. An unknown type is tried in both."""
    if kind == "jet":
        return (HIGH,)
    if kind == "unknown":
        return (LOW, HIGH)
    return (LOW,)


def half_window(d: np.ndarray, v: np.ndarray, band: Band) -> np.ndarray:
    return np.clip(2 * d / np.maximum(v, 10.0), 4.0, band.max_half_window_s)


@dataclass
class Scored:
    score: np.ndarray  # (n,) curve minus flat-line score; NaN where the point cannot be scored
    f0: np.ndarray  # (n,) best source frequency, Hz
    harmonics: np.ndarray  # (n,) 1 when a single tone won, 3 when the harmonic stack won
    t_cpa: np.ndarray  # (n,) CPA receive time, UTC


def cpa(track: Track, points: np.ndarray, c: float = C_AIR):
    """Closest approach of the track to each point: (receive time, slant distance, speed)."""
    n = len(points)
    t_r, d, v = np.full(n, np.nan), np.full(n, np.inf), np.full(n, np.nan)
    step = max(1, 2_000_000 // max(1, len(track.t)))
    for a in range(0, n, step):
        diff = track.pos[None, :, :] - points[a:a + step, None, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        j = dist.argmin(1)
        d[a:a + step] = dist[np.arange(len(j)), j]
        t_r[a:a + step] = track.t[j] + d[a:a + step] / c
        v[a:a + step] = np.linalg.norm(track.vel[j], axis=1)
    return t_r, d, v


def predict_ratio(track: Track, points: np.ndarray, t_receive: np.ndarray, c: float = C_AIR) -> np.ndarray:
    """Doppler ratio heard at each (point, receive time); NaN where the track does not cover t_e."""
    t_e = t_receive
    for _ in range(3):  # fixed point; the error shrinks by v_r/c (~0.2) per step
        p = track.at(t_e, velocity=False)
        t_e = t_receive - np.linalg.norm(p - points[:, None, :], axis=-1) / c
    p, vel = track.at(t_e)
    rel = p - points[:, None, :]
    dist = np.linalg.norm(rel, axis=-1)
    v_r = (rel * vel).sum(-1) / np.maximum(dist, 1.0)
    return c / (c + v_r)


def score_track(track: Track, points: np.ndarray, W: np.ndarray, frame_t: np.ndarray, band: Band = LOW,
                latency: float = 0.0, reverse: bool = False, c: float = C_AIR) -> Scored:
    """Scores one aircraft at every point, in one band's whitened spectrogram W. `latency` is added to the
    audio's UTC stamps; `reverse` scores the audio played backwards (a control: no real flyover sounds
    like a rising sweep)."""
    n = len(points)
    out = Scored(np.full(n, np.nan), np.full(n, np.nan), np.zeros(n), np.full(n, np.nan))
    min_frames = int(MIN_CURVE_S / band.frame_s)
    if n == 0 or len(frame_t) < min_frames:
        return out
    t_cpa, d_cpa, v = cpa(track, points, c)
    out.t_cpa = t_cpa
    limit = MAX_SLANT.get(track.kind, 7000.0)
    hw = half_window(d_cpa, v, band)
    ft = frame_t + latency
    ok = (d_cpa <= limit) & (t_cpa + hw >= ft[0]) & (t_cpa - hw <= ft[-1])
    if not ok.any():
        return out
    idx = np.nonzero(ok)[0]
    lo = np.searchsorted(ft, (t_cpa[idx] - hw[idx]).min())
    hi = np.searchsorted(ft, (t_cpa[idx] + hw[idx]).max(), side="right")
    tu = ft[lo:hi]
    Wu = W[lo:hi][::-1] if reverse else W[lo:hi]
    if len(tu) < min_frames:
        return out

    P = points[idx]
    in_win = np.abs(tu[None, :] - t_cpa[idx, None]) <= hw[idx, None]
    # The ratio changes on a d/v >= ~4 s scale, so predict it about once a second and interpolate.
    step = max(1, int(round(1.0 / band.frame_s)))
    coarse = np.arange(0, len(tu) + step, step).clip(max=len(tu) - 1)
    rho_c = predict_ratio(track, P, np.broadcast_to(tu[coarse], (len(idx), len(coarse))), c)
    j = np.minimum(np.arange(len(tu)) // step, len(coarse) - 2)
    w = (np.arange(len(tu)) - coarse[j]) / np.maximum(coarse[j + 1] - coarse[j], 1)
    rho = rho_c[:, j] * (1 - w) + rho_c[:, j + 1] * w
    valid = in_win & np.isfinite(rho)
    count = valid.sum(1)
    expected = 2 * hw[idx] / band.frame_s
    r = np.where(valid, rho, np.nan)
    with np.errstate(all="ignore"):
        swing = np.nanmax(r, 1) - np.nanmin(r, 1)
    keep = (count >= min_frames) & (count / expected >= MIN_COVERAGE) & (swing >= MIN_SWING)
    if not keep.any():
        return out
    idx, valid, count = idx[keep], valid[keep], count[keep]
    dk = np.where(valid, np.rint(np.log(np.where(valid, rho[keep], 1.0)) / band.dlog), 0).astype(np.int64)

    curve = _shifted_sum(Wu, dk, valid) / count[:, None]
    flat = (valid.astype(np.float32) @ Wu) / count[:, None]
    c_best, c_f0, c_h = _best(curve, band)
    f_best, _, _ = _best(flat, band)
    out.score[idx] = c_best - f_best
    out.f0[idx] = c_f0
    out.harmonics[idx] = c_h
    return out


def _shifted_sum(Wu: np.ndarray, dk: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """R[n, k] = sum_t W[t, k + dk[n, t]] over valid frames, as sparse products over blocks of frames.
    Columns of the dense side are (frame, shift) pairs: X[t*D + j, k] = W[t, k + d_j]."""
    T, K = Wu.shape
    R = np.zeros((valid.shape[0], K), np.float32)
    for a in range(0, T, BLOCK_FRAMES):
        b = min(T, a + BLOCK_FRAMES)
        v = valid[:, a:b]
        if not v.any():
            continue
        d_vals = np.unique(dk[:, a:b][v])
        D = len(d_vals)
        X = np.zeros((b - a, D, K), np.float32)
        for j, d in enumerate(d_vals):
            if d >= 0:
                X[:, j, : K - d] = Wu[a:b, d:]
            else:
                X[:, j, -d:] = Wu[a:b, : K + d]
        rows, cols = np.nonzero(v)
        j = np.searchsorted(d_vals, dk[:, a:b][rows, cols])
        A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols * D + j)),
                              shape=(valid.shape[0], (b - a) * D))
        R += A @ X.reshape((b - a) * D, K)
    return R


def _best(R: np.ndarray, band: Band):
    """Best over f0 of (a) the harmonic stack's mean and (b) a single tone anywhere, discounted."""
    K = band.k
    stack = sum(R[:, band.k0 + s] for s in band.hshift if (band.k0 + s).max() < K) / len(band.hshift)
    single = 0.7 * R
    i_stack, i_single = stack.argmax(1), single.argmax(1)
    v_stack = stack[np.arange(len(R)), i_stack]
    v_single = single[np.arange(len(R)), i_single]
    use_single = v_single > v_stack
    best = np.where(use_single, v_single, v_stack)
    f0 = np.where(use_single, band.fk[i_single], band.fk[band.k0[i_stack]])
    return best, f0, np.where(use_single, 1, len(band.hshift))


def erase_curve(W: np.ndarray, track: Track, point: np.ndarray, f0: float, frame_t: np.ndarray,
                band: Band = LOW, width: int = 3, c: float = C_AIR) -> None:
    """Zeroes the whitened energy along one aircraft's predicted curve and its harmonics, in place."""
    t_cpa, d, v = cpa(track, point[None, :], c)
    hw = float(half_window(d, v, band)[0]) * 1.5
    frames = np.nonzero(np.abs(frame_t - t_cpa[0]) <= hw)[0]
    if len(frames) == 0:
        return
    rho = predict_ratio(track, point[None, :], frame_t[frames][None, :], c)[0]
    for h in (*band.harmonics, band.harmonics[-1] + 1):
        with np.errstate(invalid="ignore"):
            k = np.rint(np.log(f0 * h * rho / band.fk[0]) / band.dlog)
        for frame, kk in zip(frames, k):
            if np.isfinite(kk):
                W[frame, max(0, int(kk) - width): max(0, int(kk) + width + 1)] = 0.0


def predicted_curve(track: Track, point: np.ndarray, f0: float, frame_t: np.ndarray, t_cpa: float,
                    band: Band = LOW, latency: float = 0.0, c: float = C_AIR,
                    max_points: int = 160) -> list[list[float]]:
    """[UTC ms, Hz] pairs of the fundamental at one point, for drawing over the visitor's spectrogram."""
    d = np.linalg.norm(track.pos - point, axis=1).min()
    v = max(10.0, float(np.linalg.norm(track.vel, axis=1).mean()))
    hw = float(half_window(np.array([d]), np.array([v]), band)[0]) * 1.5
    t = np.linspace(t_cpa - hw, t_cpa + hw, max_points)
    t = t[(t >= frame_t[0] + latency) & (t <= frame_t[-1] + latency)]
    if len(t) == 0:
        return []
    rho = predict_ratio(track, point[None, :], t[None, :], c)[0]
    return [[round((ti - latency) * 1000), round(float(f0 * r), 2)] for ti, r in zip(t, rho) if np.isfinite(r)]

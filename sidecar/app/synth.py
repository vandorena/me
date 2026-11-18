"""Synthetic flyovers with a known answer: an ADS-B-like track that passes a chosen listener, and the audio
that listener would record (exact emission-time Doppler, 1/r level, a laptop mic's low-frequency roll-off,
room noise and optional mains hum). Used by the tests and by the dev-only "test flight" mode.

A propeller is a low blade-pass tone with harmonics (f0 ~ 80-200 Hz). A jet is a fan-tone stack in the
kHz range (render(..., **JET)) under broadband roar that swells and fades with the same 1/r envelope."""

from __future__ import annotations

import math

import numpy as np

from .adsb import KT, Sample, Tile, Track, hermite
from .doppler import C_AIR
from .geo import Frame
from .spectro import FS

JET = {"f0": 1800.0, "harmonic_amps": (1.0, 0.5, 0.3), "roar_db": 6.0}


def straight_flight(frame: Frame, through_xy: tuple[float, float], heading_deg: float, speed_ms: float,
                    alt_m: float, t_cpa: float, offset_m: float, duration_s: float = 600.0,
                    poll_s: float = 2.0, jitter_m: float = 5.0,
                    rng: np.random.Generator | None = None) -> list[Sample]:
    """A straight, level flight whose closest point (horizontal) to `through_xy` is `offset_m` to its right
    at UTC `t_cpa`. Positions are sampled every `poll_s` like a live poller, with `jitter_m` of noise
    (render audio from a jitter-free copy: the real aircraft flies smoothly, only its reports are noisy)."""
    rng = rng or np.random.default_rng(0)
    h = math.radians(heading_deg)
    u = np.array([math.sin(h), math.cos(h)])  # along track
    right = np.array([u[1], -u[0]])
    p_cpa = np.array(through_xy) - right * offset_m  # the listener sits `offset_m` to the aircraft's right
    out = []
    for t in np.arange(t_cpa - duration_s / 2, t_cpa + duration_s / 2, poll_s):
        xy = p_cpa + u * speed_ms * (t - t_cpa) + rng.normal(0, jitter_m, 2)
        lat, lon = frame.to_latlon(xy[0], xy[1])
        out.append(Sample(t=float(t), lat=float(lat), lon=float(lon), alt_m=alt_m, vx=float(u[0] * speed_ms),
                          vy=float(u[1] * speed_ms), vz=0.0))
    return out


def add_to_tile(tile: Tile, hex_code: str, samples: list[Sample], callsign: str = "TEST01",
                type_code: str = "PC12", kind: str = "prop") -> None:
    for s in samples:
        tile.add({"hex": hex_code, "callsign": callsign, "type_code": type_code, "kind": kind}, s)


def track_from_samples(samples: list[Sample], frame: Frame, kind: str = "prop") -> Track:
    ts = np.array([s.t for s in samples])
    x, y = frame.to_xy(np.array([s.lat for s in samples]), np.array([s.lon for s in samples]))
    p = np.column_stack([x, y, [s.alt_m for s in samples]])
    v = np.array([[s.vx, s.vy, s.vz] for s in samples])
    grid = np.arange(math.ceil(ts[0]), math.floor(ts[-1]) + 1.0)
    pos, vel = hermite(ts, p, v, grid)
    return Track("synth", "TEST01", "PC12", kind, grid, pos, vel)


def render(track: Track, listener: np.ndarray, t_start: float, duration_s: float, f0: float = 110.0,
           harmonic_amps=(1.0, 0.6, 0.35), snr_db: float = 0.0, rolloff_hz: float = 120.0,
           mains_hz: float | None = None, mains_db: float = -20.0, noise_seed: int = 1,
           level: float = 0.2, roar_db: float | None = None, tones: bool = True) -> np.ndarray:
    """Float32 audio at FS as recorded at `listener` (x, y, z in the track's frame) from UTC `t_start`.
    `snr_db` is the tone's power at closest approach relative to the room's broadband noise. `roar_db`
    adds the aircraft's own broadband noise at that level relative to the tones (a jet's roar);
    `tones=False` leaves only the roar."""
    n = int(duration_s * FS)
    t_r = t_start + np.arange(n) / FS
    t_e = t_r.copy()
    for _ in range(4):
        p = track.at(t_e, velocity=False)
        d = np.linalg.norm(p - listener, axis=1)
        t_e = t_r - d / C_AIR
    p = track.at(t_e, velocity=False)
    d = np.linalg.norm(p - listener, axis=1)
    covered = np.isfinite(d)
    d = np.where(covered, d, np.inf)
    amp = np.where(covered, np.nanmin(d) / d, 0.0)  # 1 at closest approach
    rng = np.random.default_rng(noise_seed)
    jitter = np.cumsum(rng.normal(0, 2e-4, n)) / FS  # slow engine RPM wander, seconds of phase
    tone = sum(a * np.sin(2 * math.pi * h * f0 * (np.where(covered, t_e, 0) + jitter))
               for h, a in enumerate(harmonic_amps, start=1))
    tone *= amp
    noise = pink(n, rng)
    tone_rms = math.sqrt(sum(a * a for a in harmonic_amps) / 2)
    noise *= tone_rms / (np.std(noise) + 1e-12) * 10 ** (-snr_db / 20)
    x = (tone if tones else 0.0) + noise
    if roar_db is not None:
        roar = rng.normal(0, 1, n)
        x += amp * roar / np.std(roar) * tone_rms * 10 ** (roar_db / 20)
    if mains_hz:
        hum = sum(math.exp(-0.3 * (h - 1)) * np.sin(2 * math.pi * mains_hz * h * t_r) for h in range(1, 6))
        x += hum * tone_rms * 10 ** (mains_db / 20) * 3
    x = highpass(x, rolloff_hz)
    return (x / (np.abs(x).max() + 1e-12) * level).astype(np.float32)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0, 1, n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / FS)
    spec /= np.sqrt(np.maximum(f, 20.0))
    return np.fft.irfft(spec, n)


def highpass(x: np.ndarray, fc: float) -> np.ndarray:
    """Second-order roll-off, like a small MEMS mic below its corner."""
    if not fc:
        return x
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    r = (f / fc) ** 2
    spec *= r / np.sqrt(1 + r * r)
    return np.fft.irfft(spec, len(x))


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def from_pcm16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, "<i2").astype(np.float32) / 32768.0


def heading_speed(kt: float) -> float:
    return kt * KT

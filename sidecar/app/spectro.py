"""Streaming log-frequency spectrograms and their whitened form (after colsto-leak-and-mask's ridge.py).

Audio arrives as 16 kHz mono in 5 s chunks. It is looked at in two bands:
  LOW   45-1400 Hz with ~1 Hz resolution: propeller and rotor hum (blade-pass tones of 50-450 Hz),
  HIGH  500-7500 Hz with short frames: jet fan tones (1.4-7 kHz stacks that sweep fast when close).
Each full STFT frame is computed once; whitening is redone over the whole session each tick (a few
thousand frames by a few hundred bins, so it is cheap).

The raw audio is not kept: AudioBuffer holds only what the next frames still need plus a short tail for
the mains-hum check, so the session's memory holds spectrogram rows, not a recording. Rows are kept
only for the last HISTORY_S, so a hunt left open for hours costs no more per tick than a 20-minute one;
the session freezes passes heard before that (see Session._freeze)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FS = 16000
RIDGE_SIDE = 3  # bins either side subtracted by the ridge contrast
MAX_GAP_S = 30.0
TAIL_S = 20.0  # the mains check's window
HISTORY_S = 20 * 60.0  # spectrogram rows older than this are dropped


@dataclass(frozen=True)
class Band:
    name: str
    fmin: float
    fmax: float
    nperseg: int
    hop: int
    dlog: float  # log-frequency bin: 0.01 = 1 %
    f0_range: tuple[float, float]
    harmonics: tuple[int, ...]
    max_half_window_s: float  # longest stretch of curve scored either side of closest approach
    fk: np.ndarray = field(init=False, repr=False, compare=False)
    k0: np.ndarray = field(init=False, repr=False, compare=False)  # bins a fundamental may sit in
    hshift: tuple[int, ...] = field(init=False, compare=False)  # bin offset of each harmonic

    def __post_init__(self):
        k = int(round(math.log(self.fmax / self.fmin) / self.dlog)) + 1
        fk = self.fmin * np.exp(self.dlog * np.arange(k))
        object.__setattr__(self, "fk", fk)
        object.__setattr__(self, "k0", np.nonzero((fk >= self.f0_range[0]) & (fk <= self.f0_range[1]))[0])
        object.__setattr__(self, "hshift", tuple(int(round(math.log(h) / self.dlog)) for h in self.harmonics))

    @property
    def frame_s(self) -> float:
        return self.hop / FS

    @property
    def k(self) -> int:
        return len(self.fk)


# ~1.02 s frames every 0.25 s: enough to see a 1 % Doppler step at 100 Hz.
LOW = Band("low", 45.0, 1400.0, 16384, 4000, 0.01, (50.0, 450.0), (1, 2, 3), 90.0)
# 0.256 s frames every 0.125 s (ridge.py v1.2: fast jet sweeps need short hops). Jets are heard only
# close, so their curves are short.
HIGH = Band("high", 500.0, 7500.0, 4096, 2000, 0.01, (500.0, 2500.0), (1, 2, 3), 45.0)
BANDS = (LOW, HIGH)


class AudioBuffer:
    """Session audio, tied to UTC. Gaps up to 30 s are filled with silence so sample index stays tied to
    UTC; overlaps (clock jitter) are dropped. Samples are discarded once every band has framed them and
    they have left the mains tail."""

    def __init__(self):
        self.t0: float | None = None  # UTC of sample 0
        self.total = 0  # samples received since t0, including filled gaps
        self.start = 0  # index of data[0]
        self.data = np.zeros(0, np.float32)

    @property
    def duration(self) -> float:
        return self.total / FS

    def append(self, pcm: np.ndarray, t_utc: float) -> None:
        pcm = pcm.astype(np.float32)
        if self.t0 is None:
            self.t0 = t_utc
        else:
            gap = int(round((t_utc - (self.t0 + self.duration)) * FS))
            if 0 < gap <= MAX_GAP_S * FS:
                pcm = np.concatenate([np.zeros(gap, np.float32), pcm])
            elif gap < 0:
                pcm = pcm[min(len(pcm), -gap):]
        self.data = np.concatenate([self.data, pcm])
        self.total += len(pcm)

    def view(self, a: int, b: int) -> np.ndarray:
        return self.data[a - self.start:b - self.start]

    def tail(self, seconds: float) -> np.ndarray:
        return self.data[-int(seconds * FS):].copy()

    def trim(self, keep_from: int) -> None:
        keep_from = min(max(keep_from, self.start), self.total - int(TAIL_S * FS))
        if keep_from > self.start:
            self.data = self.data[keep_from - self.start:]
            self.start = keep_from


class Spectrogram:
    def __init__(self, band: Band):
        self.band = band
        self.window = np.hanning(band.nperseg).astype(np.float32)
        flin = np.fft.rfftfreq(band.nperseg, 1 / FS)
        lo, hi = band.fk * math.exp(-band.dlog / 2), band.fk * math.exp(band.dlog / 2)
        # Each log bin takes the max of the FFT bins inside it, or interpolates when it is narrower than one.
        self.groups = []
        for k, f in enumerate(band.fk):
            sel = np.nonzero((flin >= lo[k]) & (flin < hi[k]))[0]
            if len(sel):
                self.groups.append(("max", sel))
            else:
                j = int(np.searchsorted(flin, f) - 1)
                self.groups.append(("lerp", (j, (f - flin[j]) / (flin[j + 1] - flin[j]))))
        self.rows: list[np.ndarray] = []
        self.first = 0  # frame index of rows[0]
        self.t0: float | None = None

    def frame_times(self) -> np.ndarray:
        """UTC of each frame's centre."""
        return self.t0 + ((self.first + np.arange(len(self.rows))) * self.band.hop + self.band.nperseg / 2) / FS

    def next_sample(self) -> int:
        """The first sample the next frame needs."""
        return (self.first + len(self.rows)) * self.band.hop

    def drop_before(self, t_utc: float) -> None:
        if not self.rows:
            return
        k = int(np.searchsorted(self.frame_times(), t_utc))
        if k:
            del self.rows[:k]
            self.first += k

    def update(self, audio: AudioBuffer) -> None:
        """Computes every frame the buffer now covers."""
        b = self.band
        if audio.t0 is None:
            return
        self.t0 = audio.t0
        n_total = 1 + (audio.total - b.nperseg) // b.hop if audio.total >= b.nperseg else 0
        start = self.first + len(self.rows)
        if n_total <= start:
            return
        a = start * b.hop
        x = audio.view(a, (n_total - 1) * b.hop + b.nperseg)
        idx = np.arange(b.nperseg)[None, :] + b.hop * np.arange(n_total - start)[:, None]
        power = np.abs(np.fft.rfft(x[idx] * self.window, axis=1)) ** 2
        lp = 10 * np.log10(power + 1e-12)
        out = np.empty((len(lp), b.k), np.float32)
        for k, (mode, g) in enumerate(self.groups):
            if mode == "max":
                out[:, k] = lp[:, g].max(axis=1)
            else:
                j, w = g
                out[:, k] = (1 - w) * lp[:, j] + w * lp[:, j + 1]
        self.rows.extend(out)

    def log_power(self) -> np.ndarray:
        return np.array(self.rows, np.float32) if self.rows else np.zeros((0, self.band.k), np.float32)

    def whitened(self) -> np.ndarray:
        return whiten(self.log_power())


class Recording:
    """One session's audio in every band. append() frames new audio and drops what is no longer needed."""

    def __init__(self):
        self.audio = AudioBuffer()
        self.specs = {b.name: Spectrogram(b) for b in BANDS}

    @property
    def duration(self) -> float:
        return self.audio.duration

    def append(self, pcm: np.ndarray, t_utc: float) -> None:
        self.audio.append(pcm, t_utc)
        horizon = self.audio.t0 + self.audio.duration - HISTORY_S
        for spec in self.specs.values():
            spec.update(self.audio)
            spec.drop_before(horizon)
        self.audio.trim(min(spec.next_sample() for spec in self.specs.values()))

    def whitened(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """{band name: (W, frame UTC times)}."""
        out = {}
        for name, spec in self.specs.items():
            W = spec.whitened()
            out[name] = (W, spec.frame_times() if len(W) else np.zeros(0))
        return out


def whiten(lp: np.ndarray) -> np.ndarray:
    """Removes the per-frequency background and per-frame level so a moving tone stands out, and
    stationary hum (fridges, mains, fans) sits near zero."""
    if len(lp) == 0:
        return lp
    out = lp - np.median(lp, axis=0, keepdims=True)
    med = np.median(out, axis=1, keepdims=True)
    mad = np.median(np.abs(out - med), axis=1, keepdims=True) * 1.4826 + 1e-6
    w = np.clip((out - med) / mad, -3, 6)
    # +-1 bin tolerance for model error
    w = np.maximum(w, np.maximum(np.roll(w, 1, axis=1), np.roll(w, -1, axis=1)))
    # Ridge contrast: energy 3 bins to either side counts against a bin. A predicted curve that runs
    # alongside the real ridge (wrong speed) is penalised instead of rewarded for being close; without
    # this, a plane 25 % slower fits another's pass about as well as the plane that made it. It also
    # flattens broadband roar, which is high in every bin at once.
    pad = np.pad(w, ((0, 0), (RIDGE_SIDE, RIDGE_SIDE)), mode="edge")
    w = w - 0.5 * (pad[:, : -2 * RIDGE_SIDE] + pad[:, 2 * RIDGE_SIDE:])
    return (w - w.mean()).astype(np.float32)

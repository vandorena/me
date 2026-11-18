"""One visitor's hunt: their audio, the live tracks around their IP prior, and the solver's best guess.

Each tick (after every 5 s chunk, coalesced) rescores every low aircraft that could have been heard:
  1. a coarse 400 m lattice over the prior, restricted to each aircraft's audible corridor,
  2. per aircraft: is a Doppler curve really there? (peak vs its own corridor, vs the flat-line null
     built into the score, and vs the same audio played backwards: no real pass is a rising sweep),
  3. heard aircraft are combined (each normalised to its own peak), so two passes on different lines
     break the left/right mirror ambiguity of one pass,
  4. the top peaks are refined on a 100 m lattice with a small latency scan.
Propellers and rotors are scored in the LOW band, jets in the HIGH band (their fan tones); events from
either band combine on the same lattice.
Nothing here is written to disk; the audio lives only in this object, and only its last 20 s at that."""

from __future__ import annotations

import base64
import itertools
import math
import threading
import time

import numpy as np

from . import doppler, mains
from .adsb import Tile, Track, TilePoller, build_tracks
from .dem import Dem
from .geo import Frame, haversine_m
from .spectro import HISTORY_S, Band, Recording

COARSE_M = 400.0
FINE_M = 100.0
FINE_SPAN_M = 1000.0
MIN_RADIUS_KM, MAX_RADIUS_KM = 10.0, 100.0
LATENCIES = (-1.0, 0.0, 1.0, 2.0)
MAX_AGL = 3000.0
MIN_AUDIO_S = 15.0

# A pass counts as heard only if all of these hold. Tuned on synthetic passes (tests/test_solver.py):
# a clean pass scores 1.5-3 in either band; room noise alone stays under ~0.3, and a jet's broadband
# roar without fan tones under ~0.1 (the ridge contrast flattens it). Real jets may need HIGH raised.
HEARD = {  # band: (min score, min ratio over the reversed audio, min z over the aircraft's own corridor)
    "low": (0.45, 2.0, 4.0),
    "high": (0.45, 2.0, 4.0),
}
MAX_EVENTS = 4  # events found per tick, and events combined into one answer
# The spectrogram keeps only HISTORY_S of rows. A pass that ended this long ago is frozen (its maps and a
# crop of the spectrogram around it are kept) before it slides out, so it still combines with later ones.
FREEZE_AFTER_S = HISTORY_S / 2
RIVAL_FRACTION = 0.8  # another aircraft explaining an event this well is a live alternative
MAX_RIVALS = 6
CONFIDENT_RUNNER_UP = 0.8  # runner-up peak (>= 1.5 km away) must score below this fraction of the best
RUNNER_UP_SEPARATION_M = 1500.0
HEATMAP_MAX = 96


class Session:
    def __init__(self, token: str, lat: float, lon: float, radius_km: float, cap_s: float,
                 country_code: str | None, poller: TilePoller, dem: Dem):
        self.token = token
        self.prior = (lat, lon)
        self.radius_m = 1000 * min(MAX_RADIUS_KM, max(MIN_RADIUS_KM, radius_km))
        self.cap_s = cap_s
        self.country_code = country_code
        self.frame = Frame(lat, lon)
        self.poller = poller
        self.tile: Tile = poller.acquire(lat, lon, self.radius_m / 1000)
        self.dem = dem
        self.spec = Recording()
        self.created = self.touched = time.time()
        self.lock = threading.Lock()
        self.tick_lock = threading.Lock()
        self.dirty = False
        self.closed = False
        self.final = False
        self.lattice: np.ndarray | None = None  # (G, 3) coarse points inside the prior, z = ground + mic
        self.frozen: list[list[tuple[Track, dict]]] = []  # events older than the spectrogram's window
        self.state = {"tier": "ip", "status": "waiting for audio", "listened_s": 0.0, "done": False,
                      "guess": None, "confidence": 0.0, "ambiguous": False, "aircraft": [], "heard": [], "events": [],
                      "curves": [], "heatmap": None, "mains": None, "mains_expected": mains.expected_hz(country_code),
                      "tracks": [], "adsb": None, "tick_ms": None}
        threading.Thread(target=self._build_lattice, daemon=True).start()

    # ------------------------------------------------------------------ public

    def append(self, pcm: np.ndarray, t_utc: float) -> None:
        with self.lock:
            if self.closed:
                return
            self.spec.append(pcm, t_utc)
            self.touched = time.time()
            self.dirty = True

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def result(self) -> dict:
        """What Rails needs to score the visitor's pin: our guess, the IP centre and the run's facts."""
        s = self.snapshot()
        guess = s["guess"]
        return {"tier": s["tier"], "guess": guess and {"lat": guess["lat"], "lon": guess["lon"]},
                "prior": {"lat": self.prior[0], "lon": self.prior[1]}, "confidence": s["confidence"],
                "ambiguous": s["ambiguous"],
                "listened_s": s["listened_s"], "cap_s": self.cap_s, "aircraft_scored": len(s["aircraft"]),
                "passes_detected": len(s["events"]), "mains_hz": (s["mains"] or {}).get("hz")}

    def drop_audio(self) -> None:
        """Listening is over: forget the audio but keep the guess for the visitor's pin."""
        with self.lock:
            if self.spec.duration:
                self.spec = Recording()
                self.dirty = False

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.spec = Recording()
        self.poller.release(self.tile)

    def tick(self) -> None:
        """Runs the solver if new audio arrived since the last tick. Never runs twice at once."""
        if not self.tick_lock.acquire(blocking=False):
            return
        try:
            while self.dirty and not self.closed:
                self.dirty = False
                started = time.time()
                update = self._solve()
                update["tick_ms"] = round(1000 * (time.time() - started))
                with self.lock:
                    self.state.update(update)
        finally:
            self.tick_lock.release()

    # ------------------------------------------------------------------ solver

    def _build_lattice(self) -> None:
        r = self.radius_m
        xs = np.arange(-r, r + 1, COARSE_M)
        gx, gy = np.meshgrid(xs, xs)
        inside = gx ** 2 + gy ** 2 <= r ** 2
        x, y = gx[inside], gy[inside]
        lat, lon = self.frame.to_latlon(x, y)
        z = self.dem.elevation(lat, lon) + doppler.MIC_AGL
        self.lattice = np.column_stack([x, y, z])

    def _ground(self, xy: np.ndarray) -> np.ndarray:
        lat, lon = self.frame.to_latlon(xy[:, 0], xy[:, 1])
        return self.dem.elevation(lat, lon) + doppler.MIC_AGL

    def _solve(self) -> dict:
        with self.lock:
            bands = self.spec.whitened()
            audio_tail = self.spec.audio.tail(mains.WINDOW_S)
            listened = self.spec.duration
        out = {"listened_s": round(listened, 1), "mains": mains.detect(audio_tail)}
        out["adsb"] = {"source": self.tile.source, "error": self.tile.error,
                       "age_s": round(time.time() - self.tile.last_ok, 1) if self.tile.last_ok else None}
        times = [ft for _, ft in bands.values() if len(ft)]
        if listened < MIN_AUDIO_S or self.lattice is None or not times:
            out["status"] = "listening"
            out.update(self._decide([], [], out["mains"], listened))
            return out

        span = np.array([min(ft[0] for ft in times), max(ft[-1] for ft in times)])  # the audio, in UTC
        tracks = [t for t in build_tracks(self.tile.snapshot(), self.frame, span[0] - 60, span[-1] + 5)
                  if self._candidate(t, span)]
        out["tracks"] = [self._polyline(t, span) for t in tracks]
        scores = {}
        for track in tracks:
            results = [self._score_aircraft(track, band, *bands[band.name], span)
                       for band in doppler.band_for(track.kind)]
            results = [r for r in results if r is not None]
            if results:
                scores[track.hex] = (track, max(results, key=lambda r: r["peak"]))
        events = self._strongest(self.frozen + self._freeze(self._find_events(scores, bands, span), span))
        out["aircraft"] = sorted((r["summary"] for _, r in scores.values()), key=lambda a: -a["peak"])[:12]
        out.update(self._decide(list(scores.values()), events, out["mains"], listened))
        return out

    def _find_events(self, scores: dict, bands: dict, span: np.ndarray) -> list[list[tuple[Track, dict]]]:
        """Splits what was heard into events (one tone each) and, per event, every aircraft that could
        have made it. The pitch slide fixes only speed, closest distance and closest-approach time, so any
        plane at a similar speed has some spot on the ground where it would have sounded the same.

        Greedy: confirm the strongest candidate (ranked by a refined peak, so lattice luck does not pick
        it), erase its curve and harmonics, and rescore the rest. Whoever collapses was explaining the
        same tone: a rival hypothesis for this event. Whoever keeps its score may be a later event.
        Each band has its own copy of the spectrogram: a jet cannot explain a propeller's tone."""
        Ws = {name: W.copy() for name, (W, _) in bands.items()}
        for event in self.frozen:  # already counted: keep their tones from being found again
            lead = event[0][1]
            point, f0 = lead["erase"]
            doppler.erase_curve(Ws[lead["band"].name], event[0][0], point, f0, bands[lead["band"].name][1],
                                lead["band"])
        events: list[list[tuple[Track, dict]]] = []
        # A plane frozen for a pass is not heard again for that pass (a circling plane can pass again).
        frozen_at = {}
        for event in self.frozen:
            for t, r in event:
                times = self._pass_times(r)
                frozen_at.setdefault(t.hex, []).append((times.min() - 60, times.max() + 60))
        done: set[str] = set()
        for hex_code, (_, r) in scores.items():
            times = self._pass_times(r)
            if len(times) and any(lo <= times.max() and times.min() <= hi for lo, hi in frozen_at.get(hex_code, ())):
                done.add(hex_code)
        for _ in range(MAX_EVENTS):
            cands = sorted(((t, r) for t, r in scores.values()
                            if t.hex not in done and r["peak"] >= HEARD[r["band"].name][0]
                            and r["summary"]["z"] >= HEARD[r["band"].name][2]),
                           key=lambda c: -c[1]["peak"])
            if not cands:
                break
            for t, r in cands:
                if r["peak"] >= RIVAL_FRACTION * cands[0][1]["peak"] and "fine" not in r:
                    r["fine"] = self._fine_peak(t, r, Ws[r["band"].name])
            track, lead = max(cands, key=lambda c: c[1].get("fine", c[1]["peak"]))
            band, ft = lead["band"], lead["ft"]
            W = Ws[band.name]
            # The control costs as much as the score, so it runs only for the contender.
            rev = doppler.score_track(track, self.lattice[lead["idx"]], W, ft, band, reverse=True)
            reverse_peak = float(np.nanmax(rev.score)) if np.isfinite(rev.score).any() else 0.0
            lead["summary"]["reverse"] = round(reverse_peak, 3)
            done.add(track.hex)
            if lead["peak"] < HEARD[band.name][1] * max(reverse_peak, 0.05):
                continue
            k = len(self.frozen) + len(events)
            lead["W"] = W.copy()  # what it was judged on: later steps must not see other events' tones
            lead["summary"].update(heard=True, event=k, rival=False)
            event = [(track, lead)]
            i = int(np.nanargmax(lead["score"]))
            lead["erase"] = (self.lattice[lead["idx"][i]].copy(), float(lead["f0"][i]))
            doppler.erase_curve(W, track, *lead["erase"], ft, band)
            bar = RIVAL_FRACTION * lead.get("fine", lead["peak"])
            for hex_code, (t, r) in list(scores.items()):
                if hex_code in done or r["band"] is not band or r["peak"] < HEARD[band.name][0]:
                    continue
                again = self._score_aircraft(t, band, W, ft, span)
                collapsed = again is None or again["peak"] < 0.5 * r["peak"]
                if collapsed and r.get("fine", r["peak"]) >= bar:
                    r["W"] = lead["W"]
                    r["summary"].update(heard=True, event=k, rival=True)
                    event.append((t, r))
                    done.add(hex_code)
                elif again is not None:
                    scores[hex_code] = (t, again)
            events.append(sorted(event, key=lambda h: -h[1].get("fine", h[1]["peak"]))[:MAX_RIVALS])
        return events

    def _freeze(self, events: list, span: np.ndarray) -> list:
        """Moves events whose pass ended FREEZE_AFTER_S ago into self.frozen, keeping only the part of the
        spectrogram around each hypothesis's pass. Returns the events still live."""
        live = []
        for event in events:
            lead = event[0][1]
            pad = 1.5 * lead["band"].max_half_window_s + 5.0
            if self._pass_times(lead).max() + pad >= span[1] - FREEZE_AFTER_S:
                live.append(event)
                continue
            for _, r in event:
                t = self._pass_times(r)
                keep = (r["ft"] >= t.min() - pad) & (r["ft"] <= t.max() + pad)
                r["W"], r["ft"] = r["W"][keep], r["ft"][keep]
            self.frozen.append(event)
        return live

    @staticmethod
    def _pass_times(r: dict) -> np.ndarray:
        """Closest-approach times at the spots that fit this hypothesis (the ones its map counts)."""
        good = np.nan_to_num(r["score"], nan=-np.inf) >= r["med"] + 0.5 * (r["peak"] - r["med"])
        return r["t_cpa"][good]

    @staticmethod
    def _strongest(events: list) -> list:
        """At most MAX_EVENTS events, the most clearly heard ones, in the order they were heard."""
        if len(events) <= MAX_EVENTS:
            return events
        strength = lambda ev: ev[0][1].get("fine", ev[0][1]["peak"])  # noqa: E731
        keep = sorted(range(len(events)), key=lambda i: -strength(events[i]))[:MAX_EVENTS]
        return [events[i] for i in sorted(keep)]

    def _fine_peak(self, track: Track, res: dict, W: np.ndarray) -> float:
        i = int(np.nanargmax(res["score"]))
        c = self.lattice[res["idx"][i], :2]
        offs = np.arange(-2 * FINE_M * 2, 2 * FINE_M * 2 + 1, FINE_M)
        gx, gy = np.meshgrid(c[0] + offs, c[1] + offs)
        xy = np.column_stack([gx.ravel(), gy.ravel()])
        s = doppler.score_track(track, np.column_stack([xy, self._ground(xy)]), W, res["ft"], res["band"]).score
        return float(np.nanmax(s)) if np.isfinite(s).any() else res["peak"]

    def _candidate(self, t: Track, ft: np.ndarray) -> bool:
        sel = (t.t >= ft[0] - 40) & (t.t <= ft[-1])
        if not sel.any():
            return False
        p = t.pos[sel]
        near = np.hypot(p[:, 0], p[:, 1]) <= self.radius_m + doppler.MAX_SLANT.get(t.kind, 7000)
        return bool(near.any() and p[:, 2].min() < MAX_AGL + 1500)

    def _corridor(self, track: Track, ft: np.ndarray) -> np.ndarray:
        sel = (track.t >= ft[0] - 60) & (track.t <= ft[-1])
        p = track.pos[sel][::5]
        reach = doppler.MAX_SLANT.get(track.kind, 7000)
        lat = self.lattice
        box = ((lat[:, 0] >= p[:, 0].min() - reach) & (lat[:, 0] <= p[:, 0].max() + reach)
               & (lat[:, 1] >= p[:, 1].min() - reach) & (lat[:, 1] <= p[:, 1].max() + reach))
        cand = np.nonzero(box)[0]
        if len(cand) == 0:
            return cand
        keep = np.zeros(len(cand), bool)
        for a in range(0, len(cand), 4000):
            q = lat[cand[a:a + 4000]]
            d = np.sqrt(((q[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(1)
            keep[a:a + 4000] = d <= reach
        return cand[keep]

    def _score_aircraft(self, track: Track, band: Band, W: np.ndarray, ft: np.ndarray,
                        span: np.ndarray) -> dict | None:
        if len(ft) == 0:
            return None
        idx = self._corridor(track, span)
        if len(idx) == 0:
            return None
        pts = self.lattice[idx]
        r = doppler.score_track(track, pts, W, ft, band)
        good = np.isfinite(r.score)
        if good.sum() < 5:
            return None
        s = r.score[good]
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)) * 1.4826 + 1e-3)
        i = int(np.nanargmax(r.score))
        peak = float(r.score[i])
        z = (peak - med) / mad
        lat, lon = self.frame.to_latlon(pts[i, 0], pts[i, 1])
        summary = {"hex": track.hex, "callsign": track.callsign, "type": track.type_code, "kind": track.kind,
                   "band": band.name, "peak": round(peak, 3),
                   "reverse": None, "z": round(z, 1), "heard": False,
                   "f0": round(float(r.f0[i]), 1), "points": int(good.sum()),
                   "cpa_utc_ms": round(float(r.t_cpa[i]) * 1000)}
        return {"summary": summary, "idx": idx, "score": r.score, "f0": r.f0, "t_cpa": r.t_cpa,
                "band": band, "ft": ft, "peak": peak, "med": med, "heard": False, "best": (float(lat), float(lon))}

    def _decide(self, scored: list, events: list, mains_result: dict | None, listened: float) -> dict:
        update = {"heard": [r["summary"] for ev in events for _, r in ev], "curves": [], "heatmap": None,
                  "ambiguous": False, "events": [[self._name(r) for _, r in ev] for ev in events]}
        expected = mains.expected_hz(self.country_code)
        if not events:
            update["guess"] = {"lat": self.prior[0], "lon": self.prior[1], "radius_m": self.radius_m}
            update["tier"] = "mains" if mains_result else "ip"
            update["confidence"] = 0.0
            hint = ""
            if mains_result and expected and mains_result["hz"] != expected:
                hint = f" · your walls hum at {mains_result['hz']} Hz, which your IP's country does not"
            update["status"] = (f"no aircraft heard yet; {len(scored)} scored" if scored else "listening") + hint
            update["done"] = listened >= self.cap_s
            update["heatmap"] = self._heatmap(scored, heard_only=False)
            return update

        # Every way of explaining the events, one aircraft each. Only the true combination puts all of
        # its maps on the same spot; with one event, every hypothesis and its mirror stay possible.
        maps = {id(r): self._map(r) for ev in events for _, r in ev}
        combos = []
        for combo in itertools.product(*events):
            total = sum(maps[id(r)] for _, r in combo)
            i = int(total.argmax())
            combos.append((float(total[i]), i, combo, total))
        combos.sort(key=lambda c: -c[0])
        best_val, best_i, combo, total = combos[0]
        best_xy, _ = self._refine(self._peaks(total, 2), list(combo))
        runner = self._runner_up(total, best_xy)
        for val, i, _, _ in combos[1:]:
            if np.hypot(*(self.lattice[i, :2] - best_xy)) >= RUNNER_UP_SEPARATION_M:
                runner = max(runner, val)
                break
        ratio = runner / max(best_val, 1e-9)
        n = len(events)
        agree = best_val >= n - 0.6  # every event's map is high at the answer
        ambiguous = n == 1 or ratio >= CONFIDENT_RUNNER_UP or not agree
        lat, lon = self.frame.to_latlon(best_xy[0], best_xy[1])
        # Error scale: tight along the flight from the clock, looser across it from the curve's steepness.
        radius = 250.0 if not ambiguous else 1500.0
        confidence = 0.0 if n == 1 else float(np.clip((1 - ratio) * 1.5, 0, 1)) * (1.0 if agree else 0.5)
        shown = total if not ambiguous else np.max([maps[k] for k in maps], axis=0)
        update.update({
            "guess": {"lat": round(float(lat), 6), "lon": round(float(lon), 6), "radius_m": radius},
            "tier": "doppler", "confidence": round(confidence, 2), "ambiguous": bool(ambiguous),
            "heatmap": self._heatmap(scored, heard_only=True, combined=shown),
        })
        point = np.array([best_xy[0], best_xy[1], self._ground(np.array([best_xy]))[0]])
        for track, r in combo:
            j = int(np.nanargmin(np.where(np.isfinite(r["score"]),
                                          np.hypot(*(self.lattice[r["idx"], :2] - best_xy).T), np.inf)))
            band = r["band"]
            update["curves"].append({"callsign": self._name(r), "band": band.name,
                                     "harmonics": list(band.harmonics),
                                     "points": doppler.predicted_curve(track, point, float(r["f0"][j]), r["ft"],
                                                                       float(r["t_cpa"][j]), band)})
        names = " then ".join(" or ".join(self._name(r) for _, r in ev) for ev in events)
        if n == 1:
            spots = 2 * len(events[0])
            update["status"] = (f"heard one pass ({names}); that leaves {spots} possible spots, "
                                "waiting for another plane")
            update["done"] = listened >= self.cap_s
        elif ambiguous:
            update["status"] = f"heard {names}; they don't agree on one spot yet"
            update["done"] = listened >= self.cap_s
        else:
            update["status"] = "found you from " + " and ".join(self._name(r) for _, r in combo)
            update["done"] = True
        return update

    def _map(self, r: dict) -> np.ndarray:
        """One hypothesis as a 0..1 map over the lattice, sharpened so only near-peak spots count."""
        m = np.zeros(len(self.lattice))
        s = np.nan_to_num(r["score"], nan=0.0)
        m[r["idx"]] = np.clip((s - r["med"]) / (r["peak"] - r["med"] + 1e-9), 0, 1) ** 2
        return m

    @staticmethod
    def _name(r: dict) -> str:
        return r["summary"]["callsign"] or r["summary"]["hex"]

    def _peaks(self, combined: np.ndarray, n: int) -> list[np.ndarray]:
        order = np.argsort(-combined)
        chosen: list[np.ndarray] = []
        for i in order[:2000]:
            if combined[i] <= 0:
                break
            xy = self.lattice[i, :2]
            if all(np.hypot(*(xy - c)) >= RUNNER_UP_SEPARATION_M for c in chosen):
                chosen.append(xy)
            if len(chosen) == n:
                break
        return chosen

    def _runner_up(self, combined: np.ndarray, best_xy: np.ndarray) -> float:
        far = np.hypot(*(self.lattice[:, :2] - best_xy).T) >= RUNNER_UP_SEPARATION_M
        return float(combined[far].max()) if far.any() else 0.0

    def _refine(self, peaks: list[np.ndarray], heard: list) -> tuple[np.ndarray, float]:
        best_xy, best_val = peaks[0], -np.inf
        offs = np.arange(-FINE_SPAN_M, FINE_SPAN_M + 1, FINE_M)
        for c in peaks:
            gx, gy = np.meshgrid(c[0] + offs, c[1] + offs)
            xy = np.column_stack([gx.ravel(), gy.ravel()])
            pts = np.column_stack([xy, self._ground(xy)])
            total = np.zeros(len(pts))
            for track, r in heard:
                best = np.full(len(pts), -np.inf)
                for L in LATENCIES:  # the visitor's clock is good to ~1 s; scan what is left
                    s = doppler.score_track(track, pts, r["W"], r["ft"], r["band"], latency=L).score
                    best = np.fmax(best, np.nan_to_num(s, nan=-np.inf))
                total += np.clip((best - r["med"]) / (r["peak"] - r["med"] + 1e-9), 0, None) ** 2
            i = int(total.argmax())
            if total[i] > best_val:
                best_val, best_xy = float(total[i]), xy[i]
        return best_xy, best_val

    def _heatmap(self, scored: list, heard_only: bool, combined: np.ndarray | None = None) -> dict | None:
        """A small uint8 raster over the prior's bounding box, for the map overlay."""
        if self.lattice is None:
            return None
        if combined is None:
            combined = np.zeros(len(self.lattice))
            for _, r in scored:
                s = np.nan_to_num(r["score"][...], nan=0.0)
                combined[r["idx"]] = np.maximum(combined[r["idx"]], np.clip(s, 0, None))
        if combined.max() <= 0:
            return None
        n = int(math.ceil(2 * self.radius_m / COARSE_M)) + 1
        grid = np.zeros((n, n))
        ix = np.rint((self.lattice[:, 0] + self.radius_m) / COARSE_M).astype(int)
        iy = np.rint((self.radius_m - self.lattice[:, 1]) / COARSE_M).astype(int)  # row 0 = north
        np.maximum.at(grid, (iy, ix), combined)
        f = int(math.ceil(n / HEATMAP_MAX))
        if f > 1:
            m = int(math.ceil(n / f)) * f
            grid = np.pad(grid, ((0, m - n), (0, m - n))).reshape(m // f, f, m // f, f).max(axis=(1, 3))
        img = np.rint(255 * grid / grid.max()).astype(np.uint8)
        north, west = self.frame.to_latlon(-self.radius_m - COARSE_M / 2, self.radius_m + COARSE_M / 2)
        south, east = self.frame.to_latlon(self.radius_m + COARSE_M / 2, -self.radius_m - COARSE_M / 2)
        return {"width": img.shape[1], "height": img.shape[0], "data": base64.b64encode(img.tobytes()).decode(),
                "bounds": [[float(south), float(west)], [float(north), float(east)]], "heard": heard_only}

    def _polyline(self, t: Track, ft: np.ndarray) -> dict:
        sel = (t.t >= ft[0] - 60) & (t.t <= ft[-1])
        p = t.pos[sel][::10]
        lat, lon = self.frame.to_latlon(p[:, 0], p[:, 1])
        return {"hex": t.hex, "callsign": t.callsign, "type": t.type_code, "kind": t.kind,
                "path": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)],
                "alt_m": round(float(p[:, 2].min())) if len(p) else None}


def distance_m(a: dict, b: dict) -> float:
    return haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])

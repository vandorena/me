import time

import numpy as np
import pytest

from app import mains, session, spectro, synth
from app.adsb import TilePoller, classify, parse_aircraft
from app.dem import Dem
from app.doppler import MIC_AGL
from app.geo import haversine_m
from app.session import Session
from app.spectro import FS

LAT, LON = 41.8268, -71.4025
T0 = 1_800_000_000.0
LISTENER = (2300.0, -1700.0)


def new_session(cap_s=300.0, country="US"):
    s = Session("t", LAT, LON, 25.0, cap_s, country, TilePoller(offline=True), Dem(None, offline=True))
    for _ in range(100):
        if s.lattice is not None:
            break
        time.sleep(0.05)
    return s


def fly(s, hex_code, heading, t_cpa, offset_m, alt_m=600.0, speed=75.0, kind="prop"):
    args = (s.frame, LISTENER, heading, speed, alt_m, t_cpa, offset_m)
    synth.add_to_tile(s.tile, hex_code, synth.straight_flight(*args), callsign=hex_code.upper(), kind=kind)
    return synth.track_from_samples(synth.straight_flight(*args, jitter_m=0), s.frame)


def feed(s, audio, ticks=True):
    for i in range(0, len(audio), 5 * FS):
        s.append(audio[i:i + 5 * FS], T0 + i / FS)
        if ticks:
            s.tick()
    if not ticks:
        s.tick()
    return s.snapshot()


def listener_point():
    return np.array([LISTENER[0], LISTENER[1], MIC_AGL])


def error_m(s, snap):
    lat, lon = s.frame.to_latlon(*LISTENER)
    return haversine_m(snap["guess"]["lat"], snap["guess"]["lon"], float(lat), float(lon))


def test_one_pass_is_heard_but_leaves_a_mirror():
    s = new_session()
    track = fly(s, "a1", heading=70, t_cpa=T0 + 70, offset_m=1200)
    snap = feed(s, synth.render(track, listener_point(), T0, 150, snr_db=0, mains_hz=60))

    assert snap["tier"] == "doppler"
    assert [a["hex"] for a in snap["heard"]] == ["a1"]
    assert snap["ambiguous"]
    assert not snap["done"]
    # The guess is the listener or its mirror image across the flight line (2 x 1200 m away).
    assert min(error_m(s, snap), abs(error_m(s, snap) - 2400)) < 400


def test_two_passes_on_different_lines_find_the_listener():
    s = new_session()
    a = fly(s, "a1", heading=70, t_cpa=T0 + 60, offset_m=1200)
    b = fly(s, "b2", heading=160, t_cpa=T0 + 160, offset_m=-900, speed=60, alt_m=450)
    audio = (synth.render(a, listener_point(), T0, 230, snr_db=0, noise_seed=1)
             + synth.render(b, listener_point(), T0, 230, f0=140, snr_db=0, noise_seed=2))
    snap = feed(s, audio)

    assert {h["hex"] for h in snap["heard"]} == {"a1", "b2"}
    assert not snap["ambiguous"]
    assert snap["done"]
    assert error_m(s, snap) < 300
    assert snap["curves"] and snap["curves"][0]["points"]


def test_look_alike_aircraft_make_one_pass_ambiguous():
    # Seen live: silent aircraft passing at about the same moment and speed can each explain a loud pass
    # from some spot of their own. The pitch slide cannot say which plane it was, so all stay possible.
    s = new_session()
    loud = fly(s, "a1", heading=70, t_cpa=T0 + 60, offset_m=1200, speed=80, alt_m=500)
    for i, (heading, offset) in enumerate(((150, 2500), (20, -3000), (250, 1800))):
        fly(s, f"q{i}", heading=heading, t_cpa=T0 + 55 + 4 * i, offset_m=offset)
    snap = feed(s, synth.render(loud, listener_point(), T0, 140, snr_db=6, rolloff_hz=0, level=0.5))

    assert len(snap["events"]) == 1
    assert "A1" in snap["events"][0] and len(snap["events"][0]) > 1
    assert snap["ambiguous"] and not snap["done"] and snap["confidence"] == 0
    assert " or " in snap["status"]


def test_other_traffic_does_not_steal_a_pass():
    # Silent traffic at other speeds: a helicopter, a fast jet, a slower prop a minute later. (A plane
    # within ~10 % of the source's speed stays a rival: the pitch slide can't tell them apart.)
    s = new_session()
    loud = fly(s, "a1", heading=70, t_cpa=T0 + 60, offset_m=1200, speed=80, alt_m=500)
    fly(s, "heli", heading=150, t_cpa=T0 + 40, offset_m=2000, speed=45, alt_m=300, kind="heli")
    fly(s, "jet", heading=20, t_cpa=T0 + 62, offset_m=-3000, speed=140, alt_m=900, kind="jet")
    fly(s, "late", heading=250, t_cpa=T0 + 120, offset_m=1800, speed=60)
    snap = feed(s, synth.render(loud, listener_point(), T0, 150, snr_db=0))

    assert snap["events"] == [["A1"]]
    assert min(error_m(s, snap), abs(error_m(s, snap) - 2400)) < 400


def test_aircraft_overhead_but_silent_audio_is_not_heard():
    s = new_session()
    fly(s, "a1", heading=70, t_cpa=T0 + 70, offset_m=1200)
    rng = np.random.default_rng(3)
    room = synth.highpass(synth.pink(150 * FS, rng), 120) * 0.05
    snap = feed(s, room.astype(np.float32))

    assert snap["heard"] == []
    assert snap["tier"] == "ip"
    assert snap["guess"]["lat"] == LAT
    assert len(snap["aircraft"]) == 1


def test_a_steady_tone_is_not_mistaken_for_a_flyover():
    s = new_session()
    fly(s, "a1", heading=70, t_cpa=T0 + 70, offset_m=1200)
    t = np.arange(150 * FS) / FS
    fridge = 0.1 * np.sin(2 * np.pi * 117 * t) + 0.01 * np.random.default_rng(4).normal(size=len(t))
    snap = feed(s, fridge.astype(np.float32))

    assert snap["heard"] == []


def test_rising_sweep_fails_the_reverse_control():
    s = new_session()
    track = fly(s, "a1", heading=70, t_cpa=T0 + 70, offset_m=1200)
    audio = synth.render(track, listener_point(), T0, 150, snr_db=0)
    snap = feed(s, audio[::-1].copy())

    assert snap["heard"] == []


def test_cap_ends_the_hunt_without_a_plane():
    s = new_session(cap_s=30)
    rng = np.random.default_rng(5)
    snap = feed(s, (synth.pink(35 * FS, rng) * 0.01).astype(np.float32))
    assert snap["done"]
    assert snap["tier"] == "ip"


@pytest.mark.parametrize("hz", [50, 60])
def test_mains_hum_names_the_grid(hz):
    rng = np.random.default_rng(6)
    t = np.arange(25 * FS) / FS
    hum = sum(0.02 / h * np.sin(2 * np.pi * hz * h * t) for h in range(2, 5))  # no fundamental: laptop mic
    noise = synth.highpass(synth.pink(len(t), rng), 120)
    x = hum + noise / np.std(noise) * 0.02
    found = mains.detect(x.astype(np.float32))
    assert found and found["hz"] == hz
    assert abs(found["fundamental"] - hz) < 0.2


def test_no_hum_no_answer():
    rng = np.random.default_rng(7)
    assert mains.detect(synth.pink(25 * FS, rng).astype(np.float32)) is None


def test_mains_mismatch_is_reported():
    s = new_session(country="DE")
    t = np.arange(30 * FS) / FS
    rng = np.random.default_rng(8)
    x = sum(0.02 / h * np.sin(2 * np.pi * 60 * h * t) for h in range(1, 5)) + synth.pink(len(t), rng) * 0.002
    snap = feed(s, x.astype(np.float32))
    assert snap["tier"] == "mains"
    assert snap["mains"]["hz"] == 60 and snap["mains_expected"] == 50
    assert "60 Hz" in snap["status"]


def test_a_tick_stays_within_budget():
    s = new_session()
    a = fly(s, "a1", heading=70, t_cpa=T0 + 60, offset_m=1200)
    for i, h in enumerate((10, 100, 200, 300)):
        fly(s, f"x{i}", heading=h, t_cpa=T0 + 40 + 30 * i, offset_m=3000 + 500 * i)
    audio = synth.render(a, listener_point(), T0, 120, snr_db=0)
    for i in range(0, len(audio) - 5 * FS, 5 * FS):
        s.append(audio[i:i + 5 * FS], T0 + i / FS)
    started = time.time()
    s.append(audio[-5 * FS:], T0 + (len(audio) - 5 * FS) / FS)
    s.tick()
    assert time.time() - started < 5.0, s.snapshot()["tick_ms"]


def jet(s, hex_code, **kw):
    args = dict(heading=70, t_cpa=T0 + 50, offset_m=900, speed=75, alt_m=450, kind="jet")
    return fly(s, hex_code, **{**args, **kw})


def test_one_jet_pass_is_heard_in_the_high_band_but_leaves_a_mirror():
    s = new_session()
    track = jet(s, "j1")
    snap = feed(s, synth.render(track, listener_point(), T0, 120, snr_db=0, **synth.JET))

    assert [(a["hex"], a["band"]) for a in snap["heard"]] == [("j1", "high")]
    assert snap["ambiguous"]
    assert min(error_m(s, snap), abs(error_m(s, snap) - 1800)) < 300


def test_two_jet_passes_find_the_listener():
    s = new_session()
    a = jet(s, "j1")
    b = jet(s, "j2", heading=160, t_cpa=T0 + 130, offset_m=-700, speed=70, alt_m=350)
    audio = (synth.render(a, listener_point(), T0, 180, snr_db=0, noise_seed=1, **synth.JET)
             + synth.render(b, listener_point(), T0, 180, snr_db=0, noise_seed=2, **{**synth.JET, "f0": 2100.0}))
    snap = feed(s, audio)

    assert {h["hex"] for h in snap["heard"]} == {"j1", "j2"}
    assert not snap["ambiguous"] and snap["done"]
    assert error_m(s, snap) < 300
    assert {c["band"] for c in snap["curves"]} == {"high"}


def test_a_jet_and_a_prop_combine_across_bands():
    s = new_session()
    a = jet(s, "j1")
    b = fly(s, "p1", heading=160, t_cpa=T0 + 140, offset_m=-900, speed=60, alt_m=450)
    audio = (synth.render(a, listener_point(), T0, 200, snr_db=0, noise_seed=1, **synth.JET)
             + synth.render(b, listener_point(), T0, 200, f0=140, snr_db=0, noise_seed=2))
    snap = feed(s, audio)

    assert {(h["hex"], h["band"]) for h in snap["heard"]} == {("j1", "high"), ("p1", "low")}
    assert not snap["ambiguous"]
    assert error_m(s, snap) < 300


def test_jet_roar_without_fan_tones_is_not_heard():
    s = new_session()
    track = jet(s, "j1")
    snap = feed(s, synth.render(track, listener_point(), T0, 120, snr_db=0, tones=False, roar_db=10))

    assert snap["heard"] == []
    assert snap["tier"] == "ip"


def test_fan_tones_under_loud_roar_are_still_heard():
    s = new_session()
    track = jet(s, "j1")
    snap = feed(s, synth.render(track, listener_point(), T0, 120, snr_db=-6, **{**synth.JET, "roar_db": 12.0}))

    assert [a["hex"] for a in snap["heard"]] == ["j1"]


def test_a_tick_under_busy_jet_traffic_stays_within_budget():
    s = new_session()
    a = jet(s, "j1", t_cpa=T0 + 60)
    for i, h in enumerate((10, 100, 200, 300, 40)):
        jet(s, f"x{i}", heading=h, t_cpa=T0 + 30 + 20 * i, offset_m=2000 + 400 * i, speed=110, alt_m=900)
    for i, h in enumerate((130, 230)):
        fly(s, f"p{i}", heading=h, t_cpa=T0 + 50 + 40 * i, offset_m=3000)
    audio = synth.render(a, listener_point(), T0, 120, snr_db=0, **synth.JET)
    for i in range(0, len(audio) - 5 * FS, 5 * FS):
        s.append(audio[i:i + 5 * FS], T0 + i / FS)
    started = time.time()
    s.append(audio[-5 * FS:], T0 + (len(audio) - 5 * FS) / FS)
    s.tick()
    assert time.time() - started < 5.0, s.snapshot()["tick_ms"]
    assert [h["hex"] for h in s.snapshot()["heard"]] == ["j1"]


def test_the_session_holds_spectrogram_rows_not_a_recording():
    s = new_session()
    rng = np.random.default_rng(9)
    feed(s, (synth.pink(120 * FS, rng) * 0.01).astype(np.float32), ticks=False)

    assert s.spec.duration == 120
    assert len(s.spec.audio.data) <= (20 + 5) * FS  # the mains tail plus at most one chunk


def test_a_pass_older_than_the_window_still_combines_with_a_later_one(monkeypatch):
    # A hunt runs until the visitor stops it, so the spectrogram keeps only a window (here 8 min).
    monkeypatch.setattr(spectro, "HISTORY_S", 480.0)
    monkeypatch.setattr(session, "FREEZE_AFTER_S", 240.0)
    s = new_session(cap_s=3600)
    a = fly(s, "a1", heading=70, t_cpa=T0 + 60, offset_m=1200)
    b = fly(s, "b2", heading=160, t_cpa=T0 + 540, offset_m=-900, speed=60, alt_m=450)
    audio = (synth.render(a, listener_point(), T0, 600, snr_db=0, noise_seed=1)
             + synth.render(b, listener_point(), T0, 600, f0=140, snr_db=0, noise_seed=2))
    for i in range(0, len(audio), 5 * FS):
        s.append(audio[i:i + 5 * FS], T0 + i / FS)
        if (i // (5 * FS)) % 4 == 3:
            s.tick()
    s.tick()
    snap = s.snapshot()

    assert s.spec.specs["low"].frame_times()[0] > T0 + 100  # pass a1's audio is gone
    assert len(s.spec.specs["high"].rows) <= 480 / spectro.HIGH.frame_s + 1  # memory stays bounded
    assert [r["summary"]["hex"] for ev in s.frozen for _, r in ev][:1] == ["a1"]
    assert {h["hex"] for h in snap["heard"]} >= {"a1", "b2"}
    assert not snap["ambiguous"]
    assert error_m(s, snap) < 300


def test_adsb_records_are_parsed_and_classified():
    ac = {"hex": "A1B2C3", "flight": "N123AB  ", "t": "PC12", "category": "A1", "alt_baro": 3000,
          "alt_geom": 3100, "gs": 200, "track": 90, "lat": 41.9, "lon": -71.5, "seen_pos": 1.5}
    ident, sample = parse_aircraft(ac, 1000.0)
    assert ident == {"hex": "a1b2c3", "callsign": "N123AB", "type_code": "PC12", "kind": "prop"}
    assert sample.t == 998.5 and abs(sample.alt_m - 3100 * 0.3048) < 1e-6
    assert abs(sample.vx - 200 * 0.514444) < 1e-3 and abs(sample.vy) < 1e-6
    assert parse_aircraft({**ac, "alt_baro": "ground"}, 1000.0) is None
    assert parse_aircraft({**ac, "gs": 10}, 1000.0) is None
    assert classify("A3", "B738") == "jet" and classify("A7", None) == "heli" and classify(None, "R44") == "heli"

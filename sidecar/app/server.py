"""HTTP API for the Rails app. Only Rails talks to this service (internal network plus a shared key).

  POST   /sessions                     {lat, lon, radius_km, cap_s, country_code} -> {token}
  POST   /sessions/{token}/chunk       body: little-endian int16 PCM, 16 kHz mono; ?t0_utc_ms= -> snapshot
  GET    /sessions/{token}             -> snapshot
  GET    /sessions/{token}/result      -> what Rails needs to score the visitor's pin
  DELETE /sessions/{token}
  POST   /sessions/{token}/test_flight?kind=prop|jet (SOLVER_DEV=1 only) a synthetic aircraft + its audio"""

from __future__ import annotations

import base64
import logging
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import synth
from .adsb import TilePoller
from .dem import Dem
from .doppler import MIC_AGL
from .session import Session
from .spectro import FS

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

KEY = os.environ.get("SOLVER_KEY", "")
DEV = os.environ.get("SOLVER_DEV") == "1"
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "8"))
IDLE_S = 30.0  # no chunk for this long: listening is over, so the audio is dropped
RESULT_TTL_S = 600.0  # the guess waits this long for the visitor's pin
MAX_CHUNK_BYTES = 2 * 10 * FS  # 10 s of int16

app = FastAPI(title="me solver", docs_url=None, redoc_url=None)
poller = TilePoller(offline=os.environ.get("SOLVER_OFFLINE") == "1")
dem = Dem(os.environ.get("DEM_CACHE", "/data/dem"), offline=os.environ.get("SOLVER_OFFLINE") == "1")
pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SOLVER_THREADS", "4")))
sessions: dict[str, Session] = {}
lock = threading.Lock()


def check_key(key: str | None) -> None:
    if KEY and not secrets.compare_digest(key or "", KEY):
        raise HTTPException(401, "bad key")


def get(token: str) -> Session:
    s = sessions.get(token)
    if s is None or s.closed:
        raise HTTPException(404, "no such session")
    return s


class Start(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0, le=20000)
    cap_s: float = Field(ge=30, le=900)
    country_code: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "sessions": len(sessions), "tiles": len(poller.tiles)}


@app.post("/sessions")
def start(body: Start, x_solver_key: str | None = Header(None)):
    check_key(x_solver_key)
    with lock:
        if len(sessions) >= MAX_SESSIONS:
            raise HTTPException(503, "busy")
        token = secrets.token_urlsafe(16)
        sessions[token] = Session(token, body.lat, body.lon, body.radius_km, body.cap_s, body.country_code,
                                  poller, dem)
    return {"token": token}


@app.post("/sessions/{token}/chunk")
async def chunk(token: str, t0_utc_ms: float, request: Request, x_solver_key: str | None = Header(None)):
    check_key(x_solver_key)
    s = get(token)
    body = await request.body()
    if not body or len(body) > MAX_CHUNK_BYTES or len(body) % 2:
        raise HTTPException(422, "expected up to 10 s of int16 PCM")
    if abs(t0_utc_ms / 1000 - time.time()) > 120:
        raise HTTPException(422, "chunk timestamp is too far from now")
    s.append(synth.from_pcm16(body), t0_utc_ms / 1000)
    pool.submit(s.tick)
    return s.snapshot()


@app.get("/sessions/{token}")
def show(token: str, x_solver_key: str | None = Header(None)):
    check_key(x_solver_key)
    return get(token).snapshot()


@app.get("/sessions/{token}/result")
def result(token: str, x_solver_key: str | None = Header(None)):
    check_key(x_solver_key)
    return get(token).result()


@app.delete("/sessions/{token}")
def stop(token: str, x_solver_key: str | None = Header(None)):
    check_key(x_solver_key)
    with lock:
        s = sessions.pop(token, None)
    if s:
        s.close()
    return {"ok": True}


@app.post("/sessions/{token}/test_flight")
def test_flight(token: str, kind: str = "prop", x_solver_key: str | None = Header(None)):
    """Dev only: puts a fake PC-12 (or, with kind=jet, an A320 on approach) into this session's tile on a
    line that passes a random spot near the prior, and returns the audio a listener there would hear. The browser plays it through its speakers
    and records it back, so the whole mic -> solver path is exercised with a known answer."""
    check_key(x_solver_key)
    if not DEV:
        raise HTTPException(404, "not found")
    s = get(token)
    rng = np.random.default_rng()
    r = min(s.radius_m * 0.5, 8000)
    ang = rng.uniform(0, 2 * np.pi)
    spot = (float(r * np.cos(ang) * rng.uniform(0.2, 1)), float(r * np.sin(ang) * rng.uniform(0.2, 1)))
    start_at = time.time() + 3
    t_cpa = start_at + 45
    is_jet = kind == "jet"
    ground = float(s._ground(np.array([spot]))[0])
    if is_jet:  # ~140 kt on approach, low and close: where jets are heard
        flight = dict(speed_ms=72.0, alt_m=ground + float(rng.uniform(300, 600)), offset_m=float(rng.uniform(500, 1500)))
        ident, sound = dict(callsign="TESTJET", type_code="A320", kind="jet"), synth.JET
    else:
        flight = dict(speed_ms=80.0, alt_m=ground + 500, offset_m=float(rng.uniform(600, 2000)))
        ident, sound = dict(callsign="TEST01", type_code="PC12", kind="prop"), {}
    flight.update(heading_deg=float(rng.uniform(0, 360)), t_cpa=t_cpa, duration_s=400)
    hex_code = f"t{secrets.token_hex(2)}"
    synth.add_to_tile(s.tile, hex_code, synth.straight_flight(s.frame, spot, **flight), **ident)
    track = synth.track_from_samples(synth.straight_flight(s.frame, spot, jitter_m=0, **flight), s.frame)
    listener = np.array([spot[0], spot[1], ground + MIC_AGL])
    audio = synth.render(track, listener, start_at, 100.0, snr_db=6.0, rolloff_hz=0, level=0.5, **sound)
    lat, lon = s.frame.to_latlon(spot[0], spot[1])
    return {"start_utc_ms": round(start_at * 1000), "sample_rate": FS,
            "pcm16": base64.b64encode(synth.to_pcm16(audio)).decode(), "callsign": ident["callsign"],
            "answer": {"lat": float(lat), "lon": float(lon)}}


def sweep() -> None:
    while True:
        time.sleep(10)
        now = time.time()
        with lock:
            for t, s in list(sessions.items()):
                if now - s.touched > RESULT_TTL_S or now - s.created > s.cap_s + RESULT_TTL_S:
                    sessions.pop(t).close()
                elif now - s.touched > IDLE_S:
                    s.drop_audio()


threading.Thread(target=sweep, daemon=True).start()

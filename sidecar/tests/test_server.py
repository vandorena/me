import base64
import os
import time

os.environ["SOLVER_OFFLINE"] = "1"
os.environ["SOLVER_DEV"] = "1"
os.environ["SOLVER_KEY"] = "k"

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import server, synth  # noqa: E402
from app.spectro import FS  # noqa: E402

H = {"X-Solver-Key": "k"}
client = TestClient(server.app)


def start(**kw):
    body = {"lat": 41.8268, "lon": -71.4025, "radius_km": 30, "cap_s": 60, "country_code": "US", **kw}
    r = client.post("/sessions", json=body, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_key_is_required():
    assert client.post("/sessions", json={"lat": 0, "lon": 0, "radius_km": 10, "cap_s": 60}).status_code == 401


def test_a_session_takes_chunks_and_reports_a_result():
    token = start()
    now = time.time()
    pcm = synth.to_pcm16(np.zeros(5 * FS, np.float32))
    for i in range(4):
        r = client.post(f"/sessions/{token}/chunk", params={"t0_utc_ms": (now + 5 * i) * 1000}, content=pcm, headers=H)
        assert r.status_code == 200, r.text
    deadline = time.time() + 10
    while client.get(f"/sessions/{token}", headers=H).json()["listened_s"] < 20 and time.time() < deadline:
        time.sleep(0.1)
    snap = client.get(f"/sessions/{token}", headers=H).json()
    assert snap["listened_s"] == 20.0
    assert snap["tier"] == "ip"

    result = client.get(f"/sessions/{token}/result", headers=H).json()
    assert result["prior"] == {"lat": 41.8268, "lon": -71.4025}
    assert result["guess"] == {"lat": 41.8268, "lon": -71.4025}
    assert result["cap_s"] == 60

    assert client.delete(f"/sessions/{token}", headers=H).status_code == 200
    assert client.get(f"/sessions/{token}", headers=H).status_code == 404


def test_chunks_are_validated():
    token = start()
    now_ms = time.time() * 1000
    assert client.post(f"/sessions/{token}/chunk", params={"t0_utc_ms": now_ms}, content=b"", headers=H).status_code == 422
    assert client.post(f"/sessions/{token}/chunk", params={"t0_utc_ms": now_ms}, content=b"abc", headers=H).status_code == 422
    stale = now_ms - 600_000
    assert client.post(f"/sessions/{token}/chunk", params={"t0_utc_ms": stale}, content=b"\0\0", headers=H).status_code == 422
    assert client.post("/sessions/nope/chunk", params={"t0_utc_ms": now_ms}, content=b"\0\0", headers=H).status_code == 404
    client.delete(f"/sessions/{token}", headers=H)


def test_too_many_sessions_is_busy(monkeypatch):
    monkeypatch.setattr(server, "MAX_SESSIONS", len(server.sessions))
    r = client.post("/sessions", json={"lat": 0, "lon": 0, "radius_km": 10, "cap_s": 60}, headers=H)
    assert r.status_code == 503


def test_dev_test_flight_returns_audio_and_injects_the_aircraft():
    token = start()
    r = client.post(f"/sessions/{token}/test_flight", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert len(base64.b64decode(body["pcm16"])) == 2 * 100 * FS
    assert body["start_utc_ms"] > time.time() * 1000
    s = server.sessions[token]
    assert any(a.callsign == "TEST01" for a in s.tile.snapshot())
    r = client.post(f"/sessions/{token}/test_flight?kind=jet", headers=H)
    assert r.json()["callsign"] == "TESTJET"
    assert any(a.callsign == "TESTJET" and a.kind == "jet" for a in s.tile.snapshot())
    client.delete(f"/sessions/{token}", headers=H)

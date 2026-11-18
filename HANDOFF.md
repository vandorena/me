# Handoff: `audio` branch ("find me by ear")

**State (2026-11-17):** feature-complete and pushed to `origin/audio`. Tests pass: 14 Rails, 19 solver. It has not been tested with a real microphone yet.

## What it is
The visitor grants only the microphone. A Python solver (`sidecar/`) matches aircraft Doppler curves in their audio against live ADS-B data around their IP location, then refines its guess every 5 s. Afterwards the visitor may drop a pin. The `results` table stores two distances only: our guess to the pin, and the IP location to the pin. There is no login, and audio is held in memory only. `main` still has the old portrait app. See `README.md` for how it works and how to run it.

## Key facts learned
- **One pass is always ambiguous.** Any aircraft within about 10% of the real one's speed fits some spot on the ground, and each spot has a mirror image across the flight path. A confident fix needs two events whose candidate spots agree (`session.py: _find_events`, `_decide`).
- **Speed separation comes from the ridge contrast in `spectro.whiten`.** Planes more than about 25% off in speed drop out; within about 10% they stay rivals.
- **Expect mostly IP-only results.** Close passes are rare in a few minutes of listening.
- **No visitor-chosen length.** A hunt runs until the visitor presses stop or two passes agree; `HuntsController::CAP_S` (5 h, hidden) ends one left open. The solver keeps 20 min of spectrogram (`spectro.HISTORY_S`) and freezes older passes (`Session._freeze`), so memory and tick time stay flat however long it runs.
- **Jets are in scope (branch `audio-jets`).** Capture is 16 kHz. `spectro.LOW` (45-1400 Hz) scores props and helicopters; `spectro.HIGH` (500-7500 Hz, 0.125 s hop) scores jets' fan tones. The heard thresholds are the same in both bands and were tuned only on synthetic jets; real jet roar may need `session.HEARD["high"]` raised. Raw audio is no longer kept beyond a 20 s tail; the session holds spectrogram rows.

## Next steps
1. Test with a real mic in Chrome, Safari and Firefox. Check that `track.getSettings()` shows noise suppression actually off. Try it near a window under general-aviation traffic.
2. Deploy on Coolify with `docker-compose.yml`:
   - set `RAILS_MASTER_KEY` and `SOLVER_KEY`
   - the IP database (DB-IP City Lite) downloads itself on boot; check the log for `geo: fetched`
3. Measure solver tick time on the server hardware. On an M-series Mac a tick with 5 aircraft takes about 2.4 s. The budget is under 5 s per chunk.
4. Possible improvements:
   - negative evidence: spots where ADS-B says a loud, close plane passed but nothing was heard become less likely
   - ENF (mains-frequency) matching against live grid feeds
   - backfilling ADS-B history so passes early in a session can be scored
   - a geoid correction (the DEM uses EGM2008 heights; aircraft altitude is GNSS ellipsoid height)

## Local dev
```sh
cd sidecar && SOLVER_DEV=1 SOLVER_KEY=dev DEM_CACHE=./dem-cache .venv/bin/uvicorn app.server:app --port 8000
SOLVER_KEY=dev ME_DEV_PRIOR="41.8268,-71.4025,30" bin/rails server
```
`SOLVER_DEV=1` enables the "play a test flight" button, which plays a synthetic plane with a known location.

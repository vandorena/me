# find me by ear

A challenge: give this page your microphone and nothing else, and it tries to work out where you are
from the planes flying over you, even ones you can't hear.

It is a live, browser version of [colsto's aircraft-Doppler geolocation](https://www.youtube.com/watch?v=UytYrhYLOKI),
with the solver adapted from the independent replication
[colsto-leak-and-mask](https://github.com/Bodila51/colsto-leak-and-mask).

## How it works

1. A propeller plane hums (blade-pass tones of about 50-450 Hz), and a jet's fan whines (tone stacks of about
   1.4-7 kHz). As it passes you, the tone slides down in pitch. The shape of the slide gives the plane's
   speed, how close it came and the second it was closest.
2. Every transponder-equipped plane broadcasts where it is (ADS-B). The solver polls
   [adsb.lol](https://www.adsb.lol/) around the visitor's IP location.
3. For each low aircraft, it predicts the curve that every spot on the ground would have heard. It uses
   emission-time Doppler and terrain from Copernicus GLO-90, then measures the energy on that curve in the
   visitor's whitened spectrogram: propellers and helicopters in a low band with 1 Hz resolution, jets in a
   high band with short frames, because a close jet's tone sweeps fast. Passes heard in either band combine.
4. One pass cannot say which plane made it. Any plane at about the same speed has a strip of ground where it
   would have sounded the same, and each one leaves a mirror image across its own path. So each heard pass
   becomes an event with rival explanations. The answer is the one combination of planes, one per event,
   whose spots agree. A pass counts as heard only if it also beats the same audio played backwards.
5. If no plane comes close, the mains hum (50 Hz or 60 Hz) still names the grid. That is a nice VPN tell, but
   the guess stays where the IP points.

The browser streams 5 s chunks: 16 kHz mono, with echo cancellation, noise suppression and auto-gain turned
off, each stamped with the server-corrected UTC of its first sample. The solver updates its guess after
every chunk. It listens until the visitor presses stop, and stops early once two passes agree. A hunt left
open ends after 5 hours (not shown on the page). The solver keeps only the last 20 minutes of spectrogram;
a pass heard before that is frozen, so it still combines with later ones.

Honest odds: in the reference repo's blind webcam runs this method mostly missed, and a laptop mic indoors is
worse than a webcam outside. Knowing each sample's time to well under a second helps a lot, but most sessions
hear no usable pass. `/results` keeps score against the IP address.

## Privacy model

- No accounts. The only browser permission asked for is the microphone.
- Audio is held in the solver's memory while the visitor listens and dropped 30 s after the last chunk.
  It is never written to disk.
- The IP location is the search area. It is used in memory and never stored.
- After the reveal, the visitor may drop a pin. Rails uses the pin once to compute two distances: guess to
  pin, and IP to pin. Both are rounded to 10 m and are the only things stored, in the `results` table. Skip
  the pin and nothing is stored.

## Layout

| Path | What |
|---|---|
| `app/controllers/hunts_controller.rb` | Starts a hunt, relays chunks to the solver, scores the pin |
| `app/javascript/lib/mic_capture.js` | AudioWorklet capture, resampling to 16 kHz, UTC stamping |
| `app/javascript/lib/spectrogram.js`, `hunt_map.js` | Live spectrogram with predicted curves; Leaflet map |
| `app/services/solver_client.rb`, `geo_lookup.rb`, `geo_database.rb` | Sidecar client; DB-IP City Lite IP prior and its monthly download |
| `sidecar/app/session.py` | One hunt: events, rivals, pair search, early stop |
| `sidecar/app/doppler.py` | Forward model and sparse curve scoring |
| `sidecar/app/adsb.py`, `dem.py`, `mains.py`, `spectro.py` | Live tracks, terrain, mains hum, spectrogram |
| `sidecar/app/synth.py` | Synthetic flyovers for the tests and dev test flights |

## Running it locally

```sh
# solver
cd sidecar && python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
SOLVER_DEV=1 SOLVER_KEY=dev DEM_CACHE=./dem-cache .venv/bin/uvicorn app.server:app --port 8000

# web (another shell). localhost has no IP location, so give it one: lat,lon,radius_km
bin/setup --skip-server
bin/rails geo:fetch   # optional: the IP database, ~130 MB
SOLVER_KEY=dev ME_DEV_PRIOR="41.8268,-71.4025,30" bin/rails server
```

With `SOLVER_DEV=1`, "play a test flight" puts a fake PC-12 ("play a test jet": an A320 on approach) into the live traffic and plays the sound a
listener at a random nearby spot would hear through your speakers. The mic picks it up, so the whole path is
exercised. One test flight is one event, so expect two mirror-image candidates (more if a real plane at a
similar speed passed at the same moment).

Tests: `bin/rails test` and `cd sidecar && .venv/bin/pytest`.

## Deploying

`docker-compose.yml` runs both services. On Coolify, create a Docker Compose resource and set
`RAILS_MASTER_KEY` and `SOLVER_KEY`. On boot the web container downloads DB-IP's free City Lite database
(no account) into its storage volume, and downloads it again on any boot once it is a month old. Until that
download succeeds, every hunt is refused.

## Credits

Method: colsto. Solver lineage: colsto-leak-and-mask (MIT) and digitalis-p/doppler-locate (MIT). Flight data:
adsb.lol (ODbL 1.0), with adsb.fi as a fallback. IP location:
[IP Geolocation by DB-IP](https://db-ip.com) (CC BY 4.0). Terrain: Copernicus DEM GLO-90 (© DLR e.V. 2010-2014 and
© Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA). Map:
© OpenStreetMap contributors. Leaflet (BSD-2-Clause).

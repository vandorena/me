import { Controller } from "@hotwired/stimulus"
import { MicCapture, RATE } from "lib/mic_capture"
import { Spectrogram } from "lib/spectrogram"
import { HuntMap } from "lib/hunt_map"

const FINAL_POLLS = 8

// The hunt: start a solver session around the visitor's IP prior, stream 5 s chunks of their
// microphone, show what the solver hears, and stop on a confident fix or when time is up. Then ask
// the visitor where they really are.
export default class extends Controller {
  static targets = [
    "intro", "start", "live", "status", "clock", "spectrogram", "overlay", "map",
    "aircraft", "mic", "stop", "testFlight", "reveal", "verdict", "pinPrompt", "submitPin", "score", "error"
  ]
  static values = { url: String }

  connect() {
    this.map = new HuntMap(this.mapTarget)
    this.queue = []
    this.sending = false
    this.snapshot = null
  }

  disconnect() {
    this.teardown()
    if (this.running) fetch(this.urlValue, { method: "DELETE", headers: this.headers(), keepalive: true })
  }

  async start() {
    // Browsers only offer the microphone on HTTPS pages (and localhost); without it nothing else works.
    if (!window.isSecureContext || !navigator.mediaDevices) {
      return this.showError("This page isn't served over HTTPS, and browsers only offer the microphone on secure pages. That's our setup, not yours.")
    }
    this.startTarget.disabled = true
    this.showError(null)
    this.statusTarget.textContent = "asking the solver to watch the sky around you"
    const sentAt = this.clientNow()
    const response = await fetch(this.urlValue, {
      method: "POST",
      headers: { ...this.headers(), "Content-Type": "application/json" },
      body: JSON.stringify({})
    }).catch(() => null)
    const receivedAt = this.clientNow()
    if (!response?.ok) return this.failStart(response)

    const hunt = await response.json()
    this.capS = hunt.cap_s
    this.prior = hunt.prior
    this.running = true
    this.liveTarget.hidden = false
    this.introTarget.hidden = true
    this.map.invalidate()
    this.map.showPrior(hunt.prior)
    this.testFlightTargets.forEach((button) => { button.hidden = !hunt.test_flights })

    this.mic = new MicCapture({
      clockOffsetMs: hunt.server_time_ms - (sentAt + receivedAt) / 2,
      onChunk: (chunk) => this.enqueue(chunk)
    })
    try {
      await this.mic.start()
    } catch (error) {
      this.running = false
      this.teardown()
      fetch(this.urlValue, { method: "DELETE", headers: this.headers() })
      this.liveTarget.hidden = true
      this.introTarget.hidden = false
      this.startTarget.disabled = false
      return this.showError(error.name === "NotAllowedError"
        ? "No microphone, no hunt. That is the only permission this page asks for."
        : `Could not open the microphone: ${error.message}`)
    }
    this.showMic()
    this.spectrogram = new Spectrogram(this.spectrogramTarget, this.overlayTarget)
    this.spectrogram.start(this.mic.analyser, () => this.mic.nowUtc())
    this.startedAt = this.clientNow()
    this.clock = setInterval(() => this.tick(), 500)
    this.statusTarget.textContent = `listening. Your network says ${hunt.prior.place || "somewhere"}, give or take ${hunt.prior.radius_km} km`
  }

  async failStart(response) {
    this.startTarget.disabled = false
    const reasons = {
      no_geo_db: "This server has no IP location database, so it can't place anyone. That's our fault, not yours.",
      private_address: "This server sees a private address instead of yours (a proxy setting on our side), so it can't place you.",
      no_dev_prior: "Localhost has no location. Restart Rails with ME_DEV_PRIOR=\"lat,lon,radius_km\" (see the README) to stand in for your IP.",
      unplaceable: "We can't place your network at all (a VPN or a private address?), so there is nowhere for the planes to narrow down."
    }
    const error = response ? await response.json().then(body => body.error, () => null) : null
    const messages = {
      422: "The server turned the request down (an expired page? try reloading).",
      503: "Too many people are being hunted right now. Try again in a minute.",
      502: "The solver is down. Try again later."
    }
    this.showError(reasons[error] || messages[response?.status] || "Could not start the hunt.")
    this.statusTarget.textContent = ""
  }

  tick() {
    const elapsed = (this.clientNow() - this.startedAt) / 1000
    this.clockTarget.textContent = clock(elapsed)
    if (elapsed >= this.capS + 1) this.finish() // the server's hidden limit, not shown
  }

  enqueue(chunk) {
    if (!this.running) return
    this.queue.push(chunk)
    this.drain()
  }

  async drain() {
    if (this.sending) return
    this.sending = true
    while (this.queue.length) {
      const { pcm, t0 } = this.queue.shift()
      try {
        const response = await fetch(`${this.urlValue}/chunk?t0_utc_ms=${Math.round(t0)}`, {
          method: "POST",
          headers: { ...this.headers(), "Content-Type": "application/octet-stream" },
          body: pcm.buffer
        })
        if (response.status === 410) {
          this.showError("The solver forgot this hunt (it restarted or you were away too long). Reload to start over.")
          this.running = false
          this.teardown()
          break
        }
        if (response.ok) this.render(await response.json())
      } catch {
        this.statusTarget.textContent = "lost the solver for a moment, still recording"
      }
    }
    this.sending = false
  }

  render(snapshot) {
    this.snapshot = snapshot
    this.statusTarget.textContent = snapshot.status
    this.map.update(snapshot)
    this.spectrogram?.setCurves(snapshot.curves)
    const lines = (snapshot.aircraft || []).map((a) => {
      const name = (a.callsign || a.hex).padEnd(9)
      const what = `${a.type || "?"} ${a.kind}`.padEnd(12)
      const verdict = a.heard ? "HEARD" : a.peak > 0.2 ? "maybe" : "no curve"
      return `${name} ${what} score ${a.peak.toFixed(2).padStart(5)}  ${verdict}`
    })
    const adsb = snapshot.adsb
    const feed = adsb?.age_s == null ? "flight data: waiting" : `flight data: ${adsb.source}, ${adsb.age_s}s old`
    const hum = snapshot.mains
      ? `mains hum: ${snapshot.mains.hz} Hz (measured ${snapshot.mains.fundamental} Hz, ${snapshot.mains.harmonics} harmonics)`
      : "mains hum: none heard"
    this.aircraftTarget.textContent = [feed, hum, "", ...(lines.length ? lines : ["no low aircraft near you yet"])].join("\n")
    if (snapshot.done && this.running) this.finish()
  }

  stop() {
    this.finish()
  }

  async finish() {
    if (!this.running) return
    this.running = false
    clearInterval(this.clock)
    const sent = this.mic ? this.mic.sent / RATE : 0
    this.teardown()
    this.stopTarget.disabled = true
    this.testFlightTargets.forEach((button) => { button.hidden = true })
    this.statusTarget.textContent = "thinking about what we heard"
    for (let i = 0; i < FINAL_POLLS && (this.sending || (this.snapshot?.listened_s || 0) < sent - 1); i++) {
      await sleep(1000)
      const response = await fetch(`${this.urlValue}/status`, { headers: this.headers() }).catch(() => null)
      if (response?.ok) this.render(await response.json())
    }
    this.reveal()
  }

  reveal() {
    const s = this.snapshot
    this.revealTarget.hidden = false
    if (!s?.guess) {
      this.verdictTarget.textContent = "We didn't hear enough to guess anything."
      return
    }
    const where = `${s.guess.lat.toFixed(4)}, ${s.guess.lon.toFixed(4)}`
    const heard = (s.heard || []).map((a) => `${a.callsign || a.hex} (${a.type || a.kind})`).join(" and ")
    let text
    if (s.tier === "doppler") {
      text = `We think you are at ${where}, within about ${Math.round(s.guess.radius_m)} m. ` +
        `We heard ${heard} pass and matched the pitch of its engine against where it really flew.`
      if (s.ambiguous) text += " One plane leaves two mirror-image spots on either side of its path, and we had to pick one."
    } else if (s.tier === "mains") {
      text = `No plane came close enough to hear. Your room hums at ${s.mains.hz} Hz, which says which grid you're on, ` +
        `but not where on it. So our guess is just where your network points: ${this.prior.place || where}.`
    } else {
      text = `No plane came close enough to hear, and there was no mains hum. Our guess is just where your network ` +
        `points: ${this.prior.place || where}.`
    }
    this.verdictTarget.textContent = text
    this.map.showGuess(s.guess, true)
    if (this.answer) this.map.showAnswer(this.answer)
    this.map.askForPin((pin) => {
      this.pin = pin
      this.submitPinTarget.disabled = false
    })
  }

  async submitPin() {
    if (!this.pin) return
    this.submitPinTarget.disabled = true
    const response = await fetch(`${this.urlValue}/truth`, {
      method: "POST",
      headers: { ...this.headers(), "Content-Type": "application/json" },
      body: JSON.stringify(this.pin)
    }).catch(() => null)
    if (!response?.ok) {
      this.submitPinTarget.disabled = false
      return this.showError(response?.status === 410 ? "This hunt expired before you dropped the pin." : "Could not save that.")
    }
    const result = await response.json()
    this.map.stopPinning()
    this.map.connectPin(this.pin, this.snapshot.guess)
    this.pinPromptTarget.hidden = true
    this.scoreTarget.hidden = false
    const verdict = result.tier !== "doppler"
      ? "The microphone added nothing this time: that was your IP's guess."
      : result.audio_won ? "The planes beat your IP address." : "Your IP address beat the planes."
    this.scoreTarget.querySelector("pre").textContent =
      `we were      ${km(result.error_m)} off\nIP alone was ${km(result.ip_error_m)} off\n\n${verdict}`
  }

  async skip() {
    await fetch(this.urlValue, { method: "DELETE", headers: this.headers() }).catch(() => null)
    this.map.stopPinning()
    this.pinPromptTarget.hidden = true
    this.verdictTarget.textContent += " (You didn't say where you are, so nothing was saved.)"
  }

  async testFlight({ currentTarget: button, params: { kind } }) {
    button.disabled = true
    const response = await fetch(`${this.urlValue}/test_flight?kind=${kind}`, { method: "POST", headers: this.headers() }).catch(() => null)
    if (!response?.ok) {
      button.disabled = false
      return this.showError("The solver isn't running with SOLVER_DEV=1, so there are no test flights.")
    }
    const flight = await response.json()
    const bytes = Uint8Array.from(atob(flight.pcm16), (c) => c.charCodeAt(0))
    this.mic.playAt(new Int16Array(bytes.buffer), flight.sample_rate, flight.start_utc_ms)
    this.answer = flight.answer
    button.textContent = `${flight.callsign} inbound, turn your volume up`
  }

  showMic() {
    const s = this.mic.settings()
    const flag = (value) => (value === undefined ? "unknown" : value ? "ON" : "off")
    this.micTarget.textContent = [
      s.device,
      `noise suppression ${flag(s.noiseSuppression)} · echo cancellation ${flag(s.echoCancellation)} · auto gain ${flag(s.autoGainControl)}`,
      `${s.sampleRate} Hz, resampled to ${RATE} Hz for the solver`,
      s.noiseSuppression ? "your browser kept noise suppression on, which can erase the faint hum we need" : "",
      s.sampleRate < RATE ? "this microphone runs below 16 kHz, so jets' fan tones are out of reach; only propellers count" : ""
    ].filter(Boolean).join("\n")
  }

  showError(message) {
    this.errorTarget.hidden = !message
    this.errorTarget.textContent = message || ""
  }

  teardown() {
    clearInterval(this.clock)
    this.spectrogram?.stop()
    this.mic?.stop()
  }

  clientNow() {
    return performance.timeOrigin + performance.now()
  }

  headers() {
    return { "Accept": "application/json", "X-CSRF-Token": document.querySelector("meta[name=csrf-token]")?.content }
  }
}

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds))
  const mmss = `${Math.floor(s / 60) % 60}:${String(s % 60).padStart(2, "0")}`
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mmss.padStart(5, "0")}` : mmss
}

function km(meters) {
  return meters >= 10000 ? `${Math.round(meters / 1000)} km` : meters >= 1000 ? `${(meters / 1000).toFixed(1)} km` : `${meters} m`
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

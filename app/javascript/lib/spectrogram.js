// A scrolling spectrogram from propeller hum (~50-450 Hz) up to jets' fan tones (~1.4-7 kHz), whitened
// per frequency so a curve far below hearing still shows, with the solver's predicted Doppler curves
// drawn over it by UTC time.
const FMIN = 40
const FMAX = 8000
const COLUMN_MS = 200
const BACKGROUND_ALPHA = 0.02 // how fast each frequency's background level adapts

export class Spectrogram {
  constructor(canvas, overlay) {
    this.canvas = canvas
    this.overlay = overlay
    this.ctx = canvas.getContext("2d")
    this.octx = overlay.getContext("2d")
    this.curves = []
    this.background = null
    this.lastUtc = null
    this.resize()
  }

  resize() {
    const ratio = window.devicePixelRatio || 1
    for (const c of [this.canvas, this.overlay]) {
      c.width = Math.round(c.clientWidth * ratio)
      c.height = Math.round(c.clientHeight * ratio)
    }
    this.ctx.fillStyle = "#000"
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)
  }

  // Pulls one column from an AnalyserNode stamped with the UTC it was read at.
  start(analyser, nowUtc) {
    this.analyser = analyser
    this.nowUtc = nowUtc
    this.bins = new Float32Array(analyser.frequencyBinCount)
    this.timer = setInterval(() => this.column(), COLUMN_MS)
  }

  stop() {
    clearInterval(this.timer)
  }

  windowSeconds() {
    return (this.canvas.width * COLUMN_MS) / 1000
  }

  y(freq) {
    const h = this.canvas.height
    return h - (Math.log(freq / FMIN) / Math.log(FMAX / FMIN)) * h
  }

  x(utcMs) {
    if (this.lastUtc === null) return -1
    return this.canvas.width - 1 - (this.lastUtc - utcMs) / COLUMN_MS
  }

  column() {
    const { canvas, ctx, analyser } = this
    analyser.getFloatFrequencyData(this.bins)
    const hzPerBin = analyser.context.sampleRate / analyser.fftSize
    const h = canvas.height
    if (!this.background) this.background = new Float32Array(h).fill(NaN)

    ctx.drawImage(canvas, -1, 0)
    const image = ctx.createImageData(1, h)
    for (let row = 0; row < h; row++) {
      const freq = FMIN * Math.pow(FMAX / FMIN, (h - row) / h)
      const db = this.bins[Math.min(this.bins.length - 1, Math.round(freq / hzPerBin))]
      const bg = Number.isNaN(this.background[row]) ? db : this.background[row]
      this.background[row] = bg + BACKGROUND_ALPHA * (db - bg)
      const v = Math.max(0, Math.min(1, (db - bg - 2) / 14))
      const level = Math.round(255 * v * v)
      const i = row * 4
      image.data[i] = image.data[i + 1] = image.data[i + 2] = level
      image.data[i + 3] = 255
    }
    ctx.putImageData(image, canvas.width - 1, 0)
    this.lastUtc = this.nowUtc()
    this.draw()
  }

  // curves: [{ callsign, band: "low" | "high", harmonics: [1, 2, 3], points: [[utcMs, hz], ...] }]
  setCurves(curves) {
    this.curves = curves || []
    this.draw()
  }

  draw() {
    const { overlay, octx } = this
    octx.clearRect(0, 0, overlay.width, overlay.height)
    const ratio = window.devicePixelRatio || 1
    octx.font = `${11 * ratio}px ui-monospace, Menlo, monospace`
    octx.lineWidth = 1.5 * ratio
    octx.setLineDash([4 * ratio, 4 * ratio])
    for (const curve of this.curves) {
      for (const harmonic of curve.harmonics) {
        octx.strokeStyle = harmonic === 1 ? "rgba(255, 176, 0, 0.9)" : "rgba(255, 176, 0, 0.45)"
        octx.beginPath()
        let started = false
        for (const [utc, hz] of curve.points) {
          const x = this.x(utc)
          const y = this.y(hz * harmonic)
          if (x < 0 || y < 0) continue
          if (started) octx.lineTo(x, y)
          else octx.moveTo(x, y)
          started = true
        }
        octx.stroke()
      }
      const visible = curve.points.find(([utc]) => this.x(utc) >= 0)
      if (visible) {
        octx.fillStyle = "#ffb000"
        octx.fillText(curve.callsign, Math.max(4, this.x(visible[0])), this.y(visible[1]) - 6 * ratio)
      }
    }
    octx.setLineDash([])
    octx.fillStyle = "#8a8a8a"
    for (const f of [50, 100, 200, 500, 1000, 2000, 5000]) {
      const label = f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`
      octx.fillText(label, 4 * ratio, Math.min(overlay.height - 4, this.y(f) - 2))
    }
  }
}

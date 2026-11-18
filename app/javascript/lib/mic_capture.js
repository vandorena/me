// Records the microphone with every voice-call "improvement" turned off, resamples it to 16 kHz mono
// (enough for jets' fan tones up to ~7 kHz) and hands out 5 s chunks stamped with the UTC time of their
// first sample.
export const RATE = 16000
export const CHUNK_SAMPLES = 5 * RATE

// Lives in the audio thread: batches 128-frame blocks and stamps each batch with its first frame index.
const WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(4096); this.n = 0; this.first = 0 }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (!ch) return true
    if (this.n + ch.length > this.buf.length) this.flush()
    if (this.n === 0) this.first = currentFrame
    this.buf.set(ch, this.n)
    this.n += ch.length
    if (this.n >= 2048) this.flush()
    return true
  }
  flush() {
    this.port.postMessage({ frame: this.first, samples: this.buf.slice(0, this.n) })
    this.n = 0
  }
}
registerProcessor("tap", Tap)
`

export const CONSTRAINTS = {
  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 }
}

export class MicCapture {
  // clockOffsetMs: server UTC minus this browser's clock. onChunk({ pcm: Int16Array, t0: UTC ms })
  constructor({ clockOffsetMs = 0, onChunk }) {
    this.clockOffsetMs = clockOffsetMs
    this.onChunk = onChunk
    this.pending = new Float32Array(0)
    this.sent = 0
    this.utcFirst = null
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS)
    this.track = this.stream.getAudioTracks()[0]
    this.ctx = new AudioContext()
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }))
    await this.ctx.audioWorklet.addModule(url)
    URL.revokeObjectURL(url)
    this.source = this.ctx.createMediaStreamSource(this.stream)
    this.tap = new AudioWorkletNode(this.ctx, "tap")
    this.tap.port.onmessage = ({ data }) => this.receive(data)
    this.source.connect(this.tap)
    this.resampler = new Resampler(this.ctx.sampleRate, RATE)
    this.analyser = this.ctx.createAnalyser()
    this.analyser.fftSize = 16384
    this.analyser.smoothingTimeConstant = 0
    this.source.connect(this.analyser)
    if (this.ctx.state === "suspended") await this.ctx.resume()
  }

  // What the browser actually applied; some ignore the request to switch processing off.
  settings() {
    const s = this.track?.getSettings?.() || {}
    return {
      echoCancellation: s.echoCancellation,
      noiseSuppression: s.noiseSuppression,
      autoGainControl: s.autoGainControl,
      sampleRate: this.ctx?.sampleRate,
      device: this.track?.label || "microphone"
    }
  }

  // This browser's clock, corrected to the server's.
  nowUtc() {
    return performance.timeOrigin + performance.now() + this.clockOffsetMs
  }

  // UTC of an audio-context time, via the context's own clock mapping when available.
  contextTimeToUtc(seconds) {
    const ts = this.ctx.getOutputTimestamp?.()
    if (ts && ts.performanceTime > 0) {
      return performance.timeOrigin + ts.performanceTime + (seconds - ts.contextTime) * 1000 + this.clockOffsetMs
    }
    return this.nowUtc() - (this.ctx.currentTime - seconds) * 1000
  }

  receive({ frame, samples }) {
    if (this.utcFirst === null) this.utcFirst = this.contextTimeToUtc(frame / this.ctx.sampleRate)
    const out = this.resampler.push(samples)
    const merged = new Float32Array(this.pending.length + out.length)
    merged.set(this.pending)
    merged.set(out, this.pending.length)
    this.pending = merged
    while (this.pending.length >= CHUNK_SAMPLES) {
      const chunk = this.pending.subarray(0, CHUNK_SAMPLES)
      this.onChunk({ pcm: toInt16(chunk), t0: this.utcFirst + (this.sent / RATE) * 1000 })
      this.sent += CHUNK_SAMPLES
      this.pending = this.pending.slice(CHUNK_SAMPLES)
    }
  }

  // Plays int16 audio through the speakers starting at a UTC time (dev test flights).
  playAt(pcm16, sampleRate, startUtcMs) {
    const buffer = this.ctx.createBuffer(1, pcm16.length, sampleRate)
    const data = buffer.getChannelData(0)
    for (let i = 0; i < pcm16.length; i++) data[i] = pcm16[i] / 32768
    const node = this.ctx.createBufferSource()
    node.buffer = buffer
    node.connect(this.ctx.destination)
    node.start(this.ctx.currentTime + Math.max(0, (startUtcMs - this.nowUtc()) / 1000))
    return node
  }

  stop() {
    this.tap?.port.close()
    this.source?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())
    this.ctx?.close()
  }
}

// Windowed-sinc low-pass, then linear interpolation down to the target rate.
export class Resampler {
  constructor(inRate, outRate, taps = 127) {
    this.step = inRate / outRate
    this.taps = lowpass(inRate, outRate * 0.45, taps)
    this.history = new Float32Array(taps - 1)
    this.rest = new Float32Array(0)
    this.pos = 0
  }

  push(x) {
    const taps = this.taps
    const input = new Float32Array(this.history.length + x.length)
    input.set(this.history)
    input.set(x, this.history.length)
    const filtered = new Float32Array(x.length)
    for (let n = 0; n < x.length; n++) {
      let acc = 0
      const end = n + taps.length - 1
      for (let k = 0; k < taps.length; k++) acc += taps[k] * input[end - k]
      filtered[n] = acc
    }
    this.history = input.slice(input.length - this.history.length)

    const s = new Float32Array(this.rest.length + filtered.length)
    s.set(this.rest)
    s.set(filtered, this.rest.length)
    const out = []
    while (this.pos + 1 < s.length) {
      const i = Math.floor(this.pos)
      const f = this.pos - i
      out.push(s[i] * (1 - f) + s[i + 1] * f)
      this.pos += this.step
    }
    const consumed = Math.floor(this.pos)
    this.rest = s.slice(consumed)
    this.pos -= consumed
    return Float32Array.from(out)
  }
}

function lowpass(rate, cutoff, n) {
  const fc = cutoff / rate
  const mid = (n - 1) / 2
  const h = new Float32Array(n)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const x = i - mid
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x)
    const blackman = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (n - 1))
    h[i] = sinc * blackman
    sum += h[i]
  }
  return h.map((v) => v / sum)
}

function toInt16(x) {
  const out = new Int16Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767)))
  return out
}

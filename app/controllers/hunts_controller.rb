# The hunt: the visitor grants the microphone, the browser streams 5 s chunks here, and we relay them to
# the solver sidecar, which tries to place the visitor from aircraft Doppler curves. Afterwards the
# visitor may drop a pin where they really are; we store only how far off we (and their IP) were.
#
# The solver's token sits in the cookie session. Nothing else about the hunt is kept on this side.
class HuntsController < ApplicationController
  MIN_CAP_S = 60
  MAX_CAP_S = 900
  MAX_CHUNK_BYTES = 2 * 16_000 * 10 # 10 s of 16 kHz int16
  MIN_RADIUS_KM = 25

  # Rails tries the last match first, so the general case goes first.
  rescue_from SolverClient::Error, with: :solver_down
  rescue_from SolverClient::Gone, with: :hunt_gone
  rescue_from SolverClient::Busy, with: :solver_busy

  def show
  end

  def create
    prior = ip_prior
    return unplaceable unless prior

    solver.stop(session[:hunt]) if session[:hunt]
    started = Time.current
    session[:hunt] = solver.start(lat: prior[:lat], lon: prior[:lon], radius_km: prior[:radius_km],
                                  cap_s: cap_s, country_code: prior[:country_code])
    render json: {
      server_time_ms: (started.to_f * 1000).round,
      cap_s: cap_s,
      prior: prior.slice(:lat, :lon, :radius_km).merge(place: prior[:place]),
      test_flights: Rails.configuration.x.test_flights
    }
  end

  def chunk
    pcm = request.raw_post
    if pcm.empty? || pcm.bytesize > MAX_CHUNK_BYTES || pcm.bytesize.odd?
      return render(json: { error: "bad chunk" }, status: :unprocessable_content)
    end

    t0 = Float(params[:t0_utc_ms], exception: false)
    return render(json: { error: "bad t0_utc_ms" }, status: :unprocessable_content) unless t0

    render json: solver.chunk(require_hunt, pcm, t0)
  end

  def status
    render json: solver.status(require_hunt)
  end

  # The visitor's pin. Its coordinates are used once, to measure two distances, and never stored.
  def truth
    lat = Float(params[:lat], exception: false)
    lon = Float(params[:lon], exception: false)
    unless lat&.between?(-90, 90) && lon&.between?(-180, 180)
      return render(json: { error: "bad pin" }, status: :unprocessable_content)
    end

    token = require_hunt
    run = solver.result(token)
    pin = { "lat" => lat, "lon" => lon }
    result = Result.create!(
      tier: run["tier"],
      error_m: Result.round_m(distance_m(run["guess"] || run["prior"], pin)),
      ip_error_m: Result.round_m(distance_m(run["prior"], pin)),
      confidence: run["confidence"].to_f,
      ambiguous: run["ambiguous"] ? true : false,
      listened_s: run["listened_s"].to_f.round,
      cap_s: run["cap_s"].to_f.round,
      aircraft_scored: run["aircraft_scored"].to_i,
      passes_detected: run["passes_detected"].to_i,
      mains_hz: run["mains_hz"]
    )
    finish(token)
    render json: { error_m: result.error_m, ip_error_m: result.ip_error_m, tier: result.tier,
                   audio_won: result.audio_won? }
  end

  def destroy
    finish(session[:hunt]) if session[:hunt]
    head :no_content
  end

  def test_flight
    return head(:not_found) unless Rails.configuration.x.test_flights

    render json: solver.test_flight(require_hunt)
  end

  private

  def solver
    SolverClient.instance
  end

  def require_hunt
    session[:hunt] || raise(SolverClient::Gone, "no hunt in this session")
  end

  def finish(token)
    solver.stop(token)
    session.delete(:hunt)
  end

  def cap_s
    params[:cap_s].to_i.clamp(MIN_CAP_S, MAX_CAP_S)
  end

  def ip_prior
    geo = GeoLookup.instance.call(request.remote_ip)
    if geo&.located?
      place = [ geo.city, geo.country ].compact.join(", ").presence
      { lat: geo.latitude, lon: geo.longitude, radius_km: [ geo.accuracy_radius.to_i, MIN_RADIUS_KM ].max,
        country_code: geo.country_code, place: place }
    elsif (dev = Rails.configuration.x.dev_prior)
      { lat: dev[0], lon: dev[1], radius_km: dev[2], country_code: nil, place: "the dev prior (ME_DEV_PRIOR)" }
    end
  end

  # Two very different failures: a server without the IP database (every visitor fails), or one
  # visitor whose address has no location. A private address usually means a proxy in front of Rails is
  # not passing the visitor's address on.
  def unplaceable
    reason =
      if !GeoLookup.instance.available?
        "no_geo_db"
      elsif private_address?(request.remote_ip)
        "private_address"
      else
        "unplaceable"
      end
    Rails.logger.warn("hunt refused: #{reason}") unless reason == "unplaceable"
    render json: { error: reason }, status: :unprocessable_content
  end

  def private_address?(ip)
    addr = IPAddr.new(ip)
    addr.private? || addr.loopback? || addr.link_local?
  rescue IPAddr::InvalidAddressError
    false
  end

  def distance_m(a, b)
    rad = Math::PI / 180
    dlat = (b["lat"] - a["lat"]) * rad
    dlon = (b["lon"] - a["lon"]) * rad
    h = Math.sin(dlat / 2)**2 + Math.cos(a["lat"] * rad) * Math.cos(b["lat"] * rad) * Math.sin(dlon / 2)**2
    2 * 6_371_008.8 * Math.asin(Math.sqrt(h))
  end

  def hunt_gone
    session.delete(:hunt)
    render json: { error: "gone" }, status: :gone
  end

  def solver_busy
    render json: { error: "busy" }, status: :service_unavailable
  end

  def solver_down(error)
    Rails.logger.warn("solver: #{error.message}")
    render json: { error: "solver unavailable" }, status: :bad_gateway
  end
end

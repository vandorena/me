require "net/http"

# Client for the Python solver sidecar (sidecar/app/server.py). One keep-alive connection per thread;
# no retries, because every call is repeated by the browser's next 5 s chunk anyway.
class SolverClient
  class Error < StandardError; end
  class Gone < Error; end   # the sidecar no longer knows this hunt (expired or restarted)
  class Busy < Error; end   # too many hunts at once

  class << self
    attr_writer :instance # tests swap in a fake

    def instance
      @instance ||= new
    end
  end

  def initialize(url: ENV.fetch("SOLVER_URL", "http://localhost:8000"), key: ENV["SOLVER_KEY"], timeout: 4.0)
    @uri = URI(url)
    @key = key
    @timeout = timeout
  end

  def start(lat:, lon:, radius_km:, cap_s:, country_code:)
    post_json("/sessions", lat:, lon:, radius_km:, cap_s:, country_code:).fetch("token")
  end

  # pcm: little-endian int16 at 8 kHz, mono. Returns the solver's latest snapshot.
  def chunk(token, pcm, t0_utc_ms)
    request = Net::HTTP::Post.new("/sessions/#{escape(token)}/chunk?t0_utc_ms=#{t0_utc_ms.to_f}")
    request["Content-Type"] = "application/octet-stream"
    request.body = pcm
    call(request)
  end

  def status(token)
    call(Net::HTTP::Get.new("/sessions/#{escape(token)}"))
  end

  def result(token)
    call(Net::HTTP::Get.new("/sessions/#{escape(token)}/result"))
  end

  def stop(token)
    call(Net::HTTP::Delete.new("/sessions/#{escape(token)}"))
  rescue Error
    nil
  end

  def test_flight(token, kind = "prop")
    call(Net::HTTP::Post.new("/sessions/#{escape(token)}/test_flight?kind=#{escape(kind)}"))
  end

  private

  def post_json(path, body)
    request = Net::HTTP::Post.new(path)
    request["Content-Type"] = "application/json"
    request.body = JSON.generate(body)
    call(request)
  end

  def call(request)
    request["X-Solver-Key"] = @key if @key.present?
    response = connection.request(request)
    case response
    when Net::HTTPSuccess then JSON.parse(response.body)
    when Net::HTTPNotFound then raise Gone, "hunt not found"
    when Net::HTTPServiceUnavailable then raise Busy, "solver busy"
    else raise Error, "HTTP #{response.code}: #{response.body.to_s[0, 200]}"
    end
  rescue Timeout::Error, SocketError, SystemCallError, EOFError, IOError, JSON::ParserError => e
    reset
    raise Error, "#{e.class}: #{e.message}"
  end

  def escape(token)
    ERB::Util.url_encode(token)
  end

  def connection
    Thread.current[:solver_http] ||= Net::HTTP.new(@uri.host, @uri.port).tap do |http|
      http.use_ssl = @uri.scheme == "https"
      http.open_timeout = @timeout / 2
      http.read_timeout = @timeout
      http.write_timeout = @timeout
      http.keep_alive_timeout = 20
      http.max_retries = 0
      http.start
    end
  end

  def reset
    Thread.current[:solver_http]&.finish
  rescue IOError
    nil
  ensure
    Thread.current[:solver_http] = nil
  end
end

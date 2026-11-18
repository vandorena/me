require "test_helper"

class HuntFlowTest < ActionDispatch::IntegrationTest
  setup do
    @solver = FakeSolver.new
    SolverClient.instance = @solver
    @dev_prior = Rails.configuration.x.dev_prior
    Rails.configuration.x.dev_prior = [ 41.8268, -71.4025, 30.0 ]
  end

  teardown do
    SolverClient.instance = nil
    Rails.configuration.x.dev_prior = @dev_prior
  end

  test "the page loads" do
    get root_path

    assert_response :success
    assert_select "[data-controller=hunt]"
    assert_select "input[type=range][data-hunt-target=cap]"
    assert_select "footer a[href=\"https://db-ip.com\"]", "IP Geolocation by DB-IP"
  end

  test "starting a hunt hands the solver the IP prior and a clamped cap" do
    post hunt_path, params: { cap_s: 99_999 }, as: :json

    assert_response :success
    body = response.parsed_body
    assert_equal 900, body["cap_s"]
    assert_equal 30.0, body["prior"]["radius_km"]
    assert body["server_time_ms"].positive?
    assert_equal [ :start, { lat: 41.8268, lon: -71.4025, radius_km: 30.0, cap_s: 900, country_code: nil } ],
                 @solver.calls.first
  end

  test "an address we cannot place cannot be hunted" do
    Rails.configuration.x.dev_prior = nil
    post hunt_path, params: { cap_s: 180 }, as: :json

    assert_response :unprocessable_content
    assert_empty @solver.calls
  end

  test "a refusal says whether the server lacks its database or sees a private address" do
    Rails.configuration.x.dev_prior = nil

    GeoLookup.instance = GeoLookup.new(city_db: "/missing")
    post hunt_path, params: { cap_s: 180 }, as: :json
    assert_equal "no_geo_db", response.parsed_body["error"]

    GeoLookup.instance = Struct.new(:available?) { def call(_ip) = nil }.new(true)
    post hunt_path, params: { cap_s: 180 }, as: :json
    assert_equal "private_address", response.parsed_body["error"]
  ensure
    GeoLookup.instance = nil
  end

  test "chunks are relayed to the solver with their timestamp" do
    post hunt_path, params: { cap_s: 180 }, as: :json
    pcm = "\x00\x01".b * 40_000
    post chunk_hunt_path(t0_utc_ms: 1_794_000_000_000), params: pcm,
         headers: { "Content-Type" => "application/octet-stream" }

    assert_response :success
    assert_equal "listening", response.parsed_body["status"]
    assert_equal [ :chunk, "tok-1", 80_000, 1_794_000_000_000.0 ], @solver.calls.last
  end

  test "bad chunks are refused before they reach the solver" do
    post hunt_path, params: { cap_s: 180 }, as: :json
    post chunk_hunt_path(t0_utc_ms: 1), params: "\x00".b, headers: { "Content-Type" => "application/octet-stream" }
    assert_response :unprocessable_content
    post chunk_hunt_path, params: "\x00\x00".b, headers: { "Content-Type" => "application/octet-stream" }
    assert_response :unprocessable_content
    assert_equal [ :start ], @solver.calls.map(&:first)
  end

  test "a chunk may hold up to 10 s of 16 kHz audio" do
    post hunt_path, params: { cap_s: 180 }, as: :json
    headers = { "Content-Type" => "application/octet-stream" }

    post chunk_hunt_path(t0_utc_ms: 1), params: ("\x00" * 2 * 16_000 * 10).b, headers: headers
    assert_response :success
    post chunk_hunt_path(t0_utc_ms: 1), params: ("\x00" * (2 * 16_000 * 10 + 2)).b, headers: headers
    assert_response :unprocessable_content
  end

  test "a chunk without a hunt is gone" do
    post chunk_hunt_path(t0_utc_ms: 1), params: "\x00\x00".b, headers: { "Content-Type" => "application/octet-stream" }

    assert_response :gone
  end

  test "the pin stores two distances and nothing that locates anyone" do
    post hunt_path, params: { cap_s: 300 }, as: :json
    pin = { lat: 41.8310, lon: -71.4010 }

    assert_difference "Result.count", 1 do
      post truth_hunt_path, params: pin, as: :json
    end

    assert_response :success
    result = Result.last
    assert_equal "doppler", result.tier
    assert_in_delta 134, result.error_m, 6         # guess -> pin: 111 m north, 75 m west
    assert_in_delta 480, result.ip_error_m, 6      # IP centre -> pin: 467 m north, 125 m east
    assert_equal 0, result.error_m % 10
    assert_equal [ 182, 300, 2, 60 ], [ result.listened_s, result.cap_s, result.passes_detected, result.mains_hz ]
    assert response.parsed_body["audio_won"]

    stored = Result.columns.map(&:name)
    assert_empty stored.grep(/lat|lon|ip_address|token|audio/)
    assert_equal [ :stop, "tok-1" ], @solver.calls.last

    post truth_hunt_path, params: pin, as: :json
    assert_response :gone
  end

  test "without a Doppler guess our error is the IP's error" do
    @solver.run = @solver.run.merge("tier" => "ip", "guess" => nil)
    post hunt_path, params: { cap_s: 300 }, as: :json
    post truth_hunt_path, params: { lat: 41.9, lon: -71.4 }, as: :json

    result = Result.last
    assert_equal result.ip_error_m, result.error_m
    assert_not response.parsed_body["audio_won"]
  end

  test "a pin off the planet is refused" do
    post hunt_path, params: { cap_s: 300 }, as: :json
    assert_no_difference "Result.count" do
      post truth_hunt_path, params: { lat: 123, lon: 0 }, as: :json
    end
    assert_response :unprocessable_content
  end

  test "skipping the pin stops the solver and saves nothing" do
    post hunt_path, params: { cap_s: 300 }, as: :json
    assert_no_difference "Result.count" do
      delete hunt_path
    end
    assert_response :no_content
    assert_equal [ :stop, "tok-1" ], @solver.calls.last
  end

  test "solver trouble becomes a clear status" do
    @solver.fail_with = SolverClient::Busy
    post hunt_path, params: { cap_s: 300 }, as: :json
    assert_response :service_unavailable

    @solver.fail_with = SolverClient::Error
    post hunt_path, params: { cap_s: 300 }, as: :json
    assert_response :bad_gateway
  end

  test "test flights are available outside production" do
    post hunt_path, params: { cap_s: 300 }, as: :json
    post test_flight_hunt_path

    assert_response :success
    assert_equal [ :test_flight, "tok-1" ], @solver.calls.last
  end
end

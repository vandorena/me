ENV["RAILS_ENV"] ||= "test"
require_relative "../config/environment"
require "rails/test_help"

module ActiveSupport
  class TestCase
    # Run tests in parallel with specified workers
    parallelize(workers: :number_of_processors)
  end
end

# Stands in for the Python sidecar. Records what it was sent; answers like the real one.
class FakeSolver
  attr_reader :calls
  attr_accessor :run, :snapshot, :fail_with

  def initialize
    @calls = []
    @snapshot = { "tier" => "ip", "status" => "listening", "listened_s" => 5.0, "done" => false }
    @run = {
      "tier" => "doppler", "guess" => { "lat" => 41.8300, "lon" => -71.4000 },
      "prior" => { "lat" => 41.8268, "lon" => -71.4025 }, "confidence" => 0.8, "ambiguous" => false,
      "listened_s" => 182.4, "cap_s" => 300, "aircraft_scored" => 4, "passes_detected" => 2, "mains_hz" => 60
    }
  end

  def start(**args)
    record(:start, args)
    "tok-#{calls.size}"
  end

  def chunk(token, pcm, t0) = record(:chunk, token, pcm.bytesize, t0) && snapshot
  def status(token) = record(:status, token) && snapshot
  def result(token) = record(:result, token) && run
  def stop(token) = record(:stop, token) && { "ok" => true }
  def test_flight(token, kind) = record(:test_flight, token, kind) && { "pcm16" => "", "start_utc_ms" => 0 }

  private

  def record(*call)
    raise fail_with if fail_with
    calls << call
    true
  end
end

// Dynamic forcings of the autoregressive rollout: the values anemoi-inference's ComputedForcings
// writes into the newest input slot after each step (docs/forcings.md). Double precision,
// same operation order as earthkit-data's `forcings` source, cast to float at the end.
// Standard library only; dates are UTC and no time zone / locale is consulted.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace varda {

// A UTC calendar date-time (proleptic Gregorian), second resolution.
struct DateTime {
    int year = 1970, month = 1, day = 1, hour = 0, minute = 0, second = 0;
};

// "YYYY-MM-DDTHH:MM:SS" (also "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DDTHH:MM", optional trailing 'Z').
// Throws std::runtime_error on anything else or on an invalid calendar date.
DateTime parse_iso8601_utc(const std::string& s);
std::string format_iso8601_utc(const DateTime& d);  // "YYYY-MM-DDTHH:MM:SS"

DateTime add_hours(const DateTime& d, int64_t hours);  // negative allowed; leap years respected
int64_t to_unix_seconds(const DateTime& d);
DateTime from_unix_seconds(int64_t t);

// Names accepted by compute_forcings. The five dynamic forcings of the Varda family are first.
//   cos_julian_day sin_julian_day cos_local_time sin_local_time insolation
// plus the time-independent cos_latitude sin_latitude cos_longitude sin_longitude
// (anemoi computes these through the same code path; the model reads them from the node attributes).
bool is_forcing_name(const std::string& name);

// out[g * names.size() + k] = float(forcing names[k] at grid point g, valid time `valid`).
// `lat`, `lon`: F64 [G], degrees, as stored in the converted file (`grid.latitudes` / `grid.longitudes`).
// Throws std::invalid_argument for an unknown name. `out` has room for G * names.size() floats.
// Threaded with OpenMP; the result does not depend on the thread count.
void compute_forcings(const DateTime& valid, const std::vector<std::string>& names, const double* lat,
                      const double* lon, size_t G, float* out);

// Same, from an ISO-8601 string.
void compute_forcings(const std::string& valid_iso, const std::vector<std::string>& names, const double* lat,
                      const double* lon, size_t G, float* out);

}  // namespace varda

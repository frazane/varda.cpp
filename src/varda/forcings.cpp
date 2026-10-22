#include "varda/forcings.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace varda {

namespace {

// Howard Hinnant's civil-date algorithms (public domain), proleptic Gregorian.
int64_t days_from_civil(int64_t y, int m, int d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const int64_t yoe = y - era * 400;
    const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

void civil_from_days(int64_t z, int& y, int& m, int& d) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t yy = yoe + era * 400;
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const int64_t mp = (5 * doy + 2) / 153;
    d = int(doy - (153 * mp + 2) / 5 + 1);
    m = int(mp < 10 ? mp + 3 : mp - 9);
    y = int(yy + (m <= 2));
}

bool is_leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

int days_in_month(int y, int m) {
    static const int dm[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return m == 2 && is_leap(y) ? 29 : dm[m - 1];
}

// Python: numpy.deg2rad(x) == x * (pi / 180) (checked bitwise against numpy 2.3.5).
constexpr double kPi = 3.14159265358979323846;
inline double deg2rad(double x) { return x * (kPi / 180.0); }

// Python: x % m for doubles (numpy npy_divmod, m > 0): fmod, then shift a negative remainder.
inline double py_mod(double a, double m) {
    double r = std::fmod(a, m);
    if (r != 0.0) {
        if (r < 0.0) r += m;
    } else {
        r = std::copysign(0.0, m);
    }
    return r;
}

enum class Kind { CosJulian, SinJulian, CosLocal, SinLocal, Insolation, CosLat, SinLat, CosLon, SinLon };

bool parse_kind(const std::string& n, Kind& k) {
    static const struct {
        const char* name;
        Kind kind;
    } table[] = {{"cos_julian_day", Kind::CosJulian}, {"sin_julian_day", Kind::SinJulian},
                 {"cos_local_time", Kind::CosLocal},  {"sin_local_time", Kind::SinLocal},
                 {"insolation", Kind::Insolation},    {"cos_latitude", Kind::CosLat},
                 {"sin_latitude", Kind::SinLat},      {"cos_longitude", Kind::CosLon},
                 {"sin_longitude", Kind::SinLon}};
    for (auto& e : table)
        if (n == e.name) {
            k = e.kind;
            return true;
        }
    return false;
}

int to_int(const std::string& s, size_t pos, size_t len, const std::string& whole) {
    if (pos + len > s.size()) throw std::runtime_error("bad ISO-8601 date: " + whole);
    int v = 0;
    for (size_t i = pos; i < pos + len; ++i) {
        if (s[i] < '0' || s[i] > '9') throw std::runtime_error("bad ISO-8601 date: " + whole);
        v = v * 10 + (s[i] - '0');
    }
    return v;
}

}  // namespace

int64_t to_unix_seconds(const DateTime& d) {
    return days_from_civil(d.year, d.month, d.day) * 86400 + d.hour * 3600 + d.minute * 60 + d.second;
}

DateTime from_unix_seconds(int64_t t) {
    int64_t days = t / 86400, rem = t % 86400;
    if (rem < 0) rem += 86400, --days;
    DateTime d;
    civil_from_days(days, d.year, d.month, d.day);
    d.hour = int(rem / 3600);
    d.minute = int(rem % 3600 / 60);
    d.second = int(rem % 60);
    return d;
}

DateTime add_hours(const DateTime& d, int64_t hours) { return from_unix_seconds(to_unix_seconds(d) + hours * 3600); }

DateTime parse_iso8601_utc(const std::string& s) {
    std::string t = s;
    if (!t.empty() && t.back() == 'Z') t.pop_back();
    const std::string want = "bad ISO-8601 date (want YYYY-MM-DDTHH:MM:SS): " + s;
    if (t.size() != 19 && t.size() != 16) throw std::runtime_error(want);
    if (t[4] != '-' || t[7] != '-' || (t[10] != 'T' && t[10] != ' ') || t[13] != ':' ||
        (t.size() == 19 && t[16] != ':'))
        throw std::runtime_error(want);
    DateTime d;
    d.year = to_int(t, 0, 4, s);
    d.month = to_int(t, 5, 2, s);
    d.day = to_int(t, 8, 2, s);
    d.hour = to_int(t, 11, 2, s);
    d.minute = to_int(t, 14, 2, s);
    d.second = t.size() == 19 ? to_int(t, 17, 2, s) : 0;
    if (d.month < 1 || d.month > 12 || d.day < 1 || d.day > days_in_month(d.year, d.month) || d.hour > 23 ||
        d.minute > 59 || d.second > 59)
        throw std::runtime_error("invalid calendar date: " + s);
    return d;
}

std::string format_iso8601_utc(const DateTime& d) {
    char b[48];
    std::snprintf(b, sizeof b, "%04d-%02d-%02dT%02d:%02d:%02d", d.year, d.month, d.day, d.hour, d.minute, d.second);
    return b;
}

bool is_forcing_name(const std::string& name) {
    Kind k;
    return parse_kind(name, k);
}

void compute_forcings(const DateTime& date, const std::vector<std::string>& names, const double* lat,
                      const double* lon, size_t G, float* out) {
    const size_t nf = names.size();
    std::vector<Kind> kinds(nf);
    for (size_t k = 0; k < nf; ++k)
        if (!parse_kind(names[k], kinds[k])) throw std::invalid_argument("unknown forcing name: " + names[k]);

    // ForcingMaker.julian_day: (date - Jan 1 of its year) -> delta.days + delta.seconds / 86400.0
    const int64_t secs_of_day = date.hour * 3600 + date.minute * 60 + date.second;
    const int64_t day0 = days_from_civil(date.year, date.month, date.day) - days_from_civil(date.year, 1, 1);
    const double julian = double(day0) + double(secs_of_day) / 86400.0;
    // radians = julian_day / 365.25 * np.pi * 2   (left to right)
    const double jangle = julian / 365.25 * kPi * 2;
    const float cos_jd = float(std::cos(jangle)), sin_jd = float(std::sin(jangle));

    // ForcingMaker.local_time: hours_since_midnight = (delta.days + delta.seconds / 86400.0) * 24, delta.days == 0
    const double hours_since_midnight = (double(0) + double(secs_of_day) / 86400.0) * 24;

    // earthkit.data.utils.meteo.solar_declination_angle / cos_solar_zenith_angle
    double dec_sin = 0, dec_cos = 0, time_correction = 0;
    for (Kind k : kinds)
        if (k == Kind::Insolation) {
            const double a = jangle;  // identical expression: julian_day / DAYS_PER_YEAR * np.pi * 2
            const double declination = 0.396372 - 22.91327 * std::cos(a) + 4.025430 * std::sin(a) -
                                       0.387205 * std::cos(2 * a) + 0.051967 * std::sin(2 * a) -
                                       0.154527 * std::cos(3 * a) + 0.084798 * std::sin(3 * a);
            time_correction = 0.004297 + 0.107029 * std::cos(a) - 1.837877 * std::sin(a) -
                              0.837378 * std::cos(2 * a) - 2.340475 * std::sin(2 * a);
            const double dec = deg2rad(declination);
            dec_sin = std::sin(dec);
            dec_cos = std::cos(dec);
            break;
        }
    const double hour_angle0 = double((date.hour - 12) * 15);

#pragma omp parallel for schedule(static)
    for (long long g = 0; g < (long long)G; ++g) {
        float* o = out + size_t(g) * nf;
        for (size_t k = 0; k < nf; ++k) {
            double v = 0;
            switch (kinds[k]) {
                case Kind::CosJulian: o[k] = cos_jd; continue;
                case Kind::SinJulian: o[k] = sin_jd; continue;
                case Kind::CosLocal:
                case Kind::SinLocal: {
                    // (lon / 360.0 * 24.0 + hours_since_midnight) % 24 ; radians = lt / 24 * np.pi * 2
                    const double lt = py_mod(lon[g] / 360.0 * 24.0 + hours_since_midnight, 24.0);
                    const double r = lt / 24 * kPi * 2;
                    v = kinds[k] == Kind::CosLocal ? std::cos(r) : std::sin(r);
                    break;
                }
                case Kind::Insolation: {
                    const double latr = deg2rad(lat[g]);
                    const double sindec_sinlat = dec_sin * std::sin(latr);
                    const double cosdec_coslat = dec_cos * std::cos(latr);
                    const double solar_angle = deg2rad(hour_angle0 + lon[g] + time_correction);
                    const double z = sindec_sinlat + cosdec_coslat * std::cos(solar_angle);
                    v = z < 0.0 ? 0.0 : z;  // np.clip(z, 0.0, None)
                    break;
                }
                case Kind::CosLat: v = std::cos(deg2rad(lat[g])); break;
                case Kind::SinLat: v = std::sin(deg2rad(lat[g])); break;
                case Kind::CosLon: v = std::cos(deg2rad(lon[g])); break;
                case Kind::SinLon: v = std::sin(deg2rad(lon[g])); break;
            }
            o[k] = float(v);
        }
    }
}

void compute_forcings(const std::string& valid_iso, const std::vector<std::string>& names, const double* lat,
                      const double* lon, size_t G, float* out) {
    compute_forcings(parse_iso8601_utc(valid_iso), names, lat, lon, G, out);
}

}  // namespace varda

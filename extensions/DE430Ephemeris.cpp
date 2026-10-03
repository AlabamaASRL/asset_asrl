#include "DE430Ephemeris.h"

#include "calceph.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ASSET {

namespace {
constexpr int earth_naif_id = 399;
constexpr int output_units = CALCEPH_UNIT_KM | CALCEPH_UNIT_SEC |
                             CALCEPH_USE_NAIFID;
}

DE430Ephemeris::DE430Ephemeris(const std::string& kernel_file)
    : kernel_file_(kernel_file) {
  if (kernel_file.empty())
    throw std::invalid_argument("DE430 kernel_file must not be empty");
  if (!std::ifstream(kernel_file, std::ios::binary))
    throw std::invalid_argument("Cannot open DE430 ephemeris kernel: " + kernel_file);
  ephemeris_ = calceph_open(kernel_file.c_str());
  if (!ephemeris_)
    throw std::invalid_argument("Cannot open DE430 ephemeris kernel: " + kernel_file);
  if (!calceph_prefetch(ephemeris_)) {
    calceph_close(ephemeris_);
    ephemeris_ = nullptr;
    throw std::runtime_error("CALCEPH could not prefetch DE430 kernel: " + kernel_file);
  }
  int continuous = 0;
  if (!calceph_gettimespan(ephemeris_, &first_jd_, &last_jd_, &continuous)) {
    calceph_close(ephemeris_);
    ephemeris_ = nullptr;
    throw std::runtime_error("CALCEPH could not read DE430 coverage: " + kernel_file);
  }
  thread_safe_ = calceph_isthreadsafe(ephemeris_) != 0;
}

DE430Ephemeris::~DE430Ephemeris() {
  if (ephemeris_) calceph_close(ephemeris_);
}

DE430Ephemeris::State DE430Ephemeris::earth_relative_state(
    double epoch_jd, double elapsed_seconds, int target_naif_id) const {
  if (!std::isfinite(epoch_jd) || !std::isfinite(elapsed_seconds))
    throw std::invalid_argument("DE430 epoch and elapsed seconds must be finite");
  const double jd0 = std::floor(epoch_jd);
  const double offset_days = (epoch_jd - jd0) + elapsed_seconds / 86400.0;
  const double julian_date = jd0 + offset_days;
  if (julian_date < first_jd_ || julian_date > last_jd_)
    throw std::out_of_range("DE430 kernel does not cover Julian date " +
                            std::to_string(julian_date));

  // Keep the small elapsed-time term out of the large Julian-date term. This
  // is the precision-preserving two-part interface CALCEPH provides.
  double state[9]{};
  auto compute = [&]() {
    return calceph_compute_order(ephemeris_, jd0, offset_days,
                                 target_naif_id, earth_naif_id,
                                 output_units, 2, state);
  };
  int success;
  if (thread_safe_) {
    success = compute();
  } else {
    std::lock_guard<std::mutex> lock(access_mutex_);
    success = compute();
  }
  if (!success)
    throw std::runtime_error("CALCEPH failed to evaluate target NAIF " +
                             std::to_string(target_naif_id) + " at Julian date " +
                             std::to_string(julian_date));

  State result;
  result.position_km = Eigen::Map<const Eigen::Vector3d>(state);
  result.velocity_km_s = Eigen::Map<const Eigen::Vector3d>(state + 3);
  result.acceleration_km_s2 = Eigen::Map<const Eigen::Vector3d>(state + 6);
  return result;
}

std::string DE430BodyName(const std::string& body) {
  std::string name = body;
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  if (name == "moon" || name == "sun" || name == "mars" ||
      name == "jupiter" || name == "saturn")
    return name;
  throw std::invalid_argument(
      "Unsupported DE430 body '" + body +
      "'; expected moon, sun, mars, jupiter, or saturn");
}

int DE430BodyNaifId(const std::string& body) {
  const std::string name = DE430BodyName(body);
  if (name == "moon") return 301;
  if (name == "sun") return 10;
  if (name == "mars") return 4;       // Mars system barycenter
  if (name == "jupiter") return 5;    // Jupiter system barycenter
  return 6;                            // Saturn system barycenter
}

}  // namespace ASSET

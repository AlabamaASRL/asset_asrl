#pragma once

#include <Eigen/Core>

#include <memory>
#include <mutex>
#include <string>

struct calcephbin;

namespace ASSET {

/** Shared, read-only access to a JPL DE430 SPK through CALCEPH.
 *
 * CALCEPH is prefetched when the file is opened.  Prefetched descriptors are
 * thread-safe in a thread-enabled CALCEPH build; a mutex remains available as
 * a conservative fallback for other builds.  Positions are returned relative
 * to the Earth (NAIF 399), in the ICRF axes and km/s units used by the released
 * Fortran propagator.
 */
class DE430Ephemeris {
 public:
  struct State {
    Eigen::Vector3d position_km;
    Eigen::Vector3d velocity_km_s;
    Eigen::Vector3d acceleration_km_s2;
  };

  explicit DE430Ephemeris(const std::string& kernel_file);
  ~DE430Ephemeris();

  DE430Ephemeris(const DE430Ephemeris&) = delete;
  DE430Ephemeris& operator=(const DE430Ephemeris&) = delete;

  /** Evaluate epoch_jd + elapsed_seconds/86400 as a two-part date. */
  State earth_relative_state(double epoch_jd, double elapsed_seconds,
                             int target_naif_id) const;

  const std::string& kernel_file() const { return kernel_file_; }
  double first_julian_date() const { return first_jd_; }
  double last_julian_date() const { return last_jd_; }
  bool is_thread_safe() const { return thread_safe_; }

 private:
  calcephbin* ephemeris_ = nullptr;
  std::string kernel_file_;
  double first_jd_ = 0.0;
  double last_jd_ = 0.0;
  bool thread_safe_ = false;
  mutable std::mutex access_mutex_;
};

/** Map the public body names to the NAIF IDs used by the DE430 SPK. */
int DE430BodyNaifId(const std::string& body);

/** Canonical lower-case public body name, or throws for an unsupported body. */
std::string DE430BodyName(const std::string& body);

}  // namespace ASSET

#include "ASSET_Extensions.h"
#include "DE430Ephemeris.h"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace ASSET {

/** Earth-relative DE430 body position as a differentiable function of time.
 *
 * Input is elapsed seconds from epoch_jd and output is ICRF position in km.
 * CALCEPH supplies position, velocity, and acceleration.  Those values form a
 * local second-order Taylor jet in the ASSET scalar, giving the exact DE430
 * value, first derivative, and second derivative at every evaluation point.
 * The Julian date is intentionally passed directly as TDB, matching the
 * released Fortran propagator's convention.
 */
struct DE430Position
    : VectorFunction<DE430Position, 1, 3, AutodiffFwd, AutodiffFwd> {
  using Base = VectorFunction<DE430Position, 1, 3, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;

  double epoch_jd;
  std::string kernel_file;
  std::string body;
  bool apparent;
  int target_naif_id;
  std::shared_ptr<const DE430Ephemeris> ephemeris;

  DE430Position(double epoch, const std::string& path, const std::string& body_name,
                bool apparent_position = false)
      : epoch_jd(epoch), kernel_file(path), body(DE430BodyName(body_name)),
        apparent(apparent_position), target_naif_id(DE430BodyNaifId(body_name)),
        ephemeris(std::make_shared<DE430Ephemeris>(path)) {
    if (!std::isfinite(epoch))
      throw std::invalid_argument("DE430Position epoch_jd must be finite");
    if (apparent && body != "sun")
      throw std::invalid_argument("DE430Position apparent mode is supported only for the Sun");
  }

  template<class Scalar>
  Eigen::Matrix<Scalar, 3, 1> make_jet(
      const DE430Ephemeris::State& state, const Scalar& delta_seconds,
      double first_scale = 1.0, double second_scale = 1.0,
      const Eigen::Vector3d& second_extra = Eigen::Vector3d::Zero()) const {
    Eigen::Matrix<Scalar, 3, 1> result;
    for (int i = 0; i < 3; ++i)
      result[i] = Scalar(state.position_km[i]) +
                  Scalar(state.velocity_km_s[i] * first_scale) * delta_seconds +
                  Scalar(0.5 * (state.acceleration_km_s2[i] * second_scale +
                                second_extra[i])) * delta_seconds * delta_seconds;
    return result;
  }

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> input,
                           ConstVectorBaseRef<OutType> output_) const {
    using Scalar = typename InType::Scalar;
    auto& output = output_.const_cast_derived();
    const double elapsed = static_cast<double>(autodiff::val(input[0]));
    if (!std::isfinite(elapsed))
      throw std::invalid_argument("DE430Position elapsed seconds must be finite");
    const Scalar delta = input[0] - Scalar(elapsed);

    if (!apparent) {
      const auto state = ephemeris->earth_relative_state(
          epoch_jd, elapsed, target_naif_id);
      const auto position = make_jet(state, delta);
      for (int i = 0; i < 3; ++i) output[i] = position[i];
      return;
    }

    // Match the reference's one-step light-time Sun: evaluate the geometric
    // Sun at t, then evaluate it once more at t-|R(t)|/c.
    constexpr double light_speed_km_s = 299792.458;
    const auto geometric = ephemeris->earth_relative_state(
        epoch_jd, elapsed, target_naif_id);
    const double range = geometric.position_km.norm();
    const double radial_rate = geometric.position_km.dot(geometric.velocity_km_s);
    const double retarded_seconds = elapsed - range / light_speed_km_s;
    const auto retarded = ephemeris->earth_relative_state(
        epoch_jd, retarded_seconds, target_naif_id);

    const double retarded_rate = 1.0 - radial_rate / (range * light_speed_km_s);
    const double range_second =
        (geometric.velocity_km_s.squaredNorm() +
         geometric.position_km.dot(geometric.acceleration_km_s2)) / range -
        radial_rate * radial_rate / (range * range * range);
    const double retarded_second = -range_second / light_speed_km_s;
    const auto position = make_jet(retarded, delta, retarded_rate,
                                   retarded_rate * retarded_rate,
                                   retarded.velocity_km_s * retarded_second);
    for (int i = 0; i < 3; ++i) output[i] = position[i];
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<DE430Position>(m, name,
        "Earth-relative DE430 ICRF body position [km] from elapsed TDB seconds.");
    obj.def(py::init<double, const std::string&, const std::string&, bool>(),
            py::arg("epoch_jd"), py::arg("kernel_file"), py::arg("body"),
            py::arg("apparent") = false);
    obj.def_readonly("epoch_jd", &DE430Position::epoch_jd);
    obj.def_readonly("kernel_file", &DE430Position::kernel_file);
    obj.def_readonly("body", &DE430Position::body);
    obj.def_readonly("apparent", &DE430Position::apparent);
    obj.def_property_readonly("first_julian_date", [](const DE430Position& f) {
      return f.ephemeris->first_julian_date();
    });
    obj.def_property_readonly("last_julian_date", [](const DE430Position& f) {
      return f.ephemeris->last_julian_date();
    });
    Base::DenseBaseBuild(obj);
    obj.def("__call__", [](const DE430Position& f,
                            const GenericFunction<-1, -1>& input) {
      return GenericFunction<-1, -1>(f.eval(input));
    }, py::arg("elapsed_seconds"));
  }
};

void BuildDE430Position(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<DE430Position>(m, "DE430Position");
}

}  // namespace ASSET

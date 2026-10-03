#include "ASSET_Extensions.h"
#include "DE430Ephemeris.h"

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ASSET {

/** DE430 point-mass third-body acceleration used by the Fortran propagator.
 *
 * Input is [r_icrf_km(3), elapsed_seconds] and output is km/s^2.  The
 * differential form removes the acceleration of the geocenter:
 *   mu * ((r_body-r_sc)/|r_body-r_sc|^3 - r_body/|r_body|^3).
 * moon_sun evaluates the Moon and Sun; all additionally evaluates the Mars,
 * Jupiter, and Saturn system barycenters.  The gravitational parameters are
 * reproduced from constants.f90 without decimal rounding.
 */
struct ThirdBodyGravity
    : VectorFunction<ThirdBodyGravity, 4, 3, AutodiffFwd, AutodiffFwd> {
  using Base = VectorFunction<ThirdBodyGravity, 4, 3, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;

  struct Body { int naif_id; double mu_km3_s2; };

  double epoch_jd;
  std::string kernel_file;
  std::string mode;
  std::shared_ptr<const DE430Ephemeris> ephemeris;
  std::vector<Body> bodies;

  static double au_day_to_km3_s2(double gm_au3_day2) {
    constexpr double astronomical_unit_km = 149597870.7;
    constexpr double day_seconds = 86400.0;
    return gm_au3_day2 * astronomical_unit_km * astronomical_unit_km *
           astronomical_unit_km / (day_seconds * day_seconds);
  }

  ThirdBodyGravity(double epoch, const std::string& path,
                   const std::string& selected_mode = "moon_sun")
      : epoch_jd(epoch), kernel_file(path), mode(selected_mode),
        ephemeris(std::make_shared<DE430Ephemeris>(path)) {
    if (!std::isfinite(epoch))
      throw std::invalid_argument("ThirdBodyGravity epoch_jd must be finite");
    if (mode != "moon_sun" && mode != "all")
      throw std::invalid_argument("ThirdBodyGravity mode must be 'moon_sun' or 'all'");

    bodies.push_back({301, au_day_to_km3_s2(
        0.8997011390199871e-9 - 0.8887692445125634e-9)});
    bodies.push_back({10, au_day_to_km3_s2(0.2959122082855911e-3)});
    if (mode == "all") {
      bodies.push_back({4, au_day_to_km3_s2(0.954954869555077e-10)});
      bodies.push_back({5, au_day_to_km3_s2(0.282534584083387e-6)});
      bodies.push_back({6, au_day_to_km3_s2(0.845970607324503e-7)});
    }
  }

  template<class Scalar>
  static Eigen::Matrix<Scalar, 3, 1> body_jet(
      const DE430Ephemeris::State& state, const Scalar& delta_seconds) {
    Eigen::Matrix<Scalar, 3, 1> result;
    for (int i = 0; i < 3; ++i)
      result[i] = Scalar(state.position_km[i]) +
                  Scalar(state.velocity_km_s[i]) * delta_seconds +
                  Scalar(0.5 * state.acceleration_km_s2[i]) *
                      delta_seconds * delta_seconds;
    return result;
  }

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> input,
                           ConstVectorBaseRef<OutType> output_) const {
    using Scalar = typename InType::Scalar;
    auto& output = output_.const_cast_derived();
    const double elapsed = static_cast<double>(autodiff::val(input[3]));
    if (!std::isfinite(elapsed))
      throw std::invalid_argument("ThirdBodyGravity elapsed seconds must be finite");
    const Scalar delta = input[3] - Scalar(elapsed);
    const Eigen::Matrix<Scalar, 3, 1> spacecraft = input.template head<3>();
    output.setZero();

    for (const Body& body : bodies) {
      const auto state = ephemeris->earth_relative_state(
          epoch_jd, elapsed, body.naif_id);
      const Eigen::Matrix<Scalar, 3, 1> body_position = body_jet(state, delta);
      const Eigen::Matrix<Scalar, 3, 1> relative = body_position - spacecraft;
      const Scalar relative_range = relative.norm();
      const Scalar body_range = body_position.norm();
      output += Scalar(body.mu_km3_s2) *
          (relative / (relative_range * relative_range * relative_range) -
           body_position / (body_range * body_range * body_range));
    }
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<ThirdBodyGravity>(m, name,
        "DE430 differential third-body acceleration [km/s^2] for [r_icrf_km,t_s].");
    obj.def(py::init<double, const std::string&, const std::string&>(),
            py::arg("epoch_jd"), py::arg("kernel_file"),
            py::arg("mode") = "moon_sun");
    obj.def_readonly("epoch_jd", &ThirdBodyGravity::epoch_jd);
    obj.def_readonly("kernel_file", &ThirdBodyGravity::kernel_file);
    obj.def_readonly("mode", &ThirdBodyGravity::mode);
    Base::DenseBaseBuild(obj);
    obj.def("__call__", [](const ThirdBodyGravity& f,
                            const GenericFunction<-1, -1>& input) {
      return GenericFunction<-1, -1>(f.eval(input));
    }, py::arg("input"));
  }
};

void BuildThirdBodyGravity(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<ThirdBodyGravity>(m, "ThirdBodyGravity");
}

}  // namespace ASSET

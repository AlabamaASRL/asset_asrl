#include "ASSET_Extensions.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ASSET {

/** Cannonball solar-radiation pressure with optional Earth shadow.
 *
 * Input layout is [r_spacecraft_icrf_km(3), mass_kg,
 * r_sun_geocentric_icrf_km(3)]. Output is inertial acceleration in km/s^2.
 * The force direction and inverse-square pressure reproduce the released
 * Fortran reference.  shadow_mode may be "none", "cylindrical", or
 * "smooth_cylindrical".  The exact cylindrical mode is discontinuous at the
 * terminator and cylinder wall.  The smooth mode replaces both switches with
 * logistic transitions of the configured length scale and is intended for
 * phase transcription and derivative-based optimization.
 *
 * This file is original ASSET adapter code.  It does not copy a third-party
 * SRP implementation.
 */
struct CannonballSRP
    : VectorFunction<CannonballSRP, 7, 3, AutodiffFwd, AutodiffFwd> {
  using Base = VectorFunction<CannonballSRP, 7, 3, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;

  double area_m2;
  double reflectivity_coefficient;
  double earth_radius_km;
  double solar_pressure_n_m2;
  double astronomical_unit_m;
  std::string shadow_mode;
  double transition_km;

  CannonballSRP(double area, double cr, const std::string& mode,
                double transition, double earth_radius,
                double solar_pressure, double astronomical_unit)
      : area_m2(area), reflectivity_coefficient(cr),
        earth_radius_km(earth_radius), solar_pressure_n_m2(solar_pressure),
        astronomical_unit_m(astronomical_unit), shadow_mode(mode),
        transition_km(transition) {
    if (!std::isfinite(area) || area < 0.0)
      throw std::invalid_argument("CannonballSRP area_m2 must be finite and nonnegative");
    if (!std::isfinite(cr) || cr < 0.0)
      throw std::invalid_argument("CannonballSRP reflectivity coefficient must be finite and nonnegative");
    if (mode != "none" && mode != "cylindrical" && mode != "smooth_cylindrical")
      throw std::invalid_argument("CannonballSRP shadow_mode must be none, cylindrical, or smooth_cylindrical");
    if (!std::isfinite(transition) || transition <= 0.0)
      throw std::invalid_argument("CannonballSRP transition_km must be finite and positive");
    if (!std::isfinite(earth_radius) || earth_radius <= 0.0 ||
        !std::isfinite(solar_pressure) || solar_pressure <= 0.0 ||
        !std::isfinite(astronomical_unit) || astronomical_unit <= 0.0)
      throw std::invalid_argument("CannonballSRP physical constants must be finite and positive");
  }

  template<class Scalar>
  Scalar illumination(const Eigen::Matrix<Scalar,3,1>& spacecraft,
                      const Eigen::Matrix<Scalar,3,1>& sun) const {
    if (shadow_mode == "none") return Scalar(1.0);
    const Scalar sun_range = sun.norm();
    const auto sun_hat = sun / sun_range;
    const Scalar axial = spacecraft.dot(sun_hat);
    const auto transverse = spacecraft - axial * sun_hat;
    const Scalar radial = transverse.norm();
    if (shadow_mode == "cylindrical")
      return (axial < Scalar(0.0) && radial < Scalar(earth_radius_km))
                 ? Scalar(0.0) : Scalar(1.0);

    // darkness approaches one only when the point is both behind Earth and
    // inside the cylindrical radius.  Each logistic is evaluated in this
    // overflow-resistant form because orbit optimizers may test distant nodes.
    const auto logistic = [](const Scalar& value) -> Scalar {
      Scalar result;
      if (value >= Scalar(0.0)) {
        const Scalar e = exp(-value);
        result = Scalar(1.0) / (Scalar(1.0) + e);
      } else {
        const Scalar e = exp(value);
        result = e / (Scalar(1.0) + e);
      }
      return result;
    };
    const Scalar behind = logistic(-axial / Scalar(transition_km));
    const Scalar inside = logistic((Scalar(earth_radius_km) - radial) /
                                   Scalar(transition_km));
    return Scalar(1.0) - behind * inside;
  }

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> input,
                           ConstVectorBaseRef<OutType> output_) const {
    using Scalar = typename InType::Scalar;
    auto& output = output_.const_cast_derived();
    const Vector3<Scalar> spacecraft = input.template head<3>();
    const Scalar mass = input[3];
    const Vector3<Scalar> sun = input.template segment<3>(4);
    if (mass <= Scalar(0.0))
      throw std::invalid_argument("CannonballSRP mass must be positive");
    const auto spacecraft_to_sun = sun - spacecraft;
    const Scalar distance_km = spacecraft_to_sun.norm();
    if (distance_km <= Scalar(0.0) || sun.norm() <= Scalar(0.0))
      throw std::invalid_argument("CannonballSRP Sun range must be positive");
    const auto toward_sun = spacecraft_to_sun / distance_km;
    const Scalar distance_m = distance_km * Scalar(1000.0);
    const Scalar distance_ratio = Scalar(astronomical_unit_m) / distance_m;
    const Scalar pressure = Scalar(solar_pressure_n_m2) *
                            distance_ratio * distance_ratio;
    output = illumination(spacecraft, sun) * pressure *
        Scalar(reflectivity_coefficient * area_m2 / 1000.0) / mass *
        (-toward_sun);
  }

  double illumination_value(const Eigen::Matrix<double,7,1>& input) const {
    const Eigen::Vector3d spacecraft = input.head<3>();
    const Eigen::Vector3d sun = input.segment<3>(4);
    return illumination<double>(spacecraft, sun);
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<CannonballSRP>(m, name,
        "Cannonball SRP: [r_icrf_km,mass_kg,r_sun_icrf_km] -> km/s^2.");
    obj.def(py::init<double,double,const std::string&,double,double,double,double>(),
            py::arg("area_m2"), py::arg("reflectivity_coefficient"),
            py::arg("shadow_mode")="cylindrical",
            py::arg("transition_km")=20.0,
            py::arg("earth_radius_km")=6378.1363,
            py::arg("solar_pressure_n_m2")=4.56e-6,
            py::arg("astronomical_unit_m")=149597870700.0);
    obj.def_readonly("area_m2", &CannonballSRP::area_m2);
    obj.def_readonly("reflectivity_coefficient", &CannonballSRP::reflectivity_coefficient);
    obj.def_readonly("shadow_mode", &CannonballSRP::shadow_mode);
    obj.def_readonly("transition_km", &CannonballSRP::transition_km);
    obj.def("illumination", &CannonballSRP::illumination_value, py::arg("input"));
    Base::DenseBaseBuild(obj);
    obj.def("__call__", [](const CannonballSRP& f,
                            const GenericFunction<-1,-1>& input) {
      return GenericFunction<-1,-1>(f.eval(input));
    }, py::arg("input"));
  }
};

void BuildCannonballSRP(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<CannonballSRP>(m, "CannonballSRP");
}

}  // namespace ASSET

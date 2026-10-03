#include "ASSET_Extensions.h"

#include <cmath>
#include <stdexcept>

namespace ASSET {

/** Convert ECEF Cartesian position [km] to [height_km, latitude_rad, longitude_rad].
 *
 * The iteration and polar threshold match ecef_to_geodetic_wgs84 in the
 * released Fortran propagator. Forward automatic differentiation supplies
 * derivatives away from the polar/longitude singularity. Longitude is atan2
 * wrapped to [-pi,pi]. The origin is rejected because geodetic coordinates are
 * undefined there.
 */
struct GeodeticWGS84
    : VectorFunction<GeodeticWGS84, 3, 3, AutodiffFwd, AutodiffFwd> {
  using Base = VectorFunction<GeodeticWGS84, 3, 3, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> x,
                           ConstVectorBaseRef<OutType> fx_) const {
    using Scalar = typename InType::Scalar;
    auto& fx = fx_.const_cast_derived();
    constexpr double a_km = 6378.137;
    constexpr double flattening = 1.0 / 298.257223563;
    constexpr double e2 = flattening * (2.0 - flattening);
    const Scalar p = sqrt(x[0]*x[0] + x[1]*x[1]);
    if (p <= Scalar(0.0) && x[2] == Scalar(0.0))
      throw std::invalid_argument("GeodeticWGS84 is undefined at the origin");
    fx[2] = atan2(x[1], x[0]);
    if (p < Scalar(1.0e-12)) {
      fx[1] = x[2] >= Scalar(0.0) ? Scalar(M_PI_2) : Scalar(-M_PI_2);
      fx[0] = abs(x[2]) - Scalar(a_km * (1.0 - flattening));
      return;
    }
    Scalar latitude = atan2(x[2], p * Scalar(1.0 - e2));
    Scalar latitude_new = latitude;
    for (int iteration = 0; iteration < 8; ++iteration) {
      const Scalar sin_lat = sin(latitude);
      const Scalar n = Scalar(a_km) / sqrt(Scalar(1.0) - Scalar(e2)*sin_lat*sin_lat);
      const Scalar height = p/cos(latitude) - n;
      latitude_new = atan2(x[2], p*(Scalar(1.0) - Scalar(e2)*n/(n + height)));
      if (abs(latitude_new - latitude) < Scalar(1.0e-13)) break;
      latitude = latitude_new;
    }
    const Scalar sin_lat = sin(latitude_new);
    const Scalar n = Scalar(a_km) / sqrt(Scalar(1.0) - Scalar(e2)*sin_lat*sin_lat);
    fx[0] = p/cos(latitude_new) - n;
    fx[1] = latitude_new;
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<GeodeticWGS84>(m, name,
        "WGS84 ECEF [km] to [height_km, geodetic_latitude_rad, longitude_rad].");
    obj.def(py::init<>());
    Base::DenseBaseBuild(obj);
    obj.def("__call__", [](const GeodeticWGS84& f,
                           const GenericFunction<-1, -1>& input) {
      return GenericFunction<-1, -1>(f.eval(input));
    }, py::arg("position_ecef_km"));
  }
};

void BuildGeodeticWGS84(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<GeodeticWGS84>(m, "GeodeticWGS84");
}

}  // namespace ASSET

#include "ASSET_Extensions.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace ASSET {

/**
 * Aerodynamic drag used by the released LEO environment reference.
 *
 * Input layout (11 values):
 *   [0:3]   inertial/ICRF position, km
 *   [3:6]   inertial/ICRF velocity, km/s
 *   [6]     spacecraft mass, kg
 *   [7:10]  body-to-inertial 3-2-1 roll, pitch, yaw, rad
 *   [10]    atmospheric mass density at the spacecraft, kg/m^3
 * Output: inertial acceleration, km/s^2.
 *
 * This class deliberately does not calculate density. The caller must obtain
 * density from an atmosphere model using Earth-fixed geodetic coordinates.
 * Keeping that boundary explicit permits direct validation of the force law
 * independently of Earth orientation and atmosphere data.
 *
 * Atmospheric velocity follows the reference approximation omega_Earth x r;
 * winds and the full derivative of the ICRF/ECEF transform are not included.
 * If effective_area_m2 > 0, Cd*A is constant and attitude is unused. If it is
 * zero, the reference's six rectangular panels and fixed Cd=2.2 are used.
 * drag_coefficient applies only to the effective-area mode, matching Fortran.
 *
 * Forward automatic differentiation supplies values, Jacobians, adjoint
 * gradients, and adjoint Hessians. Panel exposure max(0,n.flow) is continuous
 * but nonsmooth at zero exposure; derivatives are one-sided at that boundary.
 * The implementation owns no mutable state and is safe for concurrent calls.
 */
struct PanelDrag : VectorFunction<PanelDrag, 11, 3, AutodiffFwd, AutodiffFwd> {
  using Base = VectorFunction<PanelDrag, 11, 3, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base)
  static constexpr bool IsVectorizable = false;

  double effective_area_m2;
  double drag_coefficient;
  double earth_rotation_rate_rad_s;

  PanelDrag(double effective_area, double cd, double earth_rate)
      : effective_area_m2(effective_area), drag_coefficient(cd),
        earth_rotation_rate_rad_s(earth_rate) {
    if (!std::isfinite(effective_area) || effective_area < 0.0)
      throw std::invalid_argument("PanelDrag effective_area_m2 must be finite and nonnegative");
    if (!std::isfinite(cd) || cd <= 0.0)
      throw std::invalid_argument("PanelDrag drag_coefficient must be finite and positive");
    if (!std::isfinite(earth_rate))
      throw std::invalid_argument("PanelDrag earth_rotation_rate_rad_s must be finite");
  }

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> x,
                           ConstVectorBaseRef<OutType> fx_) const {
    using Scalar = typename InType::Scalar;
    auto& fx = fx_.const_cast_derived();
    // The reference evaluates aerodynamic momentum in SI and converts its
    // acceleration back to km/s^2 only at the end.
    const auto r_m = x.template head<3>() * Scalar(1000.0);
    const auto v_mps = x.template segment<3>(3) * Scalar(1000.0);
    const Scalar mass = x[6];
    const Scalar rho = x[10];
    if (mass <= Scalar(0.0))
      throw std::invalid_argument("PanelDrag mass must be positive");
    if (rho < Scalar(0.0))
      throw std::invalid_argument("PanelDrag density must be nonnegative");

    // omega x r for omega=[0,0,earth_rotation_rate]. This approximation is
    // evaluated in inertial Cartesian components in the Fortran reference.
    Vector3<Scalar> v_atm;
    v_atm[0] = -Scalar(earth_rotation_rate_rad_s) * r_m[1];
    v_atm[1] =  Scalar(earth_rotation_rate_rad_s) * r_m[0];
    v_atm[2] = Scalar(0.0);
    const Vector3<Scalar> v_rel = v_mps - v_atm;
    const Scalar speed = v_rel.norm();

    Scalar cda;
    if (effective_area_m2 > 0.0) {
      cda = Scalar(drag_coefficient * effective_area_m2);
    } else {
      // Columns of c_bi are the body-frame unit axes expressed in ICRF. The
      // assignment below exactly matches Fortran column-major RESHAPE data.
      const Scalar roll = x[7], pitch = x[8], yaw = x[9];
      const Scalar cr = cos(roll), sr = sin(roll);
      const Scalar cp = cos(pitch), sp = sin(pitch);
      const Scalar cy = cos(yaw), sy = sin(yaw);
      Eigen::Matrix<Scalar, 3, 3> c_bi;
      c_bi << cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
              sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
                 -sp,                cp*sr,                cp*cr;
      Vector3<Scalar> flow_hat;
      const Scalar flow_denominator = speed > Scalar(1.0e-12) ? speed : Scalar(1.0e-12);
      flow_hat = -v_rel / flow_denominator;
      // Panel order is +X,-X,+Y,-Y,+Z,-Z; area is m^2. The panel Cd is
      // intentionally fixed at 2.2 because that is the reference contract.
      constexpr std::array<double, 6> areas{12.0, 12.0, 18.0, 18.0, 24.0, 24.0};
      cda = Scalar(0.0);
      for (int axis = 0; axis < 3; ++axis) {
        for (int sign : {1, -1}) {
          const Scalar exposure = Scalar(sign) * c_bi.col(axis).dot(flow_hat);
          if (exposure > Scalar(0.0))
            cda += Scalar(2.2 * areas[2*axis + (sign < 0)]) * exposure;
        }
      }
    }
    // -1/2 rho (Cd*A)/m |v_rel| v_rel [m/s^2], then /1000 [km/s^2].
    fx = -Scalar(0.5 / 1000.0) * rho * cda / mass * speed * v_rel;
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<PanelDrag>(m, name,
        "Reference panel/effective-area drag; inputs [r_km,v_km_s,mass,RPY,rho].");
    obj.def(py::init<double, double, double>(),
            py::arg("effective_area_m2") = 0.0,
            py::arg("drag_coefficient") = 2.2,
            py::arg("earth_rotation_rate_rad_s") = 7.2921150e-5);
    obj.def_readonly("effective_area_m2", &PanelDrag::effective_area_m2);
    obj.def_readonly("drag_coefficient", &PanelDrag::drag_coefficient);
    obj.def_readonly("earth_rotation_rate_rad_s", &PanelDrag::earth_rotation_rate_rad_s);
    Base::DenseBaseBuild(obj);
    obj.def("__call__", [](const PanelDrag& f, const GenericFunction<-1, -1>& input) {
      return GenericFunction<-1, -1>(f.eval(input));
    }, py::arg("input"));
  }
};

void BuildPanelDrag(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<PanelDrag>(m, "PanelDrag");
}

}  // namespace ASSET

#pragma once

// ============================================================
// FILE: SpiceForceFrame.h
//
// Generic fused body-frame wrapper for any ForceField3 kernel.
//
//   inputs : [ET, x, y, z]  (inertial position)
//   output : a = R(ET)^T g(R(ET) r)           (inertial acceleration)
//
// With G = dg/dr_body, w = Rdot r:
//   da/dET = Rdot^T g + R^T G w
//   da/dr  = R^T G R
//
// One sxform call per evaluation. The kernel is cloned at construction, so
// the node owns its own copy and can be composed freely in the graph.
// ============================================================

#include "ForceField3.h"
#include "SpicePxform.h"

namespace ASSET {

struct SpiceForceFrame : VectorFunction<SpiceForceFrame, 4, 3, Analytic, Analytic> {
  using Base = VectorFunction<SpiceForceFrame, 4, 3, Analytic, Analytic>;
  DENSE_FUNCTION_BASE_TYPES(Base)

  std::string from_frame;
  std::string to_frame;
  std::shared_ptr<const ForceField3> field;

  SpiceForceFrame() = default;
  SpiceForceFrame(const ForceField3& kernel, const std::string& from,
                  const std::string& to)
      : from_frame(from), to_frame(to), field(kernel.clone()) {}

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> x,
                           ConstVectorBaseRef<OutType> fx_) const {
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    double X[6][6];
    spice_detail::sxform_locked(from_frame, to_frame, double(x[0]), X);
    Eigen::Vector3d rb, g;
    for (int i = 0; i < 3; i++) {
      rb[i] = 0.0;
      for (int j = 0; j < 3; j++) rb[i] += X[i][j] * double(x[j + 1]);
    }
    g = field->accel(rb);
    for (int i = 0; i < 3; i++) {
      fx[i] = 0.0;
      for (int j = 0; j < 3; j++) fx[i] += X[j][i] * g[j];
    }
  }

  template<class InType, class OutType, class JacType>
  inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                    ConstVectorBaseRef<OutType> fx_,
                                    ConstMatrixBaseRef<JacType> jx_) const {
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
    double X[6][6];
    spice_detail::sxform_locked(from_frame, to_frame, double(x[0]), X);

    double r[3];
    for (int k = 0; k < 3; k++) r[k] = double(x[k + 1]);

    Eigen::Vector3d rb, w, g;
    Eigen::Matrix3d G;
    for (int i = 0; i < 3; i++) {
      rb[i] = 0.0;
      w[i] = 0.0;
      for (int j = 0; j < 3; j++) {
        rb[i] += X[i][j] * r[j];
        w[i] += X[i + 3][j] * r[j];
      }
    }
    field->accel_jacobian(rb, g, G);

    Eigen::Vector3d u = G * w;  // G * Rdot * r
    for (int i = 0; i < 3; i++) {
      fx[i] = 0.0;
      double dT = 0.0;
      for (int j = 0; j < 3; j++) {
        fx[i] += X[j][i] * g[j];
        dT += X[j + 3][i] * g[j] + X[j][i] * u[j];
      }
      jx(i, 0) = dT;
    }

    // M = R^T G R
    Eigen::Matrix3d P;
    for (int i = 0; i < 3; i++)
      for (int k = 0; k < 3; k++) {
        P(i, k) = 0.0;
        for (int m = 0; m < 3; m++) P(i, k) += G(i, m) * X[m][k];
      }
    for (int i = 0; i < 3; i++)
      for (int k = 0; k < 3; k++) {
        double M = 0.0;
        for (int j = 0; j < 3; j++) M += X[j][i] * P(j, k);
        jx(i, k + 1) = M;
      }
  }

  template<class InType, class OutType, class JacType,
           class AdjGradType, class AdjHessType, class AdjVarType>
  inline void compute_jacobian_adjointgradient_adjointhessian_impl(
      const Eigen::MatrixBase<InType>& x,
      Eigen::MatrixBase<OutType> const& fx_,
      Eigen::MatrixBase<JacType> const& jx_,
      Eigen::MatrixBase<AdjGradType> const& adjgrad_,
      Eigen::MatrixBase<AdjHessType> const& adjhess_,
      const Eigen::MatrixBase<AdjVarType>& adjvars) const {
    spice_adjoint_terms(*this, x, fx_, jx_, adjgrad_, adjhess_, adjvars);
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<SpiceForceFrame>(m, name,
        "Fused SPICE body-frame wrapper: [ET, x, y, z] (inertial) -> inertial "
        "acceleration of any ForceField3 kernel (e.g. PinesGravity).")
                   .def(py::init<const ForceField3&, const std::string&, const std::string&>(),
                        py::arg("field"), py::arg("from_frame"), py::arg("to_frame"));
    Base::DenseBaseBuild(obj);
    DefineSpiceListEval(obj);
  }
};

}  // namespace ASSET

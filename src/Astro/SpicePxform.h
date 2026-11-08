#pragma once

// ============================================================
// FILE: SpicePxform.h
//
// Native ASSET VectorFunctions for SPICE frame transformations.
//
//   SpicePxform       : ET -> R_{from->to}(ET), flattened row-major (9)
//                       Jacobian: dR/dET, flattened (9x1)
//
//   SpicePxformApply  : [ET, x, y, z] -> R_{from->to}(ET) * [x, y, z]
//                       Jacobian: [Rdot*r | R]  (3x4)
//
//   SpiceForceFrame   : [ET, x, y, z] (inertial) -> inertial
//                       any ForceField3 evaluated in a body frame (see SpiceForceFrame.h)
//
// Rdot is taken from the lower-left block of the 6x6 state
// transformation returned by sxform_c:  X = [[R, 0], [Rdot, R]].
//
// CSPICE is not thread-safe, so all SPICE calls are serialized
// through a mutex.  Kernels must be loaded with SpiceLoadKernels
// before evaluation.
// ============================================================

#include "VectorFunctions/ASSET_VectorFunctions.h"

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

extern "C" {
void furnsh_c(const char* file);
void sxform_c(const char* from, const char* to, double et, double xform[6][6]);
void pxform_c(const char* from, const char* to, double et, double rotate[3][3]);
int failed_c(void);
void getmsg_c(const char* option, int msglen, char* msg);
void reset_c(void);
void erract_c(const char* op, int actlen, char* action);
void errprt_c(const char* op, int listlen, char* list);
}

namespace ASSET {

namespace spice_detail {

// CSPICE aborts the process on error by default.  Switch to returning
// errors and rethrow them as C++ exceptions instead.
inline std::mutex& spice_mutex() {
  static std::mutex m;
  return m;
}

inline void check_spice_error() {
  if (failed_c()) {
    char msg[1841];
    getmsg_c("LONG", 1840, msg);
    reset_c();
    throw std::runtime_error(std::string("SPICE error: ") + msg);
  }
}

// Returns the 6x6 state transform X(et) from 'from' to 'to'.
inline void sxform_locked(const std::string& from,
                          const std::string& to,
                          double et,
                          double X[6][6]) {
  std::lock_guard<std::mutex> lock(spice_mutex());
  sxform_c(from.c_str(), to.c_str(), et, X);
  check_spice_error();
}

inline void load_kernel(const std::string& file) {
  std::lock_guard<std::mutex> lock(spice_mutex());
  furnsh_c(file.c_str());
  check_spice_error();
}

inline void configure_spice_errors() {
  std::lock_guard<std::mutex> lock(spice_mutex());
  char op[] = "SET";
  char action[] = "RETURN";
  erract_c(op, sizeof(action), action);
  char listop[] = "SET";
  char listtype[] = "NONE";
  errprt_c(listop, sizeof(listtype), listtype);
}

}  // namespace spice_detail

// Lets a native node be applied to expression lists, e.g. node([t, x, y, z]),
// so it can be composed into ODE and VectorFunction graphs.
template<class FType>
void DefineSpiceListEval(py::class_<FType>& obj) {
  using Gen = GenericFunction<-1, -1>;
  obj.def("__call__", [](const FType& fun, const std::vector<Gen>& funcs) {
    return Gen(fun.eval(DynamicStack(funcs)));
  });
}

// Second-order adjoint terms for Analytic VectorFunctions whose only
// second derivatives come from finite differences of the exact Jacobian.
// adjgrad = J^T l is exact; adjhess uses central differences in each
// input of J (step h), which is exact for functions linear in the vector.
template<class F, class InType, class OutType, class JacType,
         class AdjGradType, class AdjHessType, class AdjVarType>
inline void spice_adjoint_terms(const F& f,
                                const Eigen::MatrixBase<InType>& x,
                                const Eigen::MatrixBase<OutType>& fx_,
                                const Eigen::MatrixBase<JacType>& jx_,
                                const Eigen::MatrixBase<AdjGradType>& adjgrad_,
                                const Eigen::MatrixBase<AdjHessType>& adjhess_,
                                const Eigen::MatrixBase<AdjVarType>& adjvars) {
  const double h = 1.0;
  const int n = x.rows();
  const int m = fx_.rows();

  f.compute_jacobian_impl(x, fx_, jx_);
  Eigen::MatrixXd J0 = jx_.derived();

  adjgrad_.const_cast_derived() = J0.transpose() * adjvars;

  Eigen::MatrixXd H(n, n);
  Eigen::VectorXd xv = x.derived();
  Eigen::VectorXd xp(n), xm(n), fp(m), fm(m);
  Eigen::MatrixXd Jp(m, n), Jm(m, n);
  for (int k = 0; k < n; k++) {
    xp = xv;
    xm = xv;
    xp(k) += h;
    xm(k) -= h;
    f.compute_jacobian_impl(xp, fp, Jp);
    f.compute_jacobian_impl(xm, fm, Jm);
    Eigen::MatrixXd Dk = (Jp - Jm) / (2.0 * h);
    H.col(k) = Dk.transpose() * adjvars;
  }
  adjhess_.const_cast_derived() = H;
}

// ------------------------------------------------------------
// SpicePxform: ET -> flattened rotation R (9 outputs)
// ------------------------------------------------------------
struct SpicePxform : VectorFunction<SpicePxform, 1, 9, Analytic, Analytic> {
  using Base = VectorFunction<SpicePxform, 1, 9, Analytic, Analytic>;
  DENSE_FUNCTION_BASE_TYPES(Base)

  std::string from_frame;
  std::string to_frame;

  SpicePxform() = default;
  SpicePxform(const std::string& from, const std::string& to)
      : from_frame(from), to_frame(to) {}

  // Only evaluated with double: the SPICE state transform cannot
  // carry autodiff scalars.  Analytic derivatives use compute_jacobian_impl.
  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> x,
                           ConstVectorBaseRef<OutType> fx_) const {
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    double X[6][6];
    spice_detail::sxform_locked(from_frame, to_frame, double(x[0]), X);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        fx[3 * i + j] = X[i][j];
  }

  template<class InType, class OutType, class JacType>
  inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                    ConstVectorBaseRef<OutType> fx_,
                                    ConstMatrixBaseRef<JacType> jx_) const {
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
    double X[6][6];
    spice_detail::sxform_locked(from_frame, to_frame, double(x[0]), X);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        fx[3 * i + j] = X[i][j];
        jx(3 * i + j, 0) = X[i + 3][j];
      }
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

  SpicePxform inverse() const { return SpicePxform(to_frame, from_frame); }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<SpicePxform>(m, name)
                   .def(py::init<const std::string&, const std::string&>())
                   .def("inverse", &SpicePxform::inverse);
    Base::DenseBaseBuild(obj);
    DefineSpiceListEval(obj);
  }
};

// ------------------------------------------------------------
// SpicePxformApply: [ET, x, y, z] -> R(ET) * [x, y, z]
// ------------------------------------------------------------
struct SpicePxformApply
    : VectorFunction<SpicePxformApply, 4, 3, Analytic, Analytic> {
  using Base = VectorFunction<SpicePxformApply, 4, 3, Analytic, Analytic>;
  DENSE_FUNCTION_BASE_TYPES(Base)

  std::string from_frame;
  std::string to_frame;

  SpicePxformApply() = default;
  SpicePxformApply(const std::string& from, const std::string& to)
      : from_frame(from), to_frame(to) {}

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> x,
                           ConstVectorBaseRef<OutType> fx_) const {
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    double X[6][6];
    spice_detail::sxform_locked(from_frame, to_frame, double(x[0]), X);
    for (int i = 0; i < 3; i++) {
      fx[i] = 0.0;
      for (int j = 0; j < 3; j++)
        fx[i] += X[i][j] * double(x[j + 1]);
    }
  }

  // d/dET [R(ET) r] = Rdot(ET) r ; d/dr [R(ET) r] = R(ET)
  template<class InType, class OutType, class JacType>
  inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                    ConstVectorBaseRef<OutType> fx_,
                                    ConstMatrixBaseRef<JacType> jx_) const {
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
    double X[6][6];
    spice_detail::sxform_locked(from_frame, to_frame, double(x[0]), X);
    for (int i = 0; i < 3; i++) {
      fx[i] = 0.0;
      jx(i, 0) = 0.0;
      for (int j = 0; j < 3; j++) {
        const double r = double(x[j + 1]);
        fx[i] += X[i][j] * r;
        jx(i, 0) += X[i + 3][j] * r;
        jx(i, j + 1) = X[i][j];
      }
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

  SpicePxformApply inverse() const {
    return SpicePxformApply(to_frame, from_frame);
  }

  static void Build(py::module& m, const char* name) {
    auto obj = py::class_<SpicePxformApply>(m, name)
                   .def(py::init<const std::string&, const std::string&>())
                   .def("inverse", &SpicePxformApply::inverse);
    Base::DenseBaseBuild(obj);
    DefineSpiceListEval(obj);
  }
};

}  // namespace ASSET

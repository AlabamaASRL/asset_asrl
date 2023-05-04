#pragma once

#include "VectorFunction.h"

namespace ASSET {

  //! Declaration of \struct NormalizedPower_Impl
  template<class Derived, int IR, int PW>
  struct NormalizedPower_Impl;

  template<int IR>
  struct Normalized : NormalizedPower_Impl<Normalized<IR>, IR, 1> {
    using Base = NormalizedPower_Impl<Normalized<IR>, IR, 1>;

    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
  };

  template<int IR, int PW>
  struct NormalizedPower : NormalizedPower_Impl<NormalizedPower<IR, PW>, IR, PW> {
    using Base = NormalizedPower_Impl<NormalizedPower<IR, PW>, IR, PW>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
  };

  template<class Derived, int IR, int PW>
  struct NormalizedPower_Impl : VectorFunction<Derived, IR, IR> {
    using Base = VectorFunction<Derived, IR, IR>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    static constexpr int power = PW;
    static constexpr int pp2 = power + 2;
    static constexpr int pp4 = power + 4;

    using Base::compute;
    static const bool IsVectorizable = true;

    NormalizedPower_Impl() {
    }
    NormalizedPower_Impl(int irows) {
      this->setIORows(irows, irows);
    }

    static void Build(py::module &m, const char *name) {
      auto obj = py::class_<Derived>(m, name);
      obj.def(py::init<int>());
      if constexpr (IR > 0) {
        obj.def(py::init<>());
      }
      Base::DenseBaseBuild(obj);
    }

    template<class Scalar>
    inline Scalar calc_pow_n(Scalar n) const {
      Scalar pow_n;
      if constexpr (power == 1)
        pow_n = n;
      else if constexpr (power == 2)
        pow_n = n * n;
      else if constexpr (power == 3)
        pow_n = n * n * n;
      else if constexpr (power == 4)
        pow_n = n * n * n * n;
      else if constexpr (power == 5)
        pow_n = n * n * n * n * n;
      else
        pow_n = pow(n, power);
      return pow_n;
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      using std::sqrt;
      Scalar n;
      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        n = sqrt(x.dot(x));
      } else {
        n = x.norm();
      }
      Scalar pow_n = this->calc_pow_n(n);


      Scalar np = 1.0 / (pow_n);
      fx = x * np;
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      using std::sqrt;
      Scalar n;
      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        n = sqrt(x.dot(x));
      } else {
        n = x.norm();
      }

      Scalar pow_n = this->calc_pow_n(n);


      Scalar pow_n_2 = pow_n * n * n;

      Scalar np = 1.0 / (pow_n);
      Scalar npd = -power / pow_n_2;
      fx = x * (np);

      jx.diagonal().setConstant(np);

      if constexpr (InType::RowsAtCompileTime == Eigen::Dynamic) {
        for (int i = 0; i < this->IRows(); i++) {
          jx.col(i) += x * (npd * x[i]);
        }
      } else {
        jx.noalias() += (x * npd) * x.transpose();
      }
    }

    //! Analytic second derivative
    template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_,
        Eigen::MatrixBase<AdjGradType> const &adjgrad_,
        Eigen::MatrixBase<AdjHessType> const &adjhess_,
        const Eigen::MatrixBase<AdjVarType> &adjvars) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      Eigen::MatrixBase<AdjGradType> &adjgrad = adjgrad_.const_cast_derived();
      Eigen::MatrixBase<AdjHessType> &adjhess = adjhess_.const_cast_derived();

      const int irows = this->IRows();
      auto Impl = [&](auto &xxt) {
        using std::sqrt;
        Scalar n;
        if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
          n = sqrt(x.dot(x));
        } else {
          n = x.norm();
        }

        Scalar pow_n = this->calc_pow_n(n);

        Scalar pow_n_2 = pow_n * n * n;
        Scalar pow_n_4 = pow_n_2 * n * n;

        Scalar np = 1.0 / (pow_n);
        Scalar npd = -power / pow_n_2;
        Scalar npdd = power * pp2 / pow_n_4;
        Scalar K = x.dot(adjvars);


        if constexpr (InType::RowsAtCompileTime == Eigen::Dynamic) {
          for (int i = 0; i < irows; i++) {
            xxt.col(i) = x * x[i];
          }
        } else {
          xxt.noalias() = x * x.transpose();
        }

        fx = x * (np);

        jx.diagonal().setConstant((np));
        jx += xxt * (npd);

        adjgrad.noalias() = (adjvars.transpose() * jx).transpose();
        adjhess.diagonal().setConstant(npd);
        adjhess.noalias() += xxt * npdd;
        adjhess *= K;

        if constexpr (InType::RowsAtCompileTime == Eigen::Dynamic) {
          for (int i = 0; i < irows; i++) {
            adjhess.col(i) += (x * adjvars[i] + adjvars * x[i]) * npd;
          }
        } else {
          adjhess.noalias() += (x * adjvars.transpose() + adjvars * x.transpose()) * npd;
        }
      };


      ASSET::MemoryManager::allocate_run(irows, Impl, TempSpec<Jacobian<Scalar>>(irows, irows));
    }
  };

}  // namespace ASSET

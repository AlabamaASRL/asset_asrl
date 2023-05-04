#pragma once
#include "VectorFunctions/VectorFunction.h"

namespace ASSET {

  template<int USZ>
  struct LockArgs : VectorFunction<LockArgs<USZ>, USZ, USZ> {
    using Base = VectorFunction<LockArgs<USZ>, USZ, USZ>;

    static const bool IsLinearFunction = true;

    LockArgs(int usz) {
      this->setIORows(usz, usz);
    }
    LockArgs() {
    }
    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      // Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      // Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      for (int i = 0; i < this->IRows(); i++)
        jx(i, i) = Scalar(1.0);
    }

    template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType>& x,
        Eigen::MatrixBase<OutType> const& fx_,
        Eigen::MatrixBase<JacType> const& jx_,
        Eigen::MatrixBase<AdjGradType> const& adjgrad_,
        Eigen::MatrixBase<AdjHessType> const& adjhess_,
        const Eigen::MatrixBase<AdjVarType>& adjvars) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      // Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      for (int i = 0; i < this->IRows(); i++)
        jx(i, i) = Scalar(1.0);
      adjgrad = jx * adjvars;
    }
  };

}  // namespace ASSET
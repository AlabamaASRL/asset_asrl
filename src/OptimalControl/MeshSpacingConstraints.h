#pragma once

#include "LGLCoeffs.h"
#include "VectorFunctions/VectorFunction.h"

namespace ASSET {

  struct SingleMeshSpacing : VectorFunction<SingleMeshSpacing, 3, 1> {
    using Base = VectorFunction<SingleMeshSpacing, 3, 1>;
    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;
    template<class Scalar>
    using Jacobian = typename Base::template Jacobian<Scalar>;
    template<class Scalar>
    using Hessian = typename Base::template Hessian<Scalar>;

    double CardinalSpacing;
    double Scale = 1.0;
    static const bool IsLinearFunction = true;

    SingleMeshSpacing() {
    }
    SingleMeshSpacing(double cs) {
      CardinalSpacing = cs;
    }

    void setSpacing(double cs) {
      CardinalSpacing = cs;
    }
    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Scalar h = x[2] - x[0];
      fx[0] = CardinalSpacing * h - (x[1] - x[0]);
      fx[0] *= Scale;
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Scalar h = x[2] - x[0];
      fx[0] = CardinalSpacing * h - (x[1] - x[0]);
      fx[0] *= Scale;

      jx(0, 0) = (1.0 - CardinalSpacing);
      jx(0, 1) = -1.0;
      jx(0, 2) = CardinalSpacing;
      jx *= Scale;
    }

    template<class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(const Eigen::MatrixBase<InType>& x,
                                                 Eigen::MatrixBase<OutType> const& fx_,
                                                 Eigen::MatrixBase<JacType> const& jx_,
                                                 Eigen::MatrixBase<AdjGradType> const& adjgrad_,
                                                 const Eigen::MatrixBase<AdjVarType>& adjvars) const {
      this->compute_jacobian(x, fx_, jx_);
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      adjgrad = (adjvars.transpose() * jx_).transpose();
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
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();

      Scalar h = x[2] - x[0];
      fx[0] = CardinalSpacing * h - (x[1] - x[0]);
      fx[0] *= Scale;

      jx(0, 0) = (1.0 - CardinalSpacing);
      jx(0, 1) = -1.0;
      jx(0, 2) = CardinalSpacing;
      jx *= Scale;
      adjgrad = (adjvars.transpose() * jx_).transpose();
    }
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template<int CSC>
  struct LGLMeshSpacing : VectorFunction<LGLMeshSpacing<CSC>, CSC, CSC - 2> {
    using Base = VectorFunction<LGLMeshSpacing<CSC>, CSC, CSC - 2>;
    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;
    template<class Scalar>
    using Jacobian = typename Base::template Jacobian<Scalar>;
    template<class Scalar>
    using Hessian = typename Base::template Hessian<Scalar>;
    using Coeffs = LGLCoeffs<CSC>;

    template<class InType, class OutType>
    inline void compute(const Eigen::MatrixBase<InType>& x, Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Scalar h = x[CSC - 1] - x[0];
      for (int i = 0; i < (CSC - 2); i++) {
        fx[i] = Coeffs::CardinalSpacings[i + 1] - (x[1 + i] - x[0]) / h;
      }
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Scalar h = x[CSC - 1] - x[0];
      for (int i = 0; i < (CSC - 2); i++) {
        fx[i] = Coeffs::CardinalSpacings[i + 1] - (x[1 + i] - x[0]) / h;
        jx(i, i + 1) = -1.0 / h;
        jx(i, 0) = 1.0 / h - (x[1 + i] - x[0]) / (h * h);
        jx(i, CSC - 1) = (x[1 + i] - x[0]) / (h * h);
      }
    }

    template<class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(const Eigen::MatrixBase<InType>& x,
                                                 Eigen::MatrixBase<OutType> const& fx_,
                                                 Eigen::MatrixBase<JacType> const& jx_,
                                                 Eigen::MatrixBase<AdjGradType> const& adjgrad_,
                                                 const Eigen::MatrixBase<AdjVarType>& adjvars) const {
      this->compute_jacobian(x, fx_, jx_);
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      adjgrad = (adjvars.transpose() * jx_).transpose();
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
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      Eigen::MatrixBase<AdjHessType>& adjhess = adjhess_.const_cast_derived();

      Scalar h = x[CSC - 1] - x[0];
      Scalar h2 = h * h;
      Scalar h3 = h2 * h;
      for (int i = 0; i < (CSC - 2); i++) {
        fx[i] = Coeffs::CardinalSpacings[i + 1] - (x[1 + i] - x[0]) / h;
        jx(i, i + 1) = -1.0 / h;
        jx(i, 0) = 1.0 / h - (x[1 + i] - x[0]) / (h2);
        jx(i, CSC - 1) = (x[1 + i] - x[0]) / (h2);
        adjhess(0, i + 1) += -adjvars[i] * 1.0 / (h2);
        adjhess(i + 1, 0) += -adjvars[i] * 1.0 / (h2);

        adjhess(CSC - 1, i + 1) += adjvars[i] * 1.0 / (h2);
        adjhess(i + 1, CSC - 1) += adjvars[i] * 1.0 / (h2);

        adjhess(0, 0) += adjvars[i] * 2.0 / (h2) -adjvars[i] * 2.0 * (x[1 + i] - x[0]) / (h3);
        adjhess(CSC - 1, CSC - 1) += -adjvars[i] * 2.0 * (x[1 + i] - x[0]) / (h3);

        adjhess(0, CSC - 1) += adjvars[i] * (2.0 * (x[1 + i] - x[0]) / (h3) -1.0 / h2);
        adjhess(CSC - 1, 0) += adjvars[i] * (2.0 * (x[1 + i] - x[0]) / (h3) -1.0 / h2);
      }
      adjgrad = (adjvars.transpose() * jx_).transpose();
    }
  };

}  // namespace ASSET

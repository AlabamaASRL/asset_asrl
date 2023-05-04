/*
File Name:

File Description:


////////////////////////////////////////////////////////////////////////////////

Original File Developer : James B. Pezent - jbpezent - jbpezent@crimson.ua.edu

Current File Maintainers:
    1. James B. Pezent - jbpezent         - jbpezent@crimson.ua.edu
    2. Full Name       - GitHub User Name - Current Email
    3. ....


Usage of this source code is governed by the license found
in the LICENSE file in ASSET's top level directory.

*/

#pragma once
#include "VectorFunction.h"

namespace ASSET {


  struct ArcTan2Op : VectorFunction<ArcTan2Op, 2, 1, Analytic, Analytic> {

    using Base = VectorFunction<ArcTan2Op, 2, 1, Analytic, Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    static const bool IsVectorizable = true;

    template<class Scalar>
    static Scalar calcArcTan2(Scalar yy, Scalar xx) {
      Scalar fx;
      if constexpr (Is_SuperScalar<Scalar>::value) {
        for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
          fx[i] = atan2(yy[i], xx[i]);
        }
      } else {
        fx = atan2(yy, xx);
      }
      return fx;
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      Scalar yy = x[0];
      Scalar xx = x[1];
      fx[0] = calcArcTan2(yy, xx);
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      Scalar yy = x[0];
      Scalar xx = x[1];
      fx[0] = calcArcTan2(yy, xx);

      Scalar denom = xx * xx + yy * yy;

      jx(0, 0) = xx / denom;
      jx(0, 1) = -yy / denom;
    }
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
        ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_,
        ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      Scalar yy = x[0];
      Scalar xx = x[1];
      fx[0] = calcArcTan2(yy, xx);

      Scalar denom = xx * xx + yy * yy;

      jx(0, 0) = xx / denom;
      jx(0, 1) = -yy / denom;

      adjgrad[0] = adjvars[0] * xx / denom;
      adjgrad[1] = -adjvars[0] * yy / denom;

      adjhess(0, 0) = -2 * xx * yy / (denom * denom);
      adjhess(1, 1) = 2 * xx * yy / (denom * denom);

      adjhess(0, 1) = 1 / denom - 2 * xx * xx / (denom * denom);
      adjhess(1, 0) = -1 / denom + 2 * yy * yy / (denom * denom);

      adjhess *= adjvars[0];
    }
  };


  template<class YFunc, class XFunc>
  struct ArcTan2Impl {
    static auto Definition(YFunc yf, XFunc xf) {
      return ArcTan2Op().eval(StackedOutputs {yf, xf});
    }
  };


}  // namespace ASSET
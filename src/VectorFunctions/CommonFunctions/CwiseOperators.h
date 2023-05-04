/*
File Name: CwiseOperators.h

File Description: Defines ASSET all Vector Functions which simply apply a common
coefficient (Cwise) operation/functio to all elements of input vectors.

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

  template<class Derived, class Func>
  struct CwiseFunctionOperator;

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Func>
  struct CwiseSin : CwiseFunctionOperator<CwiseSin<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseSin<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().sin();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstVectorBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().sin();
      jx = x.array().cos();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstVectorBaseRef<JacType> jx_,
                                               ConstVectorBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().sin();
      hx = -fx;
      jx = x.array().cos();
    }
  };
  /////////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Func>
  struct CwiseCos : CwiseFunctionOperator<CwiseCos<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseCos<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().cos();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().cos();
      jx.derived() = -x.array().sin();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().cos();
      hx.derived() = -fx;
      jx.derived() = -x.array().sin();
    }
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Func>
  struct CwiseTan : CwiseFunctionOperator<CwiseTan<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseTan<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().tan();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().tan();
      jx.derived() = x.array().cos().square().inverse();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().tan();
      jx.derived() = x.array().cos().square().inverse();
      hx.derived() = fx.derived().cwiseProduct(jx.derived()) * Scalar(2.0);
    }
  };
  /////////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Func>
  struct CwiseArcSin : CwiseFunctionOperator<CwiseArcSin<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseArcSin<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
    static const bool IsVectorizable = false;

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().asin();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().asin();
      jx.derived().setOnes();
      jx.derived() = jx.derived() - x.cwiseProduct(x);
      jx.derived() = (jx.derived().cwiseSqrt()).cwiseInverse();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().asin();
      jx.derived().setOnes();
      jx.derived() = jx.derived() - x.cwiseProduct(x);
      jx.derived() = (jx.derived().cwiseSqrt()).cwiseInverse();
      hx.derived() = (jx.derived().array().cube()) * x.array();
    }
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Func>
  struct CwiseArcCos : CwiseFunctionOperator<CwiseArcCos<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseArcCos<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
    static const bool IsVectorizable = false;

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().acos();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().acos();
      jx.derived().setOnes();
      jx.derived() = jx.derived() - x.cwiseProduct(x);
      jx.derived() = -(jx.derived().cwiseSqrt()).cwiseInverse();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().acos();
      jx.derived().setOnes();
      jx.derived() = jx.derived() - x.cwiseProduct(x);

      jx.derived() = -(jx.derived().cwiseSqrt()).cwiseInverse();
      hx.derived() = (jx.derived().array().cube()) * x.array();
    }
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Func>
  struct CwiseArcTan : CwiseFunctionOperator<CwiseArcTan<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseArcTan<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
    static const bool IsVectorizable = true;

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().atan();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().atan();
      jx.derived().setOnes();
      jx.derived() = (jx.derived() + x.cwiseProduct(x)).cwiseInverse();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().atan();
      jx.derived().setOnes();
      jx.derived() = (jx.derived() + x.cwiseProduct(x)).cwiseInverse();
      hx.derived() = (jx.derived().array().square()) * x.array() * Scalar(-2.0);
    }
  };


  /////////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Func>
  struct CwiseSquare : CwiseFunctionOperator<CwiseSquare<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseSquare<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().square();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().square();
      jx.derived() = Scalar(2.0) * x;
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().square();
      jx.derived() = Scalar(2.0) * x;
      hx.derived().setConstant(Scalar(2.0));
    }
  };

  template<class Func>
  struct CwiseInverse : CwiseFunctionOperator<CwiseInverse<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseInverse<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().inverse();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().inverse();
      jx.derived() = -fx.cwiseProduct(fx);
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().inverse();
      jx.derived() = -fx.cwiseProduct(fx);
      hx.derived() = -Scalar(2.0) * jx.derived().cwiseProduct(fx);
    }
  };

  template<class Func>
  struct CwiseExp : CwiseFunctionOperator<CwiseExp<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseExp<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().exp();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().exp();
      jx.derived() = fx;
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().exp();
      jx.derived() = fx;
      hx.derived() = fx;
    }
  };

  template<class Func>
  struct CwiseLog : CwiseFunctionOperator<CwiseLog<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseLog<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().log();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().log();
      jx.derived() = x.cwiseInverse();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().log();
      jx.derived() = x.cwiseInverse();
      hx.derived() = -(jx.derived().array().square());
    }
  };


  /////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Func>
  struct CwiseSinH : CwiseFunctionOperator<CwiseSinH<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseSinH<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().sinh();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstVectorBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().sinh();
      jx = x.array().cosh();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstVectorBaseRef<JacType> jx_,
                                               ConstVectorBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().sinh();
      hx = fx;
      jx = x.array().cosh();
    }
  };
  /////////////////////////////////////////////////////////////////////////////////////////////////////

  template<class Func>
  struct CwiseCosH : CwiseFunctionOperator<CwiseCosH<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseCosH<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().cosh();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().cosh();
      jx.derived() = x.array().sinh();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().cosh();
      hx.derived() = fx;
      jx.derived() = x.array().sinh();
    }
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Func>
  struct CwiseTanH : CwiseFunctionOperator<CwiseTanH<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseTanH<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().tanh();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().tanh();
      jx.derived().setOnes();
      jx.derived() = jx.derived().array() - fx.derived().array().square();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().tanh();
      jx.derived().setOnes();
      jx.derived() = jx.derived().array() - fx.derived().array().square();
      hx.derived() = Scalar(-2.0) * (fx.derived().cwiseProduct(jx.derived()));
    }
  };
  /////////////////////////////////////////////////////////////////////////////////////////////////////


  template<class Func>
  struct CwiseArcSinH : CwiseFunctionOperator<CwiseArcSinH<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseArcSinH<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().asinh();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstVectorBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();

      typedef typename InType::Scalar Scalar;


      fx = x.array().asinh();
      jx.setOnes();
      jx += x.cwiseProduct(x);

      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        if constexpr (Base::ORC == 1)
          jx[0] = sqrt(jx[0]);
        else {
          const int sz = x.size();
          for (int i = 0; i < sz; i++)
            jx[i] = sqrt(jx[i]);
        }
      } else {
        jx = jx.cwiseSqrt();
      }

      jx = jx.cwiseInverse();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstVectorBaseRef<JacType> jx_,
                                               ConstVectorBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<HessType> hx = hx_.const_cast_derived();

      typedef typename InType::Scalar Scalar;


      fx = x.array().asinh();
      jx.setOnes();
      jx += x.cwiseProduct(x);

      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        if constexpr (Base::ORC == 1)
          jx[0] = sqrt(jx[0]);
        else {
          const int sz = x.size();
          for (int i = 0; i < sz; i++)
            jx[i] = sqrt(jx[i]);
        }
      } else {
        jx = jx.cwiseSqrt();
      }


      jx = jx.cwiseInverse();
      hx = Scalar(-1.0) * jx.array().cube() * x.array();
    }
  };
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Func>
  struct CwiseArcCosH : CwiseFunctionOperator<CwiseArcCosH<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseArcCosH<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().acosh();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstVectorBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();

      typedef typename InType::Scalar Scalar;


      fx = x.array().acosh();
      jx.setOnes();
      jx *= Scalar(-1.0);
      jx += x.cwiseProduct(x);

      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        if constexpr (Base::ORC == 1)
          jx[0] = sqrt(jx[0]);
        else {
          const int sz = x.size();
          for (int i = 0; i < sz; i++)
            jx[i] = sqrt(jx[i]);
        }
      } else {
        jx = jx.cwiseSqrt();
      }

      jx = jx.cwiseInverse();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstVectorBaseRef<JacType> jx_,
                                               ConstVectorBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<HessType> hx = hx_.const_cast_derived();

      typedef typename InType::Scalar Scalar;


      fx = x.array().acosh();
      jx.setOnes();
      jx *= Scalar(-1.0);
      jx += x.cwiseProduct(x);

      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        if constexpr (Base::ORC == 1)
          jx[0] = sqrt(jx[0]);
        else {
          const int sz = x.size();
          for (int i = 0; i < sz; i++)
            jx[i] = sqrt(jx[i]);
        }
      } else {
        jx = jx.cwiseSqrt();
      }

      jx = jx.cwiseInverse();
      hx = Scalar(-1.0) * jx.array().cube() * x.array();
    }
  };
  /////////////////////////////////////////////////////////////////////////////////////////////////////
  template<class Func>
  struct CwiseArcTanH : CwiseFunctionOperator<CwiseArcTanH<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseArcTanH<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().atanh();
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstVectorBaseRef<JacType> jx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();

      typedef typename InType::Scalar Scalar;


      fx = x.array().atanh();
      jx.setOnes();
      jx -= x.cwiseProduct(x);
      jx = jx.cwiseInverse();
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstVectorBaseRef<JacType> jx_,
                                               ConstVectorBaseRef<HessType> hx_) {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<HessType> hx = hx_.const_cast_derived();

      typedef typename InType::Scalar Scalar;


      fx = x.array().atanh();
      jx.setOnes();
      jx -= x.cwiseProduct(x);
      jx = jx.cwiseInverse();

      hx = Scalar(2.0) * jx.array().square() * x.array();
    }
  };
  /////////////////////////////////////////////////////////////////////////////////////////////////////


  template<class Func>
  struct CwiseSqrt : CwiseFunctionOperator<CwiseSqrt<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseSqrt<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);

    static const bool IsVectorizable = true;

    template<class InType, class OutType>
    static void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        if constexpr (Base::ORC == 1)
          fx[0] = sqrt(x[0]);
        else {
          const int sz = x.size();
          for (int i = 0; i < sz; i++)
            fx[i] = sqrt(x[i]);
        }
      } else {
        fx = x.array().sqrt();
      }
    }
    template<class InType, class OutType, class JacType>
    static void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                       ConstVectorBaseRef<OutType> fx_,
                                       ConstEigenBaseRef<JacType> jx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        if constexpr (Base::ORC == 1)
          fx[0] = sqrt(x[0]);
        else {
          const int sz = x.size();
          for (int i = 0; i < sz; i++)
            fx[i] = sqrt(x[i]);
        }
      } else {
        fx = x.array().sqrt();
      }
      jx.derived() = fx.array().inverse() * Scalar(0.5);
    }
    template<class InType, class OutType, class JacType, class HessType>
    static void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                               ConstVectorBaseRef<OutType> fx_,
                                               ConstEigenBaseRef<JacType> jx_,
                                               ConstEigenBaseRef<HessType> hx_) {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();

      if constexpr (std::is_same<Scalar, DefaultSuperScalar>::value) {
        if constexpr (Base::ORC == 1)
          fx[0] = sqrt(x[0]);
        else {
          const int sz = x.size();
          for (int i = 0; i < sz; i++)
            fx[i] = sqrt(x[i]);
        }
      } else {
        fx = x.array().sqrt();
      }

      jx.derived() = fx.array().inverse() * Scalar(0.5);
      hx.derived() = Scalar(-0.5) * jx.derived().array() / x.array();
    }
  };


  template<class Func>
  struct CwisePow : CwiseFunctionOperator<CwisePow<Func>, Func> {
    using Base = CwiseFunctionOperator<CwisePow<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);


    static const bool IsVectorizable = true;
    double power = 1;

    CwisePow(Func f, double power) : Base(f), power(power) {
    }
    CwisePow() {
    }
    template<class InType, class OutType>
    void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.array().pow(Scalar(this->power));
    }
    template<class InType, class OutType, class JacType>
    void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                ConstVectorBaseRef<OutType> fx_,
                                ConstEigenBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      fx = x.array().pow(Scalar(this->power));
      jx.derived() = Scalar(this->power) * x.array().pow(Scalar(this->power - 1.0));
    }
    template<class InType, class OutType, class JacType, class HessType>
    void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                        ConstVectorBaseRef<OutType> fx_,
                                        ConstEigenBaseRef<JacType> jx_,
                                        ConstEigenBaseRef<HessType> hx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      EigenBaseRef<JacType> jx = jx_.const_cast_derived();
      EigenBaseRef<HessType> hx = hx_.const_cast_derived();
      fx = x.array().pow(Scalar(this->power));
      jx.derived() = Scalar(this->power) * x.array().pow(Scalar(this->power - 1.0));
      hx.derived() = Scalar(this->power * (this->power - 1.0)) * x.array().pow(Scalar(this->power - 2.0));
      ;
    }
  };


  template<class Func>
  struct CwiseAbs : CwiseFunctionOperator<CwiseAbs<Func>, Func> {
    using Base = CwiseFunctionOperator<CwiseAbs<Func>, Func>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);


    template<class InType, class OutType>
    void cwise_compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      fx = x.cwiseAbs();
    }
    template<class InType, class OutType, class JacType>
    void cwise_compute_jacobian(ConstVectorBaseRef<InType> x,
                                ConstVectorBaseRef<OutType> fx_,
                                ConstEigenBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();

      fx = x.cwiseAbs();
      if constexpr (Is_SuperScalar<Scalar>::value) {
        for (int i = 0; i < x.size(); i++) {
          jx[i] = x[i].sign();
        }
      } else {
        jx = x.array().sign();
      }
    }
    template<class InType, class OutType, class JacType, class HessType>
    void cwise_compute_jacobian_hessian(ConstVectorBaseRef<InType> x,
                                        ConstVectorBaseRef<OutType> fx_,
                                        ConstEigenBaseRef<JacType> jx_,
                                        ConstEigenBaseRef<HessType> hx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      VectorBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<HessType> hx = hx_.const_cast_derived();

      fx = x.cwiseAbs();
      if constexpr (Is_SuperScalar<Scalar>::value) {
        for (int i = 0; i < x.size(); i++) {
          jx[i] = x[i].sign();
          // hx[i] = jx[i];
        }
      } else {
        jx = x.array().sign();
        // hx = jx;
      }
    }
  };


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /// <summary>
  ///  This template Bas class defines all operations that are common to CwiseOperators.
  ///  Provides compute and derivative methods for derived classes so long as they implement
  ///  cwise_compute,cwise_compute_jacobian, etc.
  /// </summary>
  /// <typeparam name="Derived"></typeparam>
  /// <typeparam name="Func"></typeparam>

  template<class Derived, class Func>
  struct CwiseFunctionOperator : VectorFunction<Derived, Func::IRC, Func::ORC> {
    using Base = VectorFunction<Derived, Func::IRC, Func::ORC>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);

    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    Func func;
    static const bool IsVectorizable = Func::IsVectorizable;
    CwiseFunctionOperator() {
    }
    CwiseFunctionOperator(Func f) : func(std::move(f)) {
      this->setIORows(this->func.IRows(), this->func.ORows());
      this->set_input_domain(this->IRows(), {func.input_domain()});
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      Output<Scalar> fxt;

      if constexpr (Func::OutputIsDynamic) {
        fxt.resize(this->func.ORows());
      }

      this->func.compute(x, fxt);
      this->derived().cwise_compute(fxt, fx_);
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;

      Output<Scalar> fxt;
      Output<Scalar> jxdiag;

      if constexpr (Func::OutputIsDynamic) {
        fxt.resize(this->func.ORows());
        jxdiag.resize(this->func.ORows());
      }

      this->func.compute_jacobian(x, fxt, jx_);
      this->derived().cwise_compute_jacobian(fxt, fx_, jxdiag);
      if constexpr (Func::ORC == 1) {
        this->func.right_jacobian_product(jx_, jxdiag, jx_, DirectAssignment(), std::bool_constant<true>());
      } else {
        this->func.right_jacobian_product(
            jx_, jxdiag.asDiagonal(), jx_, DirectAssignment(), std::bool_constant<true>());
      }
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
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      // VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      Output<Scalar> fxt;
      Output<Scalar> jxdiag;
      Output<Scalar> hxdiag;

      if constexpr (Func::OutputIsDynamic) {
        fxt.resize(this->func.ORows());
        jxdiag.resize(this->func.ORows());
        hxdiag.resize(this->func.ORows());
      }

      this->func.compute(x, fxt);
      this->derived().cwise_compute_jacobian_hessian(fxt, fx_, jxdiag, hxdiag);

      fxt.setZero();
      Output<Scalar> adjtemp = jxdiag.cwiseProduct(adjvars);
      hxdiag = hxdiag.cwiseProduct(adjvars);
      this->func.compute_jacobian_adjointgradient_adjointhessian(x, fxt, jx_, adjgrad_, adjhess_, adjtemp);
      if constexpr (Func::ORC == 1) {

        this->func.symetric_jacobian_product(
            adjhess, hxdiag, jx, PlusEqualsAssignment(), std::bool_constant<false>());


        this->func.right_jacobian_product(jx_, jxdiag, jx, DirectAssignment(), std::bool_constant<true>());
      } else {
        this->func.symetric_jacobian_product(
            adjhess, hxdiag.asDiagonal(), jx, PlusEqualsAssignment(), std::bool_constant<false>());
        this->func.right_jacobian_product(
            jx_, jxdiag.asDiagonal(), jx, DirectAssignment(), std::bool_constant<true>());
      }
    }
  };

  template<class Derived, int IR>
  struct CwiseOperator : VectorFunction<Derived, IR, IR> {
    using Base = VectorFunction<Derived, IR, IR>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class InType, class OutType>
    inline void compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      this->derived().cwise_compute(x, fx_);
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian(ConstVectorBaseRef<InType> x,
                                 ConstVectorBaseRef<OutType> fx_,
                                 ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      this->derived().cwise_compute_jacobian(x, fx_, jx.diagonal());
    }
    template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian(
        ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_,
        ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_,
        ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();
      this->derived().cwise_compute_jacobian_hessian(x, fx_, jx.diagonal(), adjhess.diagonal());
      adjgrad = jx.diagonal().cwiseProduct(adjvars);
      adjhess.diagonal() = adjhess.diagonal().cwiseProduct(adjvars);
    }
  };

}  // namespace ASSET

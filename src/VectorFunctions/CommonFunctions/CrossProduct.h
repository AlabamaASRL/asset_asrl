#pragma once

#include "VectorFunction.h"

namespace ASSET {

  struct CrossProduct : VectorFunction<CrossProduct, 6, 3> {
    using Base = VectorFunction<CrossProduct, 6, 3>;
    using Base::compute;
    using Base::jacobian;

    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class Source, class Target, class Scalar2>
    static void cprodmat(const Eigen::MatrixBase<Source>& x,
                         Eigen::MatrixBase<Target> const& m_,
                         Scalar2 sign) {
      typedef typename Source::Scalar Scalar;
      Eigen::MatrixBase<Target>& m = m_.const_cast_derived();
      m(1, 0) += sign * x[2];
      m(2, 0) += -sign * x[1];
      m(2, 1) += sign * x[0];

      m(0, 1) += -sign * x[2];
      m(0, 2) += sign * x[1];
      m(1, 2) += -sign * x[0];
    }

    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      fx = x.template head<3>().cross(x.template tail<3>());
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      fx = x.template head<3>().cross(x.template tail<3>());
      cprodmat(x.template tail<3>(), jx.template leftCols<3>(), -1.0);
      cprodmat(x.template head<3>(), jx.template rightCols<3>(), 1.0);
    }

    template<class InType, class JacType>
    inline void jacobian(const Eigen::MatrixBase<InType>& x, Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      cprodmat(x.template tail<3>(), jx.template leftCols<3>(), -1.0);
      cprodmat(x.template head<3>(), jx.template rightCols<3>(), 1.0);
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

      fx = x.template head<3>().cross(x.template tail<3>());
      cprodmat(x.template tail<3>(), jx.template leftCols<3>(), -1.0);
      cprodmat(x.template head<3>(), jx.template rightCols<3>(), 1.0);

      adjgrad = (adjvars.transpose() * jx).transpose();

      cprodmat(adjvars, adjhess.template topRightCorner<3, 3>(), -1.0);
      cprodmat(adjvars, adjhess.template bottomLeftCorner<3, 3>(), 1.0);
    }
  };

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  template<class Derived, class Func1, class Func2>
  struct FunctionCrossProduct_Impl;

  template<class Func1, class Func2>
  struct FunctionCrossProduct : FunctionCrossProduct_Impl<FunctionCrossProduct<Func1, Func2>, Func1, Func2> {
    using Base = FunctionCrossProduct_Impl<FunctionCrossProduct<Func1, Func2>, Func1, Func2>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
  };

  template<class Derived, class Func1, class Func2>
  struct FunctionCrossProduct_Impl : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, 3> {
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, 3>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);

    Func1 func1;
    Func2 func2;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;

#if defined(_WIN32)
    static const bool IsVectorizable = Func1::IsVectorizable && Func2::IsVectorizable;
#else
    static const bool IsVectorizable = false;
#endif

    FunctionCrossProduct_Impl() {
    }
    FunctionCrossProduct_Impl(Func1 f1, Func2 f2) : func1(f1), func2(f2) {
      int irtemp = std::max(this->func1.IRows(), this->func2.IRows());
      this->setIORows(irtemp, 3);

      this->set_input_domain(this->IRows(), {this->func1.input_domain(), this->func2.input_domain()});

      if (this->func1.ORows() != 3) {
        throw std::invalid_argument("Function 1 in cross product must have three output rows");
      }
      if (this->func2.ORows() != 3) {
        throw std::invalid_argument("Function 2 in cross product must have three output rows");
      }
      if (this->func1.IRows() != this->func1.IRows()) {
        throw std::invalid_argument("Functions 1,2 in cross product must have same numer of input rows");
      }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class Scalar, class T1, class T2>
    Vector3<Scalar> crossimpl(Scalar sign, ConstVectorBaseRef<T1> x1, ConstVectorBaseRef<T2> x2) const {
      Vector3<Scalar> out;
      out[0] = sign * (x1[1] * x2[2] - x1[2] * x2[1]);
      out[1] = sign * (x2[0] * x1[2] - x2[2] * x1[0]);
      out[2] = sign * (x1[0] * x2[1] - x1[1] * x2[0]);
      return out;
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      Vector3<Scalar> fx1;
      fx1.setZero();
      Vector3<Scalar> fx2;
      fx2.setZero();

      this->func1.compute(x, fx1);
      this->func2.compute(x, fx2);
      // fx = (fx1.cross(fx2)).eval();
      fx = crossimpl(Scalar(1.0), fx1, fx2);
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      Vector3<Scalar> fx1;
      Vector3<Scalar> fx2;
      Eigen::Matrix<Scalar, 3, 3> cpm1;
      Eigen::Matrix<Scalar, 3, 3> cpm2;

      auto Impl = [&](auto& jx1, auto& jx2) {
        this->func1.compute_jacobian(x, fx1, jx1);
        this->func2.compute_jacobian(x, fx2, jx2);
        CrossProduct::cprodmat(fx2, cpm1, -1.0);
        CrossProduct::cprodmat(fx1, cpm2, 1.0);
        fx = crossimpl(Scalar(1.0), fx1, fx2);
        this->func1.right_jacobian_product(jx, cpm1, jx1, DirectAssignment(), std::bool_constant<false>());
        this->func2.right_jacobian_product(
            jx, cpm2, jx2, PlusEqualsAssignment(), std::bool_constant<false>());
      };


#if defined(ASSET_MEMORYMAN)
      using JType = Eigen::Matrix<Scalar, 3, Base::IRC>;
      const int irows = this->IRows();
      MemoryManager::allocate_run(irows, Impl, TempSpec<JType>(3, irows), TempSpec<JType>(3, irows));
#else
      if constexpr (Base::InputIsDynamic) {
        const int irows = this->IRows();
        auto DynImpl = [&](auto maxsize) {
          Eigen::Matrix<Scalar, 3, -1, 0, 3, maxsize.value> jx1(3, this->IRows());
          Eigen::Matrix<Scalar, 3, -1, 0, 3, maxsize.value> jx2(3, this->IRows());
          jx1.setZero();
          jx2.setZero();

          Impl(jx1, jx2);
        };
        LambdaJumpTable<6, 8, 16>::run(DynImpl, irows);
      } else {
        Eigen::Matrix<Scalar, 3, Base::IRC> jx1(3, this->IRows());
        Eigen::Matrix<Scalar, 3, Base::IRC> jx2(3, this->IRows());
        jx1.setZero();
        jx2.setZero();
        Impl(jx1, jx2);
      }
#endif
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

      Vector3<Scalar> fx1;
      Vector3<Scalar> fx2;
      Eigen::Matrix<Scalar, 3, 3> cpm1;
      Eigen::Matrix<Scalar, 3, 3> cpm2;
      Eigen::Matrix<Scalar, 3, 3> lcpm1;
      Eigen::Matrix<Scalar, 3, 3> lcpm2;


      auto Impl = [&](auto& jx1, auto& jx2, auto& jttemp, auto& gx2, auto& hx2) {
        this->func1.compute(x, fx1);
        this->func2.compute(x, fx2);

        CrossProduct::cprodmat(fx2, cpm1, -1.0);
        CrossProduct::cprodmat(fx1, cpm2, 1.0);

        CrossProduct::cprodmat(adjvars, lcpm1, 1.0);
        CrossProduct::cprodmat(adjvars, lcpm2, -1.0);

        Vector3<Scalar> adjt = adjvars;

        fx = crossimpl(Scalar(1.0), fx1, fx2);
        Vector3<Scalar> adjcross1 = crossimpl(Scalar(1.0), fx2, adjt);
        Vector3<Scalar> adjcross2 = crossimpl(Scalar(-1.0), fx1, adjt);

        fx1.setZero();
        fx2.setZero();

        this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx1, jx1, adjgrad, adjhess, adjcross1);
        this->func2.compute_jacobian_adjointgradient_adjointhessian(x, fx2, jx2, gx2, hx2, adjcross2);

        this->func2.accumulate_gradient(adjgrad, gx2, PlusEqualsAssignment());
        this->func2.accumulate_hessian(adjhess, hx2, PlusEqualsAssignment());

        this->func2.zero_matrix_domain(hx2);

        this->func2.right_jacobian_product(
            jttemp, lcpm2, jx2, DirectAssignment(), std::bool_constant<false>());
        this->func1.right_jacobian_product(
            hx2, jttemp.transpose(), jx1, DirectAssignment(), std::bool_constant<false>());

        // adjhess += hx2 + hx2^T
        // func1 does this because hx2 now has its domain structure
        this->func1.accumulate_product_hessian(adjhess, hx2);

        this->func1.right_jacobian_product(jx, cpm1, jx1, DirectAssignment(), std::bool_constant<false>());
        this->func2.right_jacobian_product(
            jx, cpm2, jx2, PlusEqualsAssignment(), std::bool_constant<false>());
      };

#if defined(ASSET_MEMORYMAN)
      using JType = Eigen::Matrix<Scalar, 3, Base::IRC>;
      using GType = Func2_gradient<Scalar>;
      using HType = Func2_hessian<Scalar>;
      const int irows = this->IRows();

      MemoryManager::allocate_run(irows,
                                  Impl,
                                  TempSpec<JType>(3, irows),
                                  TempSpec<JType>(3, irows),
                                  TempSpec<JType>(3, irows),
                                  TempSpec<GType>(irows, 1),
                                  TempSpec<HType>(irows, irows));
#else
      if constexpr (Base::InputIsDynamic) {
        const int irows = this->IRows();
        auto DynImpl = [&](auto maxsize) {
          MaxVector<Scalar, maxsize.value> gx2;
          MaxMatrix<Scalar, maxsize.value> hx2;
          Eigen::Matrix<Scalar, 3, -1, 0, 3, maxsize.value> jx1;
          Eigen::Matrix<Scalar, 3, -1, 0, 3, maxsize.value> jx2;
          Eigen::Matrix<Scalar, 3, -1, 0, 3, maxsize.value> jtemp;
          gx2.resize(this->IRows());
          hx2.resize(this->IRows(), this->IRows());
          jx1.resize(3, this->IRows());
          jx2.resize(3, this->IRows());
          jtemp.resize(3, this->IRows());
          Impl(jx1, jx2, jtemp, gx2, hx2);
        };
        LambdaJumpTable<6, 8, 16>::run(DynImpl, irows);
      } else {

        Func2_gradient<Scalar> gx2(this->func2.IRows());
        gx2.setZero();
        Func2_hessian<Scalar> hx2(this->func2.IRows(), this->func2.IRows());
        hx2.setZero();

        Eigen::Matrix<Scalar, 3, Base::IRC> jtemp(3, this->IRows());
        Eigen::Matrix<Scalar, 3, Base::IRC> jx1(3, this->IRows());
        Eigen::Matrix<Scalar, 3, Base::IRC> jx2(3, this->IRows());
        jx1.setZero();
        jx2.setZero();

        jtemp.setZero();

        Impl(jx1, jx2, jtemp, gx2, hx2);
      }
#endif
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };

}  // namespace ASSET

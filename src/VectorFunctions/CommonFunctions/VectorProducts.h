#pragma once
#include "VectorFunction.h"

namespace ASSET {


  template<class Derived, class Func1, class Func2, int Vsize>
  struct FunctionVectorProduct_Impl;


  template<class Func1, class Func2>
  struct FunctionImagProduct
      : FunctionVectorProduct_Impl<FunctionImagProduct<Func1, Func2>, Func1, Func2, 2> {
    using Base = FunctionVectorProduct_Impl<FunctionImagProduct<Func1, Func2>, Func1, Func2, 2>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
  };

  template<class Func1, class Func2>
  struct FunctionCrossProduct
      : FunctionVectorProduct_Impl<FunctionCrossProduct<Func1, Func2>, Func1, Func2, 3> {
    using Base = FunctionVectorProduct_Impl<FunctionCrossProduct<Func1, Func2>, Func1, Func2, 3>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
  };

  template<class Func1, class Func2>
  struct FunctionQuatProduct
      : FunctionVectorProduct_Impl<FunctionQuatProduct<Func1, Func2>, Func1, Func2, 4> {
    using Base = FunctionVectorProduct_Impl<FunctionQuatProduct<Func1, Func2>, Func1, Func2, 4>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
  };


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


  /// <summary>
  /// ///////////////
  /// </summary>
  /// <typeparam name="Derived"></typeparam>
  /// <typeparam name="Func1"></typeparam>
  /// <typeparam name="Func2"></typeparam>
  template<class Derived, class Func1, class Func2, int Vsize>
  struct FunctionVectorProduct_Impl : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, Vsize> {
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, Vsize>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);

    Func1 func1;
    Func2 func2;


    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;

    static const bool IsSegmentOp = Is_Segment<Func1>::value && Is_Segment<Func2>::value;

    static const bool IsVectorizable = Func1::IsVectorizable && Func2::IsVectorizable;


    FunctionVectorProduct_Impl() {
    }
    FunctionVectorProduct_Impl(Func1 f1, Func2 f2) : func1(std::move(f1)), func2(std::move(f2)) {
      int irtemp = std::max(this->func1.IRows(), this->func2.IRows());
      this->setIORows(irtemp, Vsize);

      this->set_input_domain(this->IRows(), {this->func1.input_domain(), this->func2.input_domain()});

      if (this->func1.ORows() != Vsize) {

        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in FunctionVectorProduct (VectorSize = {1:}) !!!\n"
                   "Output Size of Func1 (ORows = {0:})  must equal {1:}.\n",
                   this->func1.ORows(),
                   Vsize);
        throw std::invalid_argument("");
      }
      if (this->func2.ORows() != Vsize) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in FunctionVectorProduct (VectorSize = {1:}) !!!\n"
                   "Output Size of Func2 (ORows = {0:})  must equal {1:}.\n",
                   this->func2.ORows(),
                   Vsize);
        throw std::invalid_argument("");
      }
      if (this->func1.IRows() != this->func2.IRows()) {

        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in FunctionVectorProduct (VectorSize = {2:}) !!!\n"
                   "Input Size of Func1 (IRows = {0:}) does not match Input Size of Func2 (IRows = {1:}).\n",
                   this->func1.IRows(),
                   this->func2.IRows(),
                   Vsize);
        throw std::invalid_argument("");

        // throw std::invalid_argument("Functions 1,2 in vector product must have same numer of input rows");
      }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    template<class Scalar, class T1, class T2>
    Vector2<Scalar> imagprodimpl(Scalar sign, ConstVectorBaseRef<T1> x1, ConstVectorBaseRef<T2> x2) const {
      Vector2<Scalar> out;
      out[0] = sign * (x1[0] * x2[0] - x1[1] * x2[1]);
      out[1] = sign * (x1[0] * x2[1] + x1[1] * x2[0]);
      return out;
    }

    template<class Source, class Target, class Scalar2>
    inline void fillimagprodmatrix(const Eigen::MatrixBase<Source>& x,
                                   Eigen::MatrixBase<Target> const& m_,
                                   Scalar2 sign) const {
      typedef typename Source::Scalar Scalar;
      Eigen::MatrixBase<Target>& m = m_.const_cast_derived();
      m(0, 0) += sign * x[0];
      m(1, 0) += sign * x[1];
      m(0, 1) += -sign * x[1];
      m(1, 1) += sign * x[0];
    }

    template<class Scalar, class T1, class T2>
    Vector3<Scalar> crossprodimpl(Scalar sign, ConstVectorBaseRef<T1> x1, ConstVectorBaseRef<T2> x2) const {
      Vector3<Scalar> out;
      out[0] = sign * (x1[1] * x2[2] - x1[2] * x2[1]);
      out[1] = sign * (x2[0] * x1[2] - x2[2] * x1[0]);
      out[2] = sign * (x1[0] * x2[1] - x1[1] * x2[0]);
      return out;
    }
    template<class Source, class Target, class Scalar2>
    inline void fillcrossprodmatrix(const Eigen::MatrixBase<Source>& x,
                                    Eigen::MatrixBase<Target> const& m_,
                                    Scalar2 sign) const {
      typedef typename Source::Scalar Scalar;
      Eigen::MatrixBase<Target>& m = m_.const_cast_derived();
      m(1, 0) += sign * x[2];
      m(2, 0) += -sign * x[1];

      m(0, 1) += -sign * x[2];
      m(2, 1) += sign * x[0];

      m(0, 2) += sign * x[1];
      m(1, 2) += -sign * x[0];
    }

    template<class Scalar, class T1, class T2>
    Vector4<Scalar> quatprodimpl(Scalar sign, ConstVectorBaseRef<T1> x1, ConstVectorBaseRef<T2> x2) const {
      Vector4<Scalar> out;
      out[0] = sign * (x2[3] * x1[0] + x1[3] * x2[0] + x1[1] * x2[2] - x1[2] * x2[1]);
      out[1] = sign * (x2[3] * x1[1] + x1[3] * x2[1] + x2[0] * x1[2] - x2[2] * x1[0]);
      out[2] = sign * (x2[3] * x1[2] + x1[3] * x2[2] + x1[0] * x2[1] - x1[1] * x2[0]);
      out[3] = sign * (x2[3] * x1[3] - x1[0] * x2[0] - x1[1] * x2[1] - x1[2] * x2[2]);
      return out;
    }
    template<class Source, class Target, class Scalar2>
    inline void fillquatprodmatrix(const Eigen::MatrixBase<Source>& x,
                                   Eigen::MatrixBase<Target> const& m_,
                                   Scalar2 sign) const {
      typedef typename Source::Scalar Scalar;
      Eigen::MatrixBase<Target>& m = m_.const_cast_derived();

      m(0, 0) += x[3];
      m(1, 0) += sign * x[2];
      m(2, 0) += -sign * x[1];
      m(3, 0) += -x[0];

      m(0, 1) += -sign * x[2];
      m(1, 1) += x[3];
      m(2, 1) += sign * x[0];
      m(3, 1) += -x[1];


      m(0, 2) += sign * x[1];
      m(1, 2) += -sign * x[0];
      m(2, 2) += x[3];
      m(3, 2) += -x[2];

      m(0, 3) += x[0];
      m(1, 3) += x[1];
      m(2, 3) += x[2];
      m(3, 3) += x[3];
    }


    template<class Scalar, class T1, class T2>
    Output<Scalar> vecprodimpl(Scalar sign, ConstVectorBaseRef<T1> x1, ConstVectorBaseRef<T2> x2) const {
      if constexpr (Vsize == 2) {
        return this->imagprodimpl(sign, x1, x2);
      } else if constexpr (Vsize == 3) {
        return this->crossprodimpl(sign, x1, x2);
      } else if constexpr (Vsize == 4) {
        return this->quatprodimpl(sign, x1, x2);
      } else {
      }
    }


    template<class Source, class Target, class Scalar2>
    inline void fillprodmatrix(const Eigen::MatrixBase<Source>& x,
                               Eigen::MatrixBase<Target> const& m_,
                               Scalar2 sign) const {

      if constexpr (Vsize == 2) {
        return this->fillimagprodmatrix(x, m_, sign);
      } else if constexpr (Vsize == 3) {
        return this->fillcrossprodmatrix(x, m_, sign);
      } else if constexpr (Vsize == 4) {
        return this->fillquatprodmatrix(x, m_, sign);
      } else {
      }
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      Output<Scalar> fx1;
      Output<Scalar> fx2;
      this->func1.compute(x, fx1);
      this->func2.compute(x, fx2);
      fx = this->vecprodimpl(Scalar(1.0), fx1, fx2);
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      Output<Scalar> fx1;
      Output<Scalar> fx2;
      Scalar fsign1 = Vsize == 2 ? Scalar(1.0) : Scalar(-1.0);
      Scalar fsign2 = Vsize == 2 ? Scalar(1.0) : Scalar(1.0);

      if constexpr (IsSegmentOp) {
        this->func1.compute(x, fx1);
        this->func2.compute(x, fx2);
        fx = this->vecprodimpl(Scalar(1.0), fx1, fx2);
        this->fillprodmatrix(fx2, jx.template middleCols<Vsize>(this->func1.SegStart, Vsize), fsign1);
        this->fillprodmatrix(fx1, jx.template middleCols<Vsize>(this->func2.SegStart, Vsize), fsign2);
      } else {

        Eigen::Matrix<Scalar, Vsize, Vsize> pm1;
        Eigen::Matrix<Scalar, Vsize, Vsize> pm2;

        auto Impl = [&](auto& jx1, auto& jx2) {
          this->func1.compute_jacobian(x, fx1, jx1);
          this->func2.compute_jacobian(x, fx2, jx2);

          this->fillprodmatrix(fx2, pm1, fsign1);
          this->fillprodmatrix(fx1, pm2, fsign2);
          fx = this->vecprodimpl(Scalar(1.0), fx1, fx2);

          this->func1.right_jacobian_product(jx, pm1, jx1, DirectAssignment(), std::bool_constant<false>());
          this->func2.right_jacobian_product(
              jx, pm2, jx2, PlusEqualsAssignment(), std::bool_constant<false>());
        };


        using JType = Eigen::Matrix<Scalar, Vsize, Base::IRC>;
        const int irows = this->IRows();
        MemoryManager::allocate_run(
            irows, Impl, TempSpec<JType>(Vsize, irows), TempSpec<JType>(Vsize, irows));
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
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      Output<Scalar> fx1;
      Output<Scalar> fx2;

      Output<Scalar> adjt = adjvars;
      Scalar fsign1 = Vsize == 2 ? Scalar(1.0) : Scalar(-1.0);
      Scalar fsign2 = Vsize == 2 ? Scalar(1.0) : Scalar(1.0);

      Eigen::Matrix<Scalar, Vsize, Vsize> pm1;
      Eigen::Matrix<Scalar, Vsize, Vsize> pm2;
      Eigen::Matrix<Scalar, Vsize, Vsize> lpm1;
      Eigen::Matrix<Scalar, Vsize, Vsize> lpm2;

      auto Impl = [&](auto& jx1, auto& jx2, auto& jttemp, auto& gx2, auto& hx2) {
        this->func1.compute(x, fx1);
        this->func2.compute(x, fx2);

        this->fillprodmatrix(fx2, pm1, fsign1);
        this->fillprodmatrix(fx1, pm2, fsign2);


        this->fillprodmatrix(adjt, lpm2, fsign1);
        this->fillprodmatrix(adjt, lpm1, fsign2);

        if constexpr (Vsize == 4) {
          lpm2.diagonal().template head<3>() *= Scalar(-1.0);
          lpm2.row(3).template head<3>() *= Scalar(-1.0);
        }

        fx = this->vecprodimpl(Scalar(1.0), fx1, fx2);

        Output<Scalar> adjcross1 = pm1.transpose() * adjt;
        Output<Scalar> adjcross2 = pm2.transpose() * adjt;

        fx1.setZero();
        fx2.setZero();

        this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx1, jx1, adjgrad, adjhess, adjcross1);
        this->func2.compute_jacobian_adjointgradient_adjointhessian(x, fx2, jx2, gx2, hx2, adjcross2);

        this->func2.accumulate_gradient(adjgrad, gx2, PlusEqualsAssignment());
        this->func2.accumulate_hessian(adjhess, hx2, PlusEqualsAssignment());

        this->func2.zero_matrix_domain(hx2);

        this->func2.right_jacobian_product(
            jttemp, lpm2, jx2, DirectAssignment(), std::bool_constant<false>());
        this->func1.right_jacobian_product(
            hx2, jttemp.transpose(), jx1, DirectAssignment(), std::bool_constant<false>());

        // adjhess += hx2 + hx2^T
        // func1 does this because hx2 now has its domain structure
        this->func1.accumulate_product_hessian(adjhess, hx2);

        this->func1.right_jacobian_product(jx, pm1, jx1, DirectAssignment(), std::bool_constant<false>());
        this->func2.right_jacobian_product(jx, pm2, jx2, PlusEqualsAssignment(), std::bool_constant<false>());
      };


      using JType = Eigen::Matrix<Scalar, Vsize, Base::IRC>;
      using GType = Func2_gradient<Scalar>;
      using HType = Func2_hessian<Scalar>;
      const int irows = this->IRows();

      MemoryManager::allocate_run(irows,
                                  Impl,
                                  TempSpec<JType>(Vsize, irows),
                                  TempSpec<JType>(Vsize, irows),
                                  TempSpec<JType>(Vsize, irows),
                                  TempSpec<GType>(irows, 1),
                                  TempSpec<HType>(irows, irows));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };


}  // namespace ASSET

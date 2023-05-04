/*
File Name: DenseFunctionBase.h

File Description: Implements the Base class for all dense vector functions in asset.
Forwards the Derived class and compile time input(IR) and output(OR) rows down the CRTP inheretance chain.
Also inherits from from domain holder so dynamic sized vector functions can hold an array containing their
true input domain that is computed at run-time. Also defines the default compile time INPUT_DOMAIN of both
constant and dynamic size functions. Composite derived classes will overide this typedef by design, in order
to explicitly participate in the input domain tracking system. The default input domain simply says that all
inputs are used. This input domain info is used to implement the default set of functions for manipluting the
jacobian,hessian,and gradient of Derived function such as right_jacobian_product. Specialized derived classes
should perform simple overloads of these methods.

This class also defines most of the vector function indexing (.segment) and unary(.normalize)
and binary(.dot) mathematical operations that are used when writing expressions.

Additionally, This class also defines the .compute_jacobian etc. methods in terms of the compute_jacobian_impl
methods implemented in derived classes. Theses methods are the used to implement the dense vector functions
interface with psiopt throught the constraints_jacobian etc. methods.

Finally, this class also contains methods (Postfixed Build) defining the python binding code of a vector
functions public interface.

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
#include <bench/BenchTimer.h>

#include "AssigmentTypes.h"
#include "BinaryMath.h"
#include "CommonFunctions/ExpressionFwdDeclarations.h"
#include "Computable.h"
#include "DenseFunctionOperations.h"
#include "FunctionDomains.h"
#include "FunctionTypeDefMacros.h"
#include "PyDocString/VectorFunctions/DenseFunctionBase_doc.h"


namespace ASSET {


  template<class Derived, int IR, int OR>
  struct DenseFunctionBase : Computable<Derived, IR, OR>, DomainHolder<IR> {
    using Base = Computable<Derived, IR, OR>;

    /////////////////////////////////////////////////////////////
    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;
    template<class Scalar>
    using Gradient = typename Base::template Gradient<Scalar>;
    template<class Scalar>
    using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
    template<class Scalar>
    using Hessian = Eigen::Matrix<Scalar, IR, IR>;
    /////////////////////////////////////////////////////////////
    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using VectorBaseRef = Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using ConstMatrixBaseRef = const Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using MatrixBaseRef = Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using ConstEigenBaseRef = const Eigen::EigenBase<Scalar>&;
    template<class Scalar>
    using EigenBaseRef = Eigen::EigenBase<Scalar>&;
    /////////////////////////////////////////////////////////////

    using INPUT_DOMAIN = SingleDomain<IR, 0, IR>;

    template<class NewDerived>
    using AsBaseClass = FunctionHolder<NewDerived, Derived, IR, OR>;

    using NAME = Derived;

    template<class Func>
    using EVALOP = typename ASSET::template NestedFunctionSelector<Derived, Func>;
    template<class Func>
    using FWDOP = typename ASSET::template NestedFunctionSelector<Func, Derived>;
    template<int SZ, int ST>
    using SEGMENTOP = typename ASSET::template NestedFunctionSelector<Segment<OR, SZ, ST>, Derived>;


    void setIORows(int inputrows, int outputrows) {
      Base::setIORows(inputrows, outputrows);
      if constexpr (IR < 0) {
        DomainMatrix dmn(2, 1);
        dmn(0, 0) = 0;
        dmn(1, 0) = inputrows;
        this->set_input_domain(inputrows, {dmn});
      }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class Func, int FuncIRC>
    auto operator()(const DenseFunctionBase<Func, FuncIRC, IR>& f) const {
      return EVALOP<Func>::make_nested(this->derived(), f.derived());
    }
    template<class Func, int FuncIRC, int FuncORC>
    auto eval(const DenseFunctionBase<Func, FuncIRC, FuncORC>& f) const {
      return EVALOP<Func>::make_nested(this->derived(), f.derived());
    }

    template<int SZ, int ST>
    auto segment() const {
      return SEGMENTOP<SZ, ST>::make_nested(Segment<OR, SZ, ST>(this->ORows(), SZ, ST), this->derived());
    }

    template<int SZ, int ST>
    auto segment(int start, int size) const {
      return SEGMENTOP<SZ, ST>::make_nested(Segment<OR, SZ, ST>(this->ORows(), size, start), this->derived());
    }

    auto segment(int start, int size) const {
      return SEGMENTOP<-1, -1>::make_nested(Segment<OR, -1, -1>(this->ORows(), size, start), this->derived());
    }

    template<int SZ>
    auto segment(int start) const {
      return SEGMENTOP<SZ, -1>::make_nested(Segment<OR, SZ, -1>(this->ORows(), SZ, start), this->derived());
    }
    template<int SZ>
    auto head() const {
      return SEGMENTOP<SZ, 0>::make_nested(Segment<OR, SZ, 0>(this->ORows(), SZ, 0), this->derived());
    }
    template<int SZ>
    auto head(int sz) const {
      return SEGMENTOP<SZ, 0>::make_nested(Segment<OR, SZ, 0>(this->ORows(), sz, 0), this->derived());
    }

    auto head(int sz) const {
      return SEGMENTOP<-1, -1>::make_nested(Segment<OR, -1, -1>(this->ORows(), sz, 0), this->derived());
    }

    template<int SZ>
    auto tail() const {
      return SEGMENTOP<SZ, SZ_DIFF<OR, SZ>::value>::make_nested(
          Segment<OR, SZ, SZ_DIFF<OR, SZ>::value>(this->ORows(), SZ, this->ORows() - SZ), this->derived());
    }
    template<int SZ>
    decltype(auto) tail(int sz) const {
      return SEGMENTOP<SZ, SZ_DIFF<OR, SZ>::value>::make_nested(
          Segment<OR, SZ, SZ_DIFF<OR, SZ>::value>(this->ORows(), sz, this->ORows() - sz), this->derived());
    }

    decltype(auto) tail(int sz) const {
      return SEGMENTOP<-1, -1>::make_nested(Segment<OR, -1, -1>(this->ORows(), sz, this->ORows() - sz),
                                            this->derived());
    }


    template<int ELE>
    decltype(auto) coeff() const {
      return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->ORows(), 1, ELE), this->derived());
    }
    template<int ELE>
    decltype(auto) coeff(int ele) const {
      return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->ORows(), 1, ele), this->derived());
    }

    auto coeff(int ele) const {
      return SEGMENTOP<1, -1>::make_nested(Segment<OR, 1, -1>(this->ORows(), 1, ele), this->derived());
    }

    template<int ELE>
    decltype(auto) operator[](std::integral_constant<int, ELE> ele) const {
      return SEGMENTOP<1, ELE>::make_nested(Segment<OR, 1, ELE>(this->ORows(), 1, ELE), this->derived());
    }

    template<int EL1, int... ELS>
    decltype(auto) elements() const {
      return FWDOP<Elements<OR, EL1, ELS...>>::make_nested(Elements<OR, EL1, ELS...>(this->ORows()),
                                                           this->derived());
    }

    decltype(auto) normalized() const {
      return FWDOP<Normalized<OR>>::make_nested(Normalized<OR>(this->ORows()), this->derived());
    }
    template<int PW>
    decltype(auto) normalized_power() const {
      return FWDOP<NormalizedPower<OR, PW>>::make_nested(NormalizedPower<OR, PW>(this->ORows()),
                                                         this->derived());
    }

    decltype(auto) norm() const {
      return FWDOP<Norm<OR>>::make_nested(Norm<OR>(this->ORows()), this->derived());
    }
    decltype(auto) squared_norm() const {
      return FWDOP<SquaredNorm<OR>>::make_nested(SquaredNorm<OR>(this->ORows()), this->derived());
    }
    decltype(auto) inverse_norm() const {
      return FWDOP<InverseNorm<OR>>::make_nested(InverseNorm<OR>(this->ORows()), this->derived());
    }
    decltype(auto) inverse_squared_norm() const {
      return FWDOP<InverseSquaredNorm<OR>>::make_nested(InverseSquaredNorm<OR>(this->ORows()),
                                                        this->derived());
    }
    template<int PW>
    decltype(auto) norm_power() const {
      return FWDOP<NormPower<OR, PW>>::make_nested(NormPower<OR, PW>(this->ORows()), this->derived());
    }
    template<int PW>
    decltype(auto) inverse_norm_power() const {
      return FWDOP<InverseNormPower<OR, PW>>::make_nested(InverseNormPower<OR, PW>(this->ORows()),
                                                          this->derived());
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<int MRows, int MCols, int MMajor = Eigen::ColMajor>
    auto matrix(int rows, int cols) const {
      return MatrixFunctionView<Derived, MRows, MCols, MMajor>(this->derived(), rows, cols);
    }

    auto colmatrix(int rows, int cols) const {
      return MatrixFunctionView<Derived, -1, -1, Eigen::ColMajor>(this->derived(), rows, cols);
    }
    auto rowmatrix(int rows, int cols) const {
      return MatrixFunctionView<Derived, -1, -1, Eigen::RowMajor>(this->derived(), rows, cols);
    }

    template<int MRows, int MCols, int MMajor = Eigen::ColMajor>
    auto matrix() const {
      return MatrixFunctionView<Derived, MRows, MCols, MMajor>(this->derived(), MRows, MCols);
    }
    auto colvector() const {
      return MatrixFunctionView<Derived, OR, 1, Eigen::ColMajor>(this->derived(), this->ORows(), 1);
    }
    auto rowvector() const {
      return MatrixFunctionView<Derived, 1, OR, Eigen::ColMajor>(this->derived(), 1, this->ORows());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto Sin() const {
      return CwiseSin<Derived>(this->derived());
    }
    auto Cos() const {
      return CwiseCos<Derived>(this->derived());
    }
    auto ArcSin() const {
      return CwiseArcSin<Derived>(this->derived());
    }
    auto ArcCos() const {
      return CwiseArcCos<Derived>(this->derived());
    }
    auto Tan() const {
      return CwiseTan<Derived>(this->derived());
    }
    auto Square() const {
      return CwiseSquare<Derived>(this->derived());
    }
    auto Sqrt() const {
      return CwiseSqrt<Derived>(this->derived());
    }
    auto Exp() const {
      return CwiseExp<Derived>(this->derived());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<int LP>
    auto padded_lower() const {
      return PaddedOutput<Derived, 0, LP>(this->derived(), 0, LP);
    }
    template<int UP>
    auto padded_upper() const {
      return PaddedOutput<Derived, UP, 0>(this->derived(), UP, 0);
    }
    template<int UP, int LP>
    auto padded() const {
      return PaddedOutput<Derived, UP, LP>(this->derived(), UP, LP);
    }


    auto padded_lower(int LP) const {
      return PaddedOutput<Derived, 0, -1>(this->derived(), 0, LP);
    }
    auto padded_upper(int UP) const {
      return PaddedOutput<Derived, -1, 0>(this->derived(), UP, 0);
    }
    auto padded(int UP, int LP) const {
      return PaddedOutput<Derived, -1, -1>(this->derived(), UP, LP);
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class Func>
    decltype(auto) cwiseProduct(const DenseFunctionBase<Func, IR, OR>& f) const {
      return CwiseFunctionProduct<Derived, Func>(this->derived(), f.derived());
    }
    decltype(auto) cwiseInverse() const {
      return CwiseInverse<Derived>(this->derived());
    }
    decltype(auto) sum() const {
      if constexpr (OR == 1)
        return Derived(this->derived());
      else
        return CwiseSum<Derived>(this->derived());
    }

    template<class Func>
    decltype(auto) cross(const DenseFunctionBase<Func, IR, 3>& f) const {
      return FunctionCrossProduct<Derived, Func>(this->derived(), f.derived());
    }

    template<class Func>
    decltype(auto) dot(const DenseFunctionBase<Func, IR, OR>& f) const {
      return FunctionDotProduct<Derived, Func>(this->derived(), f.derived());
    }
    template<class ScalFunc>
    decltype(auto) scale(const DenseFunctionBase<ScalFunc, IR, 1>& f) const {
      return VectorScalarFunctionProduct<Derived, ScalFunc>(this->derived(), f.derived());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian(ConstVectorBaseRef<InType> x,
                                 ConstVectorBaseRef<OutType> fx_,
                                 ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      if constexpr (!Derived::IsVectorizable) {
        if constexpr (Is_SuperScalar<Scalar>::value) {
          VectorBaseRef<OutType> fx = fx_.const_cast_derived();
          VectorBaseRef<JacType> jx = jx_.const_cast_derived();

          typedef typename Scalar::Scalar RealScalar;

          Input<RealScalar> x_r;
          Output<RealScalar> fx_r;
          Jacobian<RealScalar> jx_r;

          const int IRR = this->IRows();
          const int ORR = this->ORows();

          if constexpr (Base::InputIsDynamic)
            x_r.resize(IRR);
          if constexpr (Base::OutputIsDynamic)
            fx_r.resize(ORR);
          if constexpr (Base::JacobianIsDynamic)
            jx_r.resize(ORR, IRR);

          for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
            for (int j = 0; j < IRR; j++)
              x_r[j] = x[j][i];
            this->derived().compute_jacobian_impl(x_r, fx_r, jx_r);
            for (int j = 0; j < ORR; j++)
              fx[j][i] = fx_r[j];

            for (int j = 0; j < IRR; j++)
              for (int k = 0; k < ORR; k++)
                jx(k, j)[i] = jx_r(k, j);

            fx_r.setZero();
            jx_r.setZero();
          }

        } else {
          this->derived().compute_jacobian_impl(x, fx_, jx_);
        }
      } else {
        this->derived().compute_jacobian_impl(x, fx_, jx_);
      }
    }

    template<class InType, class JacType>
    inline void jacobian(ConstVectorBaseRef<InType> x, ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      Output<Scalar> fx_(this->ORows());
      fx_.setZero();
      this->derived().compute_jacobian(x, fx_, jx_);
    }

    template<class InType>
    inline Jacobian<typename InType::Scalar> jacobian(ConstVectorBaseRef<InType> x) const {
      typedef typename InType::Scalar Scalar;
      Jacobian<Scalar> jx(this->ORows(), this->IRows());
      jx.setZero();
      this->derived().jacobian(x, jx);
      return jx;
    }

    template<class JacType, class AdjGradType, class AdjVarType>
    inline void jacobianXadjoint(ConstMatrixBaseRef<JacType> jx_,
                                 ConstVectorBaseRef<AdjGradType> adjgrad_,
                                 ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename JacType::Scalar JScalar;
      typedef typename AdjVarType::Scalar AScalar;
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();

      if constexpr (std::is_same<JScalar, AScalar>::value) {
        adjgrad.noalias() = (adjvars.transpose() * jx_).transpose();
      } else {
        for (int i = 0; i < this->IRows(); i++) {
          for (int j = 0; j < this->ORows(); j++) {
            adjgrad[i] += adjvars[j] * jx_(j, i);
          }
        }
      }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class InType, class OutType, class AdjGradType, class AdjVarType>
    inline void compute_adjointgradient(ConstVectorBaseRef<InType> x,
                                        ConstVectorBaseRef<OutType> fx_,
                                        ConstVectorBaseRef<AdjGradType> adjgrad_,
                                        ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      Jacobian<Scalar> jx(this->ORows(), this->IRows());
      jx.setZero();
      this->derived().compute_jacobian(x, fx_, jx);
      this->jacobianXadjoint(jx, adjgrad_, adjvars);
    }

    template<class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(ConstVectorBaseRef<InType> x,
                                                 ConstVectorBaseRef<OutType> fx_,
                                                 ConstMatrixBaseRef<JacType> jx_,
                                                 ConstVectorBaseRef<AdjGradType> adjgrad_,
                                                 ConstVectorBaseRef<AdjVarType> adjvars) const {
      this->derived().compute_jacobian(x, fx_, jx_);
      this->jacobianXadjoint(jx_, adjgrad_, adjvars);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
      if constexpr (!Derived::IsVectorizable) {

        if constexpr (Is_SuperScalar<Scalar>::value) {
          VectorBaseRef<OutType> fx = fx_.const_cast_derived();
          MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
          Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
          Eigen::MatrixBase<AdjHessType>& adjhess = adjhess_.const_cast_derived();

          typedef typename Scalar::Scalar RealScalar;

          Input<RealScalar> x_r;
          Output<RealScalar> fx_r;
          Jacobian<RealScalar> jx_r;
          Gradient<RealScalar> gx_r;
          Hessian<RealScalar> hx_r;
          Output<RealScalar> l_r;

          const int IRR = this->IRows();
          const int ORR = this->ORows();

          if constexpr (Base::InputIsDynamic)
            x_r.resize(IRR);
          if constexpr (Base::InputIsDynamic)
            gx_r.resize(IRR);
          if constexpr (Base::InputIsDynamic)
            hx_r.resize(IRR, IRR);

          if constexpr (Base::OutputIsDynamic)
            fx_r.resize(ORR);
          if constexpr (Base::OutputIsDynamic)
            l_r.resize(ORR);

          if constexpr (Base::JacobianIsDynamic)
            jx_r.resize(ORR, IRR);

          for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
            for (int j = 0; j < IRR; j++)
              x_r[j] = x[j][i];
            for (int j = 0; j < ORR; j++)
              l_r[j] = adjvars[j][i];

            this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
                x_r, fx_r, jx_r, gx_r, hx_r, l_r);
            for (int j = 0; j < ORR; j++)
              fx[j][i] = fx_r[j];
            for (int j = 0; j < IRR; j++)
              adjgrad[j][i] = gx_r[j];

            for (int j = 0; j < IRR; j++)
              for (int k = 0; k < ORR; k++)
                jx(k, j)[i] = jx_r(k, j);

            for (int j = 0; j < IRR; j++)
              for (int k = 0; k < IRR; k++)
                adjhess(k, j)[i] = hx_r(k, j);

            fx_r.setZero();
            jx_r.setZero();
            gx_r.setZero();
            hx_r.setZero();
          }

        } else {
          this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
              x, fx_, jx_, adjgrad_, adjhess_, adjvars);
        }
      } else {
        this->derived().compute_jacobian_adjointgradient_adjointhessian_impl(
            x, fx_, jx_, adjgrad_, adjhess_, adjvars);
      }
    }

    template<class InType, class AdjHessType, class AdjVarType>
    inline void adjointhessian(ConstVectorBaseRef<InType> x,
                               ConstMatrixBaseRef<AdjHessType> adjhess_,
                               ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      Output<Scalar> fx(this->ORows());
      Input<Scalar> adjgrad(this->IRows());
      Jacobian<Scalar> jx(this->ORows(), this->IRows());
      fx.setZero();
      jx.setZero();
      adjgrad.setZero();

      this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess_, adjvars);
    }

    template<class InType, class AdjVarType>
    inline Hessian<typename InType::Scalar> adjointhessian(ConstVectorBaseRef<InType> x,
                                                           ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      Hessian<Scalar> adjhess(this->IRows(), this->IRows());
      adjhess.setZero();
      this->derived().adjointhessian(x, adjhess, adjvars);
      return adjhess;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class GenType>
    GenType MakeGeneric() const {
      return GenType(this->derived());
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*
      Implements the derived objects default implemenation of right_jacobian_product.
      This function mutlplies some arbitarly structured matrix, left, on the right by the matrix, right,
      which is assumed have identical non-zero structure to the jacobian of Derived. The output
      of this product is assigned to the target_ matrix according the the Assigment type.
      The constexpr bool, aliased, indicates whether or not any of the three matrices, target_,left, or right
      share any common memory or are the exact same matrix. Abstracting out the this product operation, allows
      us to avoid multupllying explicit blocks of zeros that may be present in right, and can be computed from
      the input domain of derived. The default implementation here only detects non-zero columns as
      deternmiend from the input domain. The actual implementaion of the products are in
      DenseFunctionOperations.h, and the one called depends on whether the input domain of Derived is known at
      compile time or not. Derived objects can overload this function in order to implement a product that
      takes advantage of the exact non-zero pattern of the jacobian.
    */
    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left,
                                       ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {

      // Run Time Input Domain
      if constexpr (Base::InputIsDynamic
                    || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {

        if constexpr (Base::InputIsDynamic) {
          ASSET::right_jacobian_product_dynamic_impl(this->SubDomains, target_, left, right, assign, aliased);
        } else {
          ASSET::right_jacobian_product_impl(target_, left, right, assign, aliased);
        }

      } else {

        // Compile Time Input Domain
        using DMN = typename Derived::INPUT_DOMAIN;
        ASSET::right_jacobian_product_constant_impl(DMN(), target_, left, right, assign, aliased);
      }
    }

    /*
      Implements the derived objects default and only implemenation of right_jacobian_domain_product.
      This function mutlplies some arbitarly structured matrix, left, on the right by the matrix, right,
      which is assumed have the input domain structure to the jacobian of Derived. The output
      of this product is assigned to the target_ matrix according the the Assigment type.
      The constexpr bool, aliased, indicates whether or not any of the three matrices, target_,left, or right
      share any common memory or are the exact same matrix. Abstracting out the this product operation, allows
      us to avoid multupllying explicit blocks of zeros that may be present in right, and can be computed from
      the input domain of derived. The default implementation here only detects non-zero columns as
      deternmiend from the input domain. The actual implementaion of the products are in
      DenseFunctionOperations.h, and the one called depends on whether the input domain of Derived is known at
      compile time or not. Derived objects should not overload this function.

    */
    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_domain_product(ConstMatrixBaseRef<Target> target_,
                                              ConstEigenBaseRef<Left> left,
                                              ConstEigenBaseRef<Right> right,
                                              Assignment assign,
                                              std::bool_constant<Aliased> aliased) const {
      this->right_jacobian_product(target_, left, right, assign, aliased);
    }

    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                          ConstEigenBaseRef<Left> left,
                                          ConstEigenBaseRef<Right> right,
                                          Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
      if constexpr (Base::InputIsDynamic
                    || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          ASSET::symetric_jacobian_product_dynamic_impl(
              this->SubDomains, target_, left, right, assign, aliased);
        } else {
          ASSET::symetric_jacobian_product_impl(target_, left, right, assign, aliased);
        }

      } else {
        using DMN = typename Derived::INPUT_DOMAIN;
        ASSET::symetric_jacobian_product_constant_impl(DMN(), target_, left, right, assign, aliased);
      }
    }


    /*
     * Accumulates a matrix with the same non-zero structure as the jacobian of
     * Derived into the target_ matrix according to the speccified Assignment type.
     */
    template<class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right,
                                    Assignment assign) const {
      if constexpr (Base::InputIsDynamic
                    || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          ASSET::accumulate_matrix_dynamic_domain_impl(this->SubDomains, target_, right, assign);

        } else {
          ASSET::accumulate_impl(target_, right, assign);
        }

      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        ConstMatrixBaseRef<JacType> right_ref(right.derived());
        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
              constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
              ASSET::accumulate_impl(target_ref.template middleCols<size>(start, size),
                                     right_ref.template middleCols<size>(start, size),
                                     assign);
            });
      }
    }
    /*
     * Accumulates a vector with the same non-zero structure as the jacobian of
     * Derived into the target_ matrix according to the speccified Assignment type.
     */
    template<class Target, class JacType, class Assignment>
    inline void accumulate_gradient(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right,
                                    Assignment assign) const {
      if constexpr (Base::InputIsDynamic
                    || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          ASSET::accumulate_vector_dynamic_domain_impl(this->SubDomains, target_, right, assign);
        } else {
          ASSET::accumulate_impl(target_, right, assign);
        }
      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        ConstMatrixBaseRef<JacType> right_ref(right.derived());
        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
              constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
              ASSET::accumulate_impl(target_ref.template segment<size>(start, size),
                                     right_ref.template segment<size>(start, size),
                                     assign);
            });
      }
    }

    /*
     * Accumulates a matrix with the same non-zero structure as the hessian of
     * Derived into the target_ matrix according to the specified Assignment type. Does
     * absolutely nothing if Derived is known to be linear at compile time.
     */
    template<class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right,
                                   Assignment assign) const {
      if constexpr (Derived::IsLinearFunction) {
      } else if constexpr (Base::InputIsDynamic
                           || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          ASSET::accumulate_symetric_matrix_dynamic_domain_impl(this->SubDomains, target_, right, assign);
        } else {
          ASSET::accumulate_impl(target_, right, assign);
        }

      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        ConstMatrixBaseRef<JacType> right_ref(right.derived());
        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());

        if constexpr (sds == 1) {
          constexpr int start = Derived::INPUT_DOMAIN::SubDomains[0][0];
          constexpr int size = Derived::INPUT_DOMAIN::SubDomains[0][1];

          ASSET::accumulate_impl(target_ref.template block<size, size>(start, start, size, size),
                                 right_ref.template block<size, size>(start, start, size, size),
                                 assign);

        } else if constexpr (sds == 2) {
          constexpr int Start1 = Derived::INPUT_DOMAIN::SubDomains[0][0];
          constexpr int Size1 = Derived::INPUT_DOMAIN::SubDomains[0][1];

          constexpr int Start2 = Derived::INPUT_DOMAIN::SubDomains[1][0];
          constexpr int Size2 = Derived::INPUT_DOMAIN::SubDomains[1][1];

          ASSET::accumulate_impl(target_ref.template block<Size1, Size1>(Start1, Start1, Size1, Size1),
                                 right_ref.template block<Size1, Size1>(Start1, Start1, Size1, Size1),
                                 assign);

          ASSET::accumulate_impl(target_ref.template block<Size2, Size2>(Start2, Start2, Size2, Size2),
                                 right_ref.template block<Size2, Size2>(Start2, Start2, Size2, Size2),
                                 assign);

          ASSET::accumulate_impl(target_ref.template block<Size1, Size2>(Start1, Start2, Size1, Size2),
                                 right_ref.template block<Size1, Size2>(Start1, Start2, Size1, Size2),
                                 assign);

          ASSET::accumulate_impl(target_ref.template block<Size2, Size1>(Start2, Start1, Size2, Size1),
                                 right_ref.template block<Size2, Size1>(Start2, Start1, Size2, Size1),
                                 assign);

        } else {
          ASSET::constexpr_for_loop(
              std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
                constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
                ASSET::accumulate_impl(target_ref.template middleCols<size>(start, size),
                                       right_ref.template middleCols<size>(start, size),
                                       assign);
              });
        }
      }
    }


    /*
     * Accumulates a vector with the input domain structure s of
     * Derived into the target_ matrix according to the speccified Assignment type.
     */
    template<class Target, class JacType, class Assignment>
    inline void accumulate_matrix_domain(ConstMatrixBaseRef<Target> target_,
                                         ConstMatrixBaseRef<JacType> right,
                                         Assignment assign) const {
      if constexpr (Base::InputIsDynamic
                    || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          ASSET::accumulate_matrix_dynamic_domain_impl(this->SubDomains, target_, right, assign);
        } else {
          ASSET::accumulate_impl(target_, right, assign);
        }

      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        ConstMatrixBaseRef<JacType> right_ref(right.derived());
        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());

        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
              constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
              ASSET::accumulate_impl(target_ref.template middleCols<size>(start, size),
                                     right_ref.template middleCols<size>(start, size),
                                     assign);
            });
      }
    }

    template<class Target, class JacType>
    inline void accumulate_product_hessian(ConstMatrixBaseRef<Target> target_,
                                           ConstMatrixBaseRef<JacType> right) const {
      MatrixBaseRef<Target> target_ref(target_.const_cast_derived());

      if constexpr (Derived::InputIsDynamic) {
        const int sds = this->SubDomains.cols();
        if (sds == 0) {
          target_ref += right + right.transpose();
        } else {
          for (int i = 0; i < sds; i++) {
            int start = this->SubDomains(0, i);
            int size = this->SubDomains(1, i);
            target_ref.middleCols(start, size) += right.middleCols(start, size);
            target_ref.middleRows(start, size) += right.middleCols(start, size).transpose();
          }
        }
      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
              constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
              target_ref.template middleCols<size>(start, size) +=
                  right.template middleCols<size>(start, size);
              target_ref.template middleRows<size>(start, size) +=
                  right.template middleCols<size>(start, size).transpose();
            });
      }
    }


    template<class Target, class Scalar>
    inline void scale_impl(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      target *= s;
    }

    template<class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      if constexpr (Base::InputIsDynamic
                    || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          ASSET::scale_matrix_dynamic_domain_impl(this->SubDomains, target_, s);

        } else {
          this->scale_impl(target_, s);
        }

      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
              constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
              this->scale_impl(target_ref.template middleCols<size>(start, size), s);
            });
      }
    }
    template<class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      if constexpr (Base::InputIsDynamic
                    || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          ASSET::scale_vector_dynamic_domain_impl(this->SubDomains, target_, s);

        } else {
          this->scale_impl(target_, s);
        }
      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
              constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
              this->scale_impl(target_ref.template segment<size>(start, size), s);
            });
      }
    }
    template<class Target, class Scalar>
    inline void scale_hessian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      if constexpr (Derived::IsLinearFunction) {
      } else if constexpr (Base::InputIsDynamic
                           || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          ASSET::scale_matrix_dynamic_domain_impl(this->SubDomains, target_, s);

        } else {
          this->scale_impl(target_, s);
        }
      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        MatrixBaseRef<Target> target_ref(target_.const_cast_derived());
        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
              constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
              this->scale_impl(target_ref.template middleCols<size>(start, size), s);
            });
      }
    }


    /*
     * Zeros target_ matrix with the same input domain as derived.
     */
    template<class Target>
    inline void zero_matrix_domain(ConstMatrixBaseRef<Target> target_) const {

      MatrixBaseRef<Target> target_ref(target_.const_cast_derived());

      if constexpr (Base::InputIsDynamic
                    || std::is_same<typename Derived::INPUT_DOMAIN, INPUT_DOMAIN>::value) {
        if constexpr (Base::InputIsDynamic) {
          const int sds = this->SubDomains.cols();
          if (sds == 0) {
            target_ref.setZero();
          } else {
            for (int i = 0; i < sds; i++) {
              int start = this->SubDomains(0, i);
              int size = this->SubDomains(1, i);
              target_ref.middleCols(start, size).setZero();
            }
          }
        } else {
          target_ref.setZero();
        }

      } else {
        constexpr int sds = Derived::INPUT_DOMAIN::SubDomains.size();
        ASSET::constexpr_for_loop(
            std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
              constexpr int start = Derived::INPUT_DOMAIN::SubDomains[i.value][0];
              constexpr int size = Derived::INPUT_DOMAIN::SubDomains[i.value][1];
              target_ref.template middleCols<size>(start, size).setZero();
            });
      }
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<bool DoHessian, class Scalar>
    void RunPrintTime(ConstEigenRef<Input<double>> xt,
                      volatile int n,
                      std::bool_constant<DoHessian> dh,
                      Scalar s) const {
      Input<Scalar> x = xt.template cast<Scalar>();
      Output<Scalar> fx(this->ORows());
      fx.setZero();
      Jacobian<Scalar> jx(this->ORows(), this->IRows());

      jx.setZero();

      this->derived().compute_jacobian(x, fx, jx);

      std::cout << "f(x):" << std::endl << fx.transpose() << std::endl << std::endl;
      std::cout << "j(x):" << std::endl << jx << std::endl << std::endl;
      Eigen::BenchTimer tfun;
      tfun.start();
      // x.setRandom();
      Output<Scalar> fxsum = fx;
      Jacobian<Scalar> jxsum = jx;

      Scalar t = Scalar(0.0);
      for (int i = 0; i < n; i++) {
        // if (i % 2000 == 0) x.setRandom();
        x[0] += 1.0 / double(n + 1);
        this->derived().compute(x, fx);
        fxsum += fx;
        t += fx[0];
      }
      tfun.stop();
      t += fxsum.norm();
      std::cout << "Dummy Value: " << t << std::endl;

      std::cout << "Compute Time:" << tfun.total() * 1000.0 << " ms" << std::endl;

      // x.setRandom();

      Eigen::BenchTimer tfun2;
      tfun2.start();
      for (int i = 0; i < n; i++) {
        // if (i % 2000 == 0) x.setRandom();
        x[0] += 1.0 / double(n + 1);
        this->derived().compute_jacobian(x, fx, jx);
        fxsum += fx;
        t += jx(0, 0) + jx(this->ORows() - 1, this->IRows() - 1);
      }
      tfun2.stop();
      t += jx.norm() + fxsum.norm();

      std::cout << "Compute_jacobian Time:" << tfun2.total() * 1000.0 << " ms" << std::endl;
      std::cout << "Dummy Value: " << t << std::endl;

      if constexpr (DoHessian) {
        t = 0;
        // x.setRandom();
        fx.setZero();
        Output<Scalar> l(this->ORows());
        l.setOnes();
        Input<Scalar> gx(this->IRows());
        gx.setZero();
        Hessian<Scalar> hx(this->IRows(), this->IRows());
        hx.setZero();

        Eigen::BenchTimer tfun3;
        tfun3.start();
        for (int i = 0; i < n; i++) {
          // if (i % 2000 == 0) x.setRandom();
          x[0] += 1.0 / double(n + 1);
          this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, l);
          fxsum += fx;
          t += jx(0, 0) + jx(this->ORows() - 1, this->IRows() - 1) + hx(0, 0)
               + hx(this->IRows() - 1, this->IRows() - 1) + gx[0];
        }
        tfun3.stop();
        t += jx.norm() + fxsum.norm() + hx.sum();

        std::cout << "Compute_jacobian_hessian Time:" << tfun3.total() * 1000.0 << " ms" << std::endl;
        std::cout << "Dummy Value: " << t << std::endl;
      }
    }

    void rpt(Eigen::VectorXd x, int n) const {
      double s = 1;
      this->RunPrintTime(x, n, std::bool_constant<true>(), s);
    }

    void ArrayTest(Eigen::VectorXd x, int n) {
      double s = 1;
      this->RunPrintTime(x, n, std::bool_constant<true>(), s);

      Eigen::Array<double, 2, 1> s2;
      this->RunPrintTime(x, n / 2, std::bool_constant<true>(), s2);

      Eigen::Array<double, 4, 1> s4;
      this->RunPrintTime(x, n / 4, std::bool_constant<true>(), s4);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*
     * Counts the number of non-zero elements in the lower triangle of the hessian and
     * the jacobian of derived and represents the number of elements that will be explicilty scattered
     * into the KKT matrix of an optimization problem on each function call.
     * For dense functions, in almost all cases they are all non-zero, or are assumed non-zero even if
     * they are not. However, this method will account for explicit non-zeros if the HessianElemIsNonZero ex.
     * is explicity overloaded by derived. The sum of the non-zeros here is used when interfacing whith the
     * optimizer throutght the wrapper classes ConstraintFunction and ObjectiveFunction.
     */
    int numKKTEles(bool dojac, bool dohess) const {
      int hesselems = 0;
      int jacelems = 0;
      for (int i = 0; i < this->IRows(); i++) {
        if (dohess) {
          for (int j = i; j < this->IRows(); j++) {
            if (this->derived().HessianElemIsNonZero(j, i))
              hesselems++;
          }
        }
        if (dojac) {
          for (int j = 0; j < this->ORows(); j++) {
            if (this->derived().JacobianElemIsNonZero(j, i))
              jacelems++;
          }
        }
      }
      return ((hesselems + jacelems));
    }

    /*
     * This function is responsibe for determing the target locations of the jacobian and hessian elements
     * of derived inside of the KKT matrix of an optimization problem when it is used as a constraint. Meta
     * data containing the locations of the input variables and constraint row for each call of derived are
     * held in the Solver indexing data struct.
     */

    void getKKTSpace(EigenRef<Eigen::VectorXi> KKTrows,
                     EigenRef<Eigen::VectorXi> KKTcols,
                     int& freeloc,
                     int conoffset,
                     bool dojac,
                     bool dohess,
                     SolverIndexingData& data) {
      data.InnerKKTStarts.resize(data.NumAppl());

      for (int V = 0; V < data.NumAppl(); V++) {
        data.InnerKKTStarts[V] = freeloc;
        for (int i = 0; i < this->IRows(); i++) {
          if (dohess) {
            for (int j = i; j < this->IRows(); j++) {
              if (this->derived().HessianElemIsNonZero(j, i)) {
                KKTrows[freeloc] = data.VLoc(j, V);
                KKTcols[freeloc] = data.VLoc(i, V);
                freeloc++;
              }
            }
          }
          if (dojac) {
            for (int j = 0; j < this->ORows(); j++) {
              if (this->derived().JacobianElemIsNonZero(j, i)) {
                KKTrows[freeloc] = data.CLoc(j, V) + conoffset;
                KKTcols[freeloc] = data.VLoc(i, V);
                freeloc++;
              }
            }
          }
        }
      }
    }

    ////
    /*
    * This function is the interface allowing Derived to be used as a constraint inside psiopt.
    * It computes both the value of the function and its jacobian and is called during psiopts solve
    * algorithm. Vector X is the total variables vector for the full optimization problem, and FX is the total
    eqaility or inequality constraints vector for the problem. Sparse Matrix KKTMat is the full KKT matrix of
    the problem. KKTLocations contains the location where every non-zero element of each jacobian arrising
    from of a call to derived should be summed into KKTMat. KKTClashes contains a flag (-1) indicating if we
    can sum into the corresponding column without clashing with the work of another function on another
    thread. If the flag is greater than zero, the element of KKTClash is the index of the mutex in KKTLocks,
      which derived should lock when summing into the column and then release. This logic is controled by
    KKTFillAll and KKTFillJac. Do not overload these functions unless you really know what you are doing.
    */

    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex>& KKTLocks,
                              const SolverIndexingData& data) const {
      Input<double> x(this->IRows());
      Eigen::Map<Output<double>> fx(NULL, this->ORows());

      Jacobian<double> jx(this->ORows(), this->IRows());

      auto ScalarImpl = [&](int start, int stop) {
        for (int V = start; V < stop; V++) {
          this->gatherInput(X, x, V, data);

          new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V], this->ORows());
          fx.setZero();
          jx.setZero();
          this->derived().compute_jacobian(x, fx, jx);

          this->derived().KKTFillJac(V, jx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
      };
      const int IRR = this->IRows();
      const int ORR = this->ORows();
      auto VectorImpl = [&]() {
        using SuperScalar = ASSET::DefaultSuperScalar;
        constexpr int vsize = SuperScalar::SizeAtCompileTime;
        int Packs = data.NumAppl() / vsize;

        Input<SuperScalar> x_vect(this->IRows());
        Output<SuperScalar> fx_vect(this->ORows());
        Jacobian<SuperScalar> jx_vect(this->ORows(), this->IRows());

        for (int i = 0; i < Packs; i++) {
          for (int j = 0; j < vsize; j++) {
            int V = i * vsize + j;
            this->gatherInput(X, x, V, data);
            for (int k = 0; k < IRR; k++) {
              x_vect[k][j] = x[k];
            }
          }
          fx_vect.setZero();
          jx_vect.setZero();
          this->derived().compute_jacobian(x_vect, fx_vect, jx_vect);

          for (int j = 0; j < vsize; j++) {
            int V = i * vsize + j;
            for (int k = 0; k < IRR; k++) {
              for (int l = 0; l < ORR; l++) {
                jx(l, k) = jx_vect(l, k)[j];
              }
            }
            this->derived().KKTFillJac(V, jx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
          }
          for (int j = 0; j < vsize; j++) {
            int V = i * vsize + j;
            new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V], this->ORows());
            for (int l = 0; l < ORR; l++) {
              fx[l] = fx_vect[l][j];
            }
          }
        }

        ScalarImpl(Packs * vsize, data.NumAppl());
      };

      if constexpr (Derived::IsVectorizable) {
        if (this->derived().EnableVectorization) {
          VectorImpl();
        } else {
          ScalarImpl(0, data.NumAppl());
        }
      } else {
        ScalarImpl(0, data.NumAppl());
      }
    }

    void constraints_jacobian_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                              ConstEigenRef<Eigen::VectorXd> L,
                                              Eigen::Ref<Eigen::VectorXd> FX,
                                              Eigen::Ref<Eigen::VectorXd> AGX,
                                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                                              EigenRef<Eigen::VectorXi> KKTLocations,
                                              EigenRef<Eigen::VectorXi> KKTClashes,
                                              std::vector<std::mutex>& KKTLocks,
                                              const SolverIndexingData& data) const {
      Input<double> x(this->IRows());
      Output<double> l(this->ORows());


      Eigen::Map<Output<double>> fx(NULL, this->ORows());
      Eigen::Map<Input<double>> agx(NULL, this->IRows());

      Jacobian<double> jx(this->ORows(), this->IRows());

      for (int V = 0; V < data.NumAppl(); V++) {
        this->gatherInput(X, x, V, data);
        this->gatherMult(L, l, V, data);


        new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V], this->ORows());
        new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.InnerGradientStarts[V], this->IRows());

        fx.setZero();
        agx.setZero();
        jx.setZero();

        this->derived().compute_jacobian_adjointgradient(x, fx, jx, agx, l);

        this->derived().KKTFillJac(V, jx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
      }
    }


    /*
    * This function is the interface allowing Derived to be used as a constraint inside psiopt.
    * It computes both the value of the function and its jacobain gradient and hessian and is called during
    psiopts optimize
    * algorithm. Vector X is the total variables vector for the full optimization problem, and FX is the total
    eqaility or inequality constraints vector for the problem. Vector L is the vector of  equality or
    inequlity lagrange multipliers for the full optimization problem and. Sparse Matrix KKTMat is the full KKT
    matrix of the problem. KKTLocations contains the location where every non-zero element of each jacobian
    arrising from of call to derived should be summed into KKTMat. KKTClashes contains a flag (-1) indicating
    if we can sum into the corresponding column without clashing with the work of another function on another
    thread. If the flag is greater than zero, the element of KKTClash is the index of the mutex in KKTLocks,
      which derived should lock when summing into the column and then release. This logic is controled by
    KKTFillAll and KKTFillJac. Do not overload these functions unless you really know what you are doing.
    */

    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X,
        ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX,
        EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations,
        EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const {
      Input<double> x(this->IRows());
      Output<double> l(this->ORows());

      Eigen::Map<Output<double>> fx(NULL, this->ORows());
      Eigen::Map<Input<double>> agx(NULL, this->IRows());

      Jacobian<double> jx(this->ORows(), this->IRows());
      Hessian<double> hx(this->IRows(), this->IRows());

      auto ScalarImpl = [&](int start, int stop) {
        for (int V = start; V < stop; V++) {
          this->gatherInput(X, x, V, data);
          this->gatherMult(L, l, V, data);

          new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V], this->ORows());
          new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.InnerGradientStarts[V], this->IRows());

          fx.setZero();
          agx.setZero();
          jx.setZero();
          hx.setZero();


          this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, agx, hx, l);


          this->derived().KKTFillAll(V, jx, hx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
        }
      };

      const int IRR = this->IRows();
      const int ORR = this->ORows();
      auto VectorImpl = [&]() {
        using SuperScalar = ASSET::DefaultSuperScalar;
        constexpr int vsize = SuperScalar::SizeAtCompileTime;
        int Packs = data.NumAppl() / vsize;


        Input<SuperScalar> x_vect(this->IRows());
        Output<SuperScalar> fx_vect(this->ORows());
        Jacobian<SuperScalar> jx_vect(this->ORows(), this->IRows());
        Gradient<SuperScalar> agx_vect(this->IRows());
        Hessian<SuperScalar> hx_vect(this->IRows(), this->IRows());
        Output<SuperScalar> l_vect(this->ORows());

        for (int i = 0; i < Packs; i++) {
          for (int j = 0; j < vsize; j++) {
            int V = i * vsize + j;
            this->gatherInput(X, x, V, data);
            this->gatherMult(L, l, V, data);

            for (int k = 0; k < IRR; k++) {
              x_vect[k][j] = x[k];
            }
            for (int k = 0; k < ORR; k++) {
              l_vect[k][j] = l[k];
            }
          }
          fx_vect.setZero();
          jx_vect.setZero();
          hx_vect.setZero();
          agx_vect.setZero();

          this->derived().compute_jacobian_adjointgradient_adjointhessian(
              x_vect, fx_vect, jx_vect, agx_vect, hx_vect, l_vect);

          for (int j = 0; j < vsize; j++) {
            int V = i * vsize + j;
            for (int k = 0; k < IRR; k++) {
              for (int l = 0; l < ORR; l++) {
                jx(l, k) = jx_vect(l, k)[j];
              }
            }
            for (int k = 0; k < IRR; k++) {
              for (int l = k; l < IRR; l++) {
                hx(l, k) = hx_vect(l, k)[j];
              }
            }
            this->derived().KKTFillAll(V, jx, hx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
          }
          for (int j = 0; j < vsize; j++) {
            int V = i * vsize + j;
            new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V], this->ORows());
            for (int l = 0; l < ORR; l++) {
              fx[l] = fx_vect[l][j];
            }
            new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.InnerGradientStarts[V], this->IRows());
            for (int l = 0; l < IRR; l++) {
              agx[l] = agx_vect[l][j];
            }
          }
        }

        ScalarImpl(Packs * vsize, data.NumAppl());
      };

      if constexpr (Derived::IsVectorizable) {
        if (this->derived().EnableVectorization) {
          VectorImpl();
        } else {
          ScalarImpl(0, data.NumAppl());
        }
      } else {
        ScalarImpl(0, data.NumAppl());
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   protected:
    inline constexpr bool JacobianElemIsNonZero(int row, int col) const {
      return true;
    }
    inline constexpr bool HessianElemIsNonZero(int row, int col) const {
      return !Derived::IsLinearFunction;
    }
    inline void AddHessianElem(double v, int row, int col, double* mpt, const int* lpt, int& freeloc) const {
      if constexpr (!Derived::IsLinearFunction) {
        mpt[lpt[freeloc]] += v;
        freeloc++;
      }
    }
    inline void AddJacobianElem(double v, int row, int col, double* mpt, const int* lpt, int& freeloc) const {
      mpt[lpt[freeloc]] += v;
      freeloc++;
    }

    void KKTFillAll(int Apl,
                    const Jacobian<double>& jx,
                    const Hessian<double>& hx,
                    Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                    EigenRef<Eigen::VectorXi> KKTLocs,
                    EigenRef<Eigen::VectorXi> VarClashes,
                    std::vector<std::mutex>& ClashLocks,
                    const SolverIndexingData& data) const {
      int freeloc = data.InnerKKTStarts[Apl];
      double* mpt = KKTmat.valuePtr();
      const int* lpt = KKTLocs.data();
      int ActiveVar;
      const int IRR = (Base::IRC > 0) ? Base::IRC : this->IRows();
      const int ORR = (Base::ORC > 0) ? Base::ORC : this->ORows();

      auto Lock = [&](int var) {
        if (VarClashes[var] == -1) {
          //// uncontested
        } else {
          /// contested lock mutex
          ClashLocks[VarClashes[var]].lock();
        }
      };
      auto UnLock = [&](int var) {
        if (VarClashes[var] == -1) {
          //// uncontested
        } else {
          /// contested unlock mutex
          ClashLocks[VarClashes[var]].unlock();
        }
      };


      const bool uniquecon = data.unique_constraints;

      // bool uc = data.unique_constraints;
      for (int i = 0; i < IRR; i++) {
        ActiveVar = data.VLoc(i, Apl);
        Lock(ActiveVar);
        ///// insert hessian column symetrically
        for (int j = i; j < IRR; j++) {
          this->derived().AddHessianElem(hx(j, i), j, i, mpt, lpt, freeloc);
        }
        if (uniquecon)
          UnLock(ActiveVar);
        ///// insert jacobian column
        for (int j = 0; j < ORR; j++) {
          this->derived().AddJacobianElem(jx(j, i), j, i, mpt, lpt, freeloc);
        }
        ///////////////////////////////////////////////////////////////////////////////
        if (!uniquecon)
          UnLock(ActiveVar);
      }
    }

    void KKTFillJac(int Apl,
                    const Jacobian<double>& jx,
                    Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                    Eigen::Ref<Eigen::VectorXi> KKTLocs,
                    Eigen::Ref<Eigen::VectorXi> VarClashes,
                    std::vector<std::mutex>& ClashLocks,
                    const SolverIndexingData& data) const {
      int freeloc = data.InnerKKTStarts[Apl];
      double* mpt = KKTmat.valuePtr();
      const int* lpt = KKTLocs.data();
      int ActiveVar;

      if (data.unique_constraints) {
        for (int i = 0; i < this->IRows(); i++) {
          ActiveVar = data.VLoc(i, Apl);
          for (int j = i; j < this->IRows(); j++) {
            if (this->derived().HessianElemIsNonZero(j, i))
              freeloc++;
          }
          for (int j = 0; j < this->ORows(); j++) {
            this->derived().AddJacobianElem(jx(j, i), j, i, mpt, lpt, freeloc);
          }
        }
      } else {
        for (int i = 0; i < this->IRows(); i++) {
          ActiveVar = data.VLoc(i, Apl);
          if (VarClashes[ActiveVar] == -1) {
            //// uncontested
          } else {
            /// contested lock mutex
            ClashLocks[VarClashes[ActiveVar]].lock();
          }

          //////////////////////////Mutex
          /// Locked////////////////////////////////////////
          //////////////////////////////////////////////////////////////////////////////
          ///// insert hessian column symetrically
          for (int j = i; j < this->IRows(); j++) {
            if (this->derived().HessianElemIsNonZero(j, i))
              freeloc++;
          }
          ///// insert jacobain column

          for (int j = 0; j < this->ORows(); j++) {
            this->derived().AddJacobianElem(jx(j, i), j, i, mpt, lpt, freeloc);
          }
          ///////////////////////////////////////////////////////////////////////////////
          if (VarClashes[ActiveVar] == -1) {
            //// uncontested
          } else {
            /// contested unlock mutex
            ClashLocks[VarClashes[ActiveVar]].unlock();
          }
        }
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*
     * Code for binding public members to python, self explanatory, hopefully....
     * Could/should probably be moved to another file, but leaving it here for now.
     */

   public:
    template<class PYClass>
    static void DenseBaseBuild(PYClass& obj) {


      using Gen = GenericFunction<-1, -1>;

      obj.def("IRows", &Derived::IRows);
      obj.def("ORows", &Derived::ORows);
      obj.def("name", &Derived::name);
      obj.def("compute", [](const Derived& func, ConstEigenRef<Input<double>> x) {
        if (x.size() != func.IRows())
          throw std::invalid_argument("Incorrectly sized input to function");
        Output<double> fx(func.ORows());
        fx.setZero();
        func.derived().compute(x, fx);
        return fx;
      });
      obj.def("__call__", [](const Derived& func, ConstEigenRef<Input<double>> x) {
        if (x.size() != func.IRows())
          throw std::invalid_argument("Incorrectly sized input to function");
        Output<double> fx(func.ORows());
        fx.setZero();
        func.derived().compute(x, fx);
        return fx;
      });

      obj.def("jacobian", [](const Derived& func, ConstEigenRef<Input<double>> x) {
        if (x.size() != func.IRows())
          throw std::invalid_argument("Incorrectly sized input to function");
        Jacobian<double> jx(func.ORows(), func.IRows());
        jx.setZero();
        func.derived().jacobian(x, jx);
        return jx;
      });
      obj.def("adjointgradient",
              [](const Derived& func, ConstEigenRef<Input<double>> x, ConstEigenRef<Output<double>> lm) {
                if (x.size() != func.IRows())
                  throw std::invalid_argument("Incorrectly sized input to function");
                if (lm.size() != func.ORows())
                  throw std::invalid_argument("Incorrectly sized multiplier input to function");

                Gradient<double> ax(func.IRows());
                ax.setZero();
                func.derived().adjointgradient(x, ax, lm);
                return ax;
              });
      obj.def("adjointhessian",
              [](const Derived& func, ConstEigenRef<Input<double>> x, ConstEigenRef<Output<double>> lm) {
                if (x.size() != func.IRows())
                  throw std::invalid_argument("Incorrectly sized input to function");
                if (lm.size() != func.ORows())
                  throw std::invalid_argument("Incorrectly sized multiplier input to function");

                Hessian<double> hx(func.IRows(), func.IRows());
                hx.setZero();
                func.derived().adjointhessian(x, hx, lm);
                return hx;
              });

      obj.def("computeall",
              [](const Derived& func, ConstEigenRef<Input<double>> x, ConstEigenRef<Output<double>> lm) {
                if (x.size() != func.IRows())
                  throw std::invalid_argument("Incorrectly sized input to function");
                if (lm.size() != func.ORows())
                  throw std::invalid_argument("Incorrectly sized multiplier input to function");

                Output<double> fx(func.ORows());

                Jacobian<double> jx(func.ORows(), func.IRows());
                Gradient<double> gx(func.IRows());
                Hessian<double> hx(func.IRows(), func.IRows());

                fx.setZero();
                jx.setZero();
                gx.setZero();
                hx.setZero();

                func.derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);
                return std::tuple {fx, jx, gx, hx};
              });


      obj.def("rpt", &Derived::rpt);
      obj.def("vf", &Derived::template MakeGeneric<GenericFunction<-1, -1>>);


      if constexpr (OR == 1) {
        obj.def("sf", &Derived::template MakeGeneric<GenericFunction<-1, 1>>);
      }
    }

    template<class PYClass>
    static void DoubleMathBuild(PYClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;
      using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

      using SEG = Segment<-1, -1, -1>;
      using SEG2 = Segment<-1, 2, -1>;
      using SEG3 = Segment<-1, 3, -1>;
      using SEG4 = Segment<-1, 4, -1>;
      using ELEM = Segment<-1, 1, -1>;

      // Prevents numpy from overriding __radd__ and __rmul__
      obj.attr("__array_ufunc__") = py::none();


      obj.def(
          "__add__", [](const Derived& a, Output<double> b) { return BinGen(a + b); }, py::is_operator());
      obj.def(
          "__radd__", [](const Derived& a, Output<double> b) { return BinGen(a + b); }, py::is_operator());
      obj.def(
          "__sub__", [](const Derived& a, Output<double> b) { return BinGen(a - b); }, py::is_operator());
      obj.def(
          "__rsub__", [](const Derived& a, Output<double> b) { return BinGen(b - a); }, py::is_operator());

      obj.def(
          "__mul__", [](const Derived& a, double b) { return BinGen(a * b); }, py::is_operator());
      obj.def(
          "__rmul__", [](const Derived& a, double b) { return BinGen(a * b); }, py::is_operator());

      obj.def(
          "__neg__", [](const Derived& a) { return BinGen(a * (-1.0)); }, py::is_operator());

      obj.def(
          "__truediv__", [](const Derived& a, double b) { return BinGen(a * (1.0 / b)); }, py::is_operator());
      obj.def(
          "__truediv__",
          [](const Derived& a, const Segment<-1, 1, -1>& b) { return BinGen(a / b); },
          py::is_operator());


      if constexpr (OR == 1) {  // Scalars
        obj.def(
            "__add__", [](const Derived& a, double b) { return BinGen(a + b); }, py::is_operator());
        obj.def(
            "__radd__", [](const Derived& a, double b) { return BinGen(a + b); }, py::is_operator());
        obj.def(
            "__sub__", [](const Derived& a, double b) { return BinGen(a - b); }, py::is_operator());
        obj.def(
            "__rsub__", [](const Derived& a, double b) { return BinGen(b - a); }, py::is_operator());
        obj.def(
            "__rtruediv__", [](const Derived& a, double b) { return BinGen(b / a); }, py::is_operator());

        obj.def(
            "__rmul__",
            [](const Derived& a, const Eigen::VectorXd& b) { return Gen(MatrixScaled<Derived, -1>(a, b)); },
            py::is_operator());

        obj.def(
            "__mul__",
            [](const Derived& a, const Eigen::VectorXd& b) { return Gen(MatrixScaled<Derived, -1>(a, b)); },
            py::is_operator());
      }
    }

    template<class PYClass>
    static void UnaryMathBuild(PYClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;
      using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

      using SEG = Segment<-1, -1, -1>;
      using SEG2 = Segment<-1, 2, -1>;
      using SEG3 = Segment<-1, 3, -1>;
      using SEG4 = Segment<-1, 4, -1>;
      using ELEM = Segment<-1, 1, -1>;


      if constexpr (OR != 1) {

        if constexpr (!std::is_same<PYClass, py::module>::value) {
          obj.def("sum", [](const Derived& a) { return GenS(a.sum()); });
        }


        obj.def("normalized_power3", [](const Derived& a, Output<double> b) {
          return Gen((a + b).template normalized_power<3>());
        });
        obj.def("normalized_power3", [](const Derived& a, Output<double> b, double s) {
          return Gen(((a + b).template normalized_power<3>()) * s);
        });
        ////////////////////////////////////////////////////////////////////
        if constexpr (OR > 0) {  // already constant size

          obj.def("norm", [](const Derived& a) { return GenS(a.norm()); });
          obj.def("squared_norm", [](const Derived& a) { return GenS(a.squared_norm()); });
          obj.def("cubed_norm", [](const Derived& a) { return GenS(a.template norm_power<3>()); });

          obj.def("inverse_norm", [](const Derived& a) { return GenS(a.inverse_norm()); });
          obj.def("inverse_squared_norm", [](const Derived& a) { return GenS(a.inverse_squared_norm()); });
          obj.def("inverse_cubed_norm",
                  [](const Derived& a) { return GenS(a.template inverse_norm_power<3>()); });
          obj.def("inverse_four_norm",
                  [](const Derived& a) { return GenS(a.template inverse_norm_power<4>()); });

          obj.def("normalized", [](const Derived& a) { return Gen(a.normalized()); });

          obj.def("normalized_power2",
                  [](const Derived& a) { return Gen(a.template normalized_power<2>()); });


          obj.def("normalized_power4",
                  [](const Derived& a) { return Gen(a.template normalized_power<4>()); });
          obj.def("normalized_power5",
                  [](const Derived& a) { return Gen(a.template normalized_power<5>()); });


          obj.def("normalized_power3",
                  [](const Derived& a) { return Gen(a.template normalized_power<3>()); });


        } else {
          /// Try to fit these to constant size 2,3 if possible

          auto SizeSwitch = [](const auto& fun, auto Lam) {
            int orr = fun.ORows();
            // if (orr == 2)     return Lam(fun, std::integral_constant<int, 2>());
            // else
            if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          };

          obj.def("norm", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return GenS(Norm<size.value>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("squared_norm", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return GenS(SquaredNorm<size.value>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("cubed_norm", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return GenS(NormPower<size.value, 3>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("inverse_norm", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return GenS(InverseNorm<size.value>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("inverse_squared_norm", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return GenS(InverseSquaredNorm<size.value>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("inverse_cubed_norm", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return GenS(InverseNormPower<size.value, 3>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("inverse_four_norm", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return GenS(InverseNormPower<size.value, 4>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("normalized", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return Gen(Normalized<size.value>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("normalized_power2", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return Gen(NormalizedPower<size.value, 2>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("normalized_power3", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return Gen(NormalizedPower<size.value, 3>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("normalized_power4", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return Gen(NormalizedPower<size.value, 4>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
          obj.def("normalized_power5", [SizeSwitch](const Derived& fun) {
            auto Lam = [](const Derived& funt, auto size) {
              return Gen(NormalizedPower<size.value, 5>(funt.ORows()).eval(funt));
            };
            return SizeSwitch(fun, Lam);
          });
        }
      }

      if constexpr (OR == 1) {

        obj.def("sin", [](const Derived& a) { return GenS(a.Sin()); });
        obj.def("cos", [](const Derived& a) { return GenS(a.Cos()); });
        obj.def("tan", [](const Derived& a) { return GenS(a.Tan()); });
        obj.def("sqrt", [](const Derived& a) { return GenS(a.Sqrt()); });
        obj.def("exp", [](const Derived& a) { return GenS(a.Exp()); });
        obj.def("log", [](const Derived& e) { return GenS(CwiseLog<Derived>(e)); });
        obj.def("squared", [](const Derived& a) { return GenS(a.Square()); });
        obj.def("arcsin", [](const Derived& e) { return GenS(CwiseArcSin<Derived>(e)); });
        obj.def("arccos", [](const Derived& e) { return GenS(CwiseArcCos<Derived>(e)); });
        obj.def("arctan", [](const Derived& e) { return GenS(CwiseArcTan<Derived>(e)); });

        obj.def("sinh", [](const Derived& e) { return GenS(CwiseSinH<Derived>(e)); });
        obj.def("cosh", [](const Derived& e) { return GenS(CwiseCosH<Derived>(e)); });
        obj.def("tanh", [](const Derived& e) { return GenS(CwiseTanH<Derived>(e)); });


        obj.def("arcsinh", [](const Derived& e) { return GenS(CwiseArcSinH<Derived>(e)); });
        obj.def("arccosh", [](const Derived& e) { return GenS(CwiseArcCosH<Derived>(e)); });
        obj.def("arctanh", [](const Derived& e) { return GenS(CwiseArcTanH<Derived>(e)); });

        obj.def("__abs__", [](const Derived& e) { return GenS(CwiseAbs<Derived>(e)); });
        obj.def("sign", [](const Derived& e) { return GenS(SignFunction<Derived>(e)); });


        obj.def("pow", [](const Derived& e, double power) { return GenS(CwisePow<Derived>(e, power)); });

        if constexpr (!std::is_same<PYClass, py::module>::value) {
          obj.def(
              "__pow__",
              [](const Derived& a, double b) { return GenS(CwisePow<Derived>(a, b)); },
              py::is_operator());
          obj.def(
              "__pow__",
              [](const Derived& a, int b) {
                if (b == 1)
                  return GenS(a);
                else if (b == 2)
                  return GenS(a.Square());
                return GenS(CwisePow<Derived>(a, b));
              },
              py::is_operator());
        }
      }
    }

    template<class PYClass>
    static void FunctionIndexingBuild(PYClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;
      using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

      using SEG = Segment<-1, -1, -1>;
      using SEG2 = Segment<-1, 2, -1>;
      using SEG3 = Segment<-1, 3, -1>;
      using SEG4 = Segment<-1, 4, -1>;
      using ELEM = Segment<-1, 1, -1>;


      obj.def("padded_lower", [](const Derived& a, int lpad) { return Gen(a.padded_lower(lpad)); });
      obj.def("padded_upper", [](const Derived& a, int upad) { return Gen(a.padded_upper(upad)); });
      obj.def("padded", [](const Derived& a, int upad, int lpad) { return Gen(a.padded(upad, lpad)); });

      constexpr bool is_seg = Is_Segment<Derived>::value || Is_Arguments<Derived>::value;

      using SegRetType = typename std::conditional<is_seg, SEG, GenericFunction<-1, -1>>::type;
      using ElemRetType = typename std::conditional<is_seg, ELEM, GenericFunction<-1, 1>>::type;


      obj.def("segment",
              [](const Derived& a, int start, int size) { return SegRetType(a.segment(start, size)); });
      obj.def("head", [](const Derived& a, int size) { return SegRetType(a.head(size)); });
      obj.def("tail", [](const Derived& a, int size) { return SegRetType(a.tail(size)); });

      obj.def("coeff", [](const Derived& a, int elem) { return ElemRetType(a.coeff(elem)); });
      obj.def(
          "__getitem__",
          [](const Derived& a, int elem) { return ElemRetType(a.coeff(elem)); },
          py::is_operator());
      obj.def(
          "__getitem__",
          [](const Derived& a, const py::slice& slice) {
            size_t start, stop, step, slicelength;
            if (!slice.compute(a.ORows(), &start, &stop, &step, &slicelength)) {
              throw py::error_already_set();
            }

            if (step != 1) {
              throw std::invalid_argument("Non continous slices not supported");
            }
            if (start >= a.ORows()) {
              throw std::invalid_argument("Segment index out of bounds.");
            }
            if (start > stop) {
              throw std::invalid_argument("Backward indexing not supported.");
            }
            if (slicelength <= 0) {
              throw std::invalid_argument("Slice length must be greater than 0.");
            }

            int start_ = start;
            int size_ = stop - start;

            return SegRetType(a.segment(start_, size_));
          },
          py::is_operator());


      // if constexpr (OR != 1) {
      //     obj.def("colmatrix", [](const Derived& a, int rows, int cols) {
      //         Gen agen(a);
      //         return agen.colmatrix(rows, cols);
      //         });
      //     obj.def("rowmatrix", [](const Derived& a, int rows, int cols) {
      //         Gen agen(a);
      //         return agen.rowmatrix(rows, cols);
      //         });
      // }


      if constexpr (OR < 0 || OR > 2) {
        using Seg2RetType = typename std::conditional<is_seg, SEG2, GenericFunction<-1, -1>>::type;

        auto seg2 = [](const Derived& a, int start) { return Seg2RetType(a.template segment<2>(start)); };
        auto head2 = [](const Derived& a) { return Seg2RetType(a.template segment<2>(0)); };
        auto tail2 = [](const Derived& a) { return Seg2RetType(a.template segment<2>(a.ORows() - 2)); };

        obj.def("segment_2", seg2);
        obj.def("head_2", head2);
        obj.def("tail_2", tail2);
        obj.def("segment2", seg2);
        obj.def("head2", head2);
        obj.def("tail2", tail2);
      }
      if constexpr (OR < 0 || OR > 3) {
        using Seg3RetType = typename std::conditional<is_seg, SEG3, GenericFunction<-1, -1>>::type;
        auto seg3 = [](const Derived& a, int start) { return Seg3RetType(a.template segment<3>(start)); };
        auto head3 = [](const Derived& a) { return Seg3RetType(a.template segment<3>(0)); };
        auto tail3 = [](const Derived& a) { return Seg3RetType(a.template segment<3>(a.ORows() - 3)); };
        obj.def("segment_3", seg3);
        obj.def("head_3", head3);
        obj.def("tail_3", tail3);
        obj.def("segment3", seg3);
        obj.def("head3", head3);
        obj.def("tail3", tail3);
      }
    }

    template<class PYClass>
    static void BinaryMathBuild(PYClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;
      using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

      using SEG = Segment<-1, -1, -1>;
      using SEG2 = Segment<-1, 2, -1>;
      using SEG3 = Segment<-1, 3, -1>;
      using SEG4 = Segment<-1, 4, -1>;
      using ELEM = Segment<-1, 1, -1>;


      constexpr bool is_seg = std::is_same<Derived, SEG>::value;
      constexpr bool is_seg2 = std::is_same<Derived, SEG2>::value;
      constexpr bool is_seg3 = std::is_same<Derived, SEG3>::value;

      constexpr bool is_elem = std::is_same<Derived, ELEM>::value;
      constexpr bool is_gen = std::is_same<Derived, Gen>::value;
      constexpr bool is_gens = std::is_same<Derived, GenS>::value;


      if constexpr (OR == 3 || OR == -1) {


        if constexpr (!is_seg)
          obj.def("cross",
                  [](const Derived& seg1, const SEG& seg2) { return Gen(crossProduct(seg1, seg2)); });

        if constexpr (!is_seg3)
          obj.def("cross",
                  [](const Derived& seg1, const SEG3& seg2) { return Gen(crossProduct(seg1, seg2)); });
        if constexpr (!is_gen)
          obj.def("cross",
                  [](const Derived& seg1, const Gen& seg2) { return Gen(crossProduct(seg1, seg2)); });

        obj.def("cross", [](const Derived& seg1, const Vector3<double>& seg2) {
          return Gen(crossProduct(seg1, Constant<-1, 3>(seg1.IRows(), seg2)));
        });


        obj.def("cross",
                [](const Derived& seg1, const Derived& seg2) { return Gen(crossProduct(seg1, seg2)); });
      }


      if constexpr (OR != 1) {
        if constexpr (!is_seg)
          obj.def("dot", [](const Derived& seg1, const SEG& seg2) { return GenS(dotProduct(seg1, seg2)); });

        if constexpr (!is_gen)
          obj.def("dot", [](const Derived& seg1, const Gen& seg2) { return GenS(dotProduct(seg1, seg2)); });


        ///////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        if constexpr (!is_seg)
          obj.def("cwiseProduct", [](const Derived& seg1, const SEG& seg2) {
            return Gen(CwiseFunctionProduct<Derived, SEG>(seg1, seg2));
          });
        if constexpr (!is_gen)
          obj.def("cwiseProduct", [](const Derived& seg1, const Gen& seg2) {
            return Gen(CwiseFunctionProduct<Derived, Gen>(seg1, seg2));
          });

        ///////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////


        if constexpr (!is_seg)
          obj.def("cwiseQuotient",
                  [](const Derived& seg1, const SEG& seg2) { return Gen(cwiseQuotient(seg1, seg2)); });
        if constexpr (!is_gen)
          obj.def("cwiseQuotient",
                  [](const Derived& seg1, const Gen& seg2) { return Gen(cwiseQuotient(seg1, seg2)); });
      }


      obj.def("dot", [](const Derived& seg1, const Derived& seg2) { return GenS(dotProduct(seg1, seg2)); });
      obj.def("cwiseProduct", [](const Derived& seg1, const Derived& seg2) {
        return BinGen(CwiseFunctionProduct<Derived, Derived>(seg1, seg2));
      });

      obj.def("cwiseQuotient",
              [](const Derived& seg1, const Derived& seg2) { return BinGen(cwiseQuotient(seg1, seg2)); });

      //////////////////////////////////////
      obj.def("dot", [](const Derived& seg1, const Output<double>& seg2) {
        return GenS(dotProduct(seg1, Constant<-1, Base::ORC>(seg1.IRows(), seg2)));
      });

      obj.def("cwiseProduct", [](const Derived& seg1, const Output<double>& seg2) {
        return BinGen(RowScaled<Derived>(seg1, seg2));
      });

      obj.def("cwiseQuotient", [](const Derived& seg1, const Output<double>& seg2) {
        Output<double> seg2i = seg2.cwiseInverse();
        return BinGen(RowScaled<Derived>(seg1, seg2i));
      });
    }

    template<class PYClass>
    static void BinaryOperatorsBuild(PYClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;
      using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

      using ARGS = Arguments<-1>;
      using SEG = Segment<-1, -1, -1>;
      using SEG2 = Segment<-1, 2, -1>;
      using SEG3 = Segment<-1, 3, -1>;
      using SEG4 = Segment<-1, 4, -1>;
      using ELEM = Segment<-1, 1, -1>;


      constexpr bool is_seg = std::is_same<Derived, SEG>::value;
      constexpr bool is_seg2 = std::is_same<Derived, SEG2>::value;
      constexpr bool is_seg3 = std::is_same<Derived, SEG3>::value;

      constexpr bool is_elem = std::is_same<Derived, ELEM>::value;
      constexpr bool is_gen = std::is_same<Derived, Gen>::value;
      constexpr bool is_gens = std::is_same<Derived, GenS>::value;


      constexpr bool is_arglike = Is_Segment<Derived>::value || Is_Arguments<Derived>::value;


      if constexpr (!is_arglike) {


        obj.def("eval", [](const Derived& a, const ELEM& b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived& a, const SEG& b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived& a, const SEG2& b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived& a, const SEG3& b) { return BinGen(a.eval(b)); });
        obj.def("eval", [](const Derived& a, const Gen& b) { return BinGen(a.eval(b)); });


        /////////////////////////////////////////////////////
        obj.def(
            "__call__", [](const Derived& a, const Gen& b) { return BinGen(a.eval(b)); }, py::is_operator());
        obj.def(
            "__call__", [](const Derived& a, const ELEM& b) { return BinGen(a.eval(b)); }, py::is_operator());
        obj.def("__call__", [](const Derived& a, const SEG& b) { return BinGen(a.eval(b)); });
        obj.def(
            "__call__", [](const Derived& a, const SEG2& b) { return BinGen(a.eval(b)); }, py::is_operator());
        obj.def(
            "__call__", [](const Derived& a, const SEG3& b) { return BinGen(a.eval(b)); }, py::is_operator());
        ///////////////////////////////////////////////////////////////
        obj.def("eval", [](const Derived& a, int ir, Eigen::VectorXi v) {
          return BinGen(ParsedInput<Derived, -1, OR>(a, v, ir));
        });
      }

      obj.def(
          "apply",
          [](const Derived& a, const Gen& b) {
            // return Gen(b.eval(a));
            return Gen(NestedFunction<Derived, Gen>(a, b));
          },
          py::is_operator());
      obj.def(
          "apply",
          [](const Derived& a, const GenS& b) {
            // return GenS(b.eval(a));
            return Gen(NestedFunction<Derived, GenS>(a, b));
          },
          py::is_operator());

      obj.def(
          "__add__", [](const Derived& a, const Derived& b) { return BinGen(a + b); }, py::is_operator());
      obj.def(
          "__sub__", [](const Derived& a, const Derived& b) { return BinGen(a - b); }, py::is_operator());


      obj.def(
          "__mul__", [](const Derived& a, const ELEM& b) { return BinGen(a * b); }, py::is_operator());
      obj.def(
          "__mul__", [](const Derived& a, const GenS& b) { return BinGen(a * b); }, py::is_operator());


      obj.def(
          "__truediv__", [](const Derived& a, const ELEM& b) { return BinGen(a / b); }, py::is_operator());

      obj.def(
          "__truediv__", [](const Derived& a, const GenS& b) { return BinGen(a / b); }, py::is_operator());


      if constexpr (is_seg || is_elem || is_seg2 || is_seg3) {  //
        obj.def(
            "__add__",
            [](const Derived& a, const BinGen& b) { return BinGen(TwoFunctionSum<Derived, BinGen>(a, b)); },
            py::is_operator());
        obj.def(
            "__sub__",
            [](const Derived& a, const BinGen& b) {
              return BinGen(FunctionDifference<Derived, BinGen>(a, b));
            },
            py::is_operator());
      }

      if constexpr (OR != 1) {


        obj.def(
            "__rmul__", [](const Derived& a, const ELEM& b) { return BinGen(a * b); }, py::is_operator());

        obj.def(
            "__rmul__", [](const Derived& a, const GenS& b) { return BinGen(a * b); }, py::is_operator());
      }


      ////////////////////////////////////////////////////////////////
    }

    template<class PYClass>
    static void ConditionalOperatorsBuild(PYClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;
      using ELEM = Segment<-1, 1, -1>;
      using GenCon = GenericConditional<IR>;

      if constexpr (OR == 1) {
        obj.def(
            "__lt__", [](const Derived& a, double b) { return GenCon(a < b); }, py::is_operator());
        obj.def(
            "__gt__", [](const Derived& a, double b) { return GenCon(a > b); }, py::is_operator());
        obj.def(
            "__rlt__", [](const Derived& a, double b) { return GenCon(a > b); }, py::is_operator());
        obj.def(
            "__rgt__",
            [](const Derived& a, double b) {
              Vector1<double> tmp;
              tmp[0] = b;
              Constant<IR, 1> bfunc(a.IRows(), tmp);
              return GenCon(a < b);
            },
            py::is_operator());
        obj.def(
            "__le__", [](const Derived& a, double b) { return GenCon(a <= b); }, py::is_operator());
        obj.def(
            "__ge__", [](const Derived& a, double b) { return GenCon(a >= b); }, py::is_operator());

        obj.def(
            "__lt__", [](const Derived& a, const Derived& b) { return GenCon(a < b); }, py::is_operator());
        obj.def(
            "__gt__", [](const Derived& a, const Derived& b) { return GenCon(a > b); }, py::is_operator());
        obj.def(
            "__le__", [](const Derived& a, const Derived& b) { return GenCon(a <= b); }, py::is_operator());
        obj.def(
            "__ge__", [](const Derived& a, const Derived& b) { return GenCon(a >= b); }, py::is_operator());

        if constexpr (std::is_same<Derived, ELEM>::value) {
          obj.def(
              "__lt__", [](const Derived& a, const GenS& b) { return GenCon(a < b); }, py::is_operator());
          obj.def(
              "__gt__", [](const Derived& a, const GenS& b) { return GenCon(a > b); }, py::is_operator());
          obj.def(
              "__le__", [](const Derived& a, const GenS& b) { return GenCon(a <= b); }, py::is_operator());
          obj.def(
              "__ge__", [](const Derived& a, const GenS& b) { return GenCon(a >= b); }, py::is_operator());
        }
      }
    }
  };

}  // namespace ASSET

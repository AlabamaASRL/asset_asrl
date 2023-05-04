/*
File Name: DenseFunctionSpecs.h

File Description: Defines the rubber_types compatible type erasure spec (DenseFunctionSpec)
for the methods of Dense ASSET vector functions. This encompasses the primary compute and
derivative methods with both double and super-scalar arguments as well as selected jacobian operations
on double valued matrices that have been shown to improve performance under certain circumstances.
This Spec is used to define the type erased GenericFunction type.


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

#include "AssigmentTypes.h"
#include "DetectSuperScalar.h"
#include "pch.h"
namespace ASSET {

  template<int IR, int OR>
  struct DenseFunctionSpec {
    template<class Scalar>
    using Output = Eigen::Matrix<Scalar, OR, 1>;
    template<class Scalar>
    using Input = Eigen::Matrix<Scalar, IR, 1>;
    template<class Scalar>
    using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
    template<class Scalar>
    using Hessian = Eigen::Matrix<Scalar, IR, IR>;

    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using VectorBaseRef = Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using ConstMatrixBaseRef = const Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using ConstEigenBaseRef = const Eigen::EigenBase<Scalar>&;
    template<class Scalar>
    using ConstDiagonalBaseRef = const Eigen::DiagonalBase<Scalar>&;

    template<class Scalar>
    using MatrixBaseRef = Eigen::MatrixBase<Scalar>&;


    using InType = Eigen::Ref<const Input<double>>;
    using OutType = Eigen::Ref<Output<double>>;

    using JacType =
        typename std::conditional<OR == 1,
                                  Eigen::Ref<Eigen::Matrix<double, -1, IR>, 0, Eigen::Stride<-1, -1>>,
                                  Eigen::Ref<Jacobian<double>>>::type;

    using AdjGradType = Eigen::Ref<Input<double>>;
    using AdjVarType = Eigen::Ref<const Output<double>>;
    using AdjHessType = Eigen::Ref<Hessian<double>>;

    using SuperInType = Eigen::Ref<const Input<ASSET::DefaultSuperScalar>>;
    using SuperOutType = Eigen::Ref<Output<ASSET::DefaultSuperScalar>>;

    using SuperJacType = typename std::conditional<
        OR == 1,
        Eigen::Ref<Eigen::Matrix<ASSET::DefaultSuperScalar, -1, IR>, 0, Eigen::Stride<-1, -1>>,
        Eigen::Ref<Jacobian<ASSET::DefaultSuperScalar>>>::type;

    using SuperAdjGradType = Eigen::Ref<Input<ASSET::DefaultSuperScalar>>;
    using SuperAdjVarType = Eigen::Ref<const Output<ASSET::DefaultSuperScalar>>;
    using SuperAdjHessType = Eigen::Ref<Hessian<ASSET::DefaultSuperScalar>>;

    using RightJacTarget = Eigen::Ref<Eigen::Matrix<double, -1, IR>>;
    using LeftJacMatrix = Eigen::Ref<const Eigen::Matrix<double, -1, OR>>;
    using LeftDiagMatrix = Eigen::DiagonalMatrix<double, OR>;

    using SuperLeftJacMatrix = Eigen::Ref<const Eigen::Matrix<ASSET::DefaultSuperScalar, -1, OR>>;
    using SuperRightJacTarget = Eigen::Ref<Eigen::Matrix<ASSET::DefaultSuperScalar, -1, IR>>;


    struct Concept {  // abstract base class for model.
      virtual ~Concept() = default;
      // Your (internal) interface goes here.

      // virtual void rpt(Eigen::VectorXd x, int n) const = 0;
      virtual DomainMatrix input_domain() const = 0;
      virtual bool is_linear() const = 0;
      virtual void enable_vectorization(bool b) const = 0;

      virtual void compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const = 0;

      virtual void compute(ConstVectorBaseRef<SuperInType> x, ConstVectorBaseRef<SuperOutType> fx_) const = 0;

      virtual void compute_jacobian(ConstVectorBaseRef<InType> x,
                                    ConstVectorBaseRef<OutType> fx_,
                                    ConstMatrixBaseRef<JacType> jx_) const = 0;

      virtual void compute_jacobian(ConstVectorBaseRef<SuperInType> x,
                                    ConstVectorBaseRef<SuperOutType> fx_,
                                    ConstMatrixBaseRef<SuperJacType> jx_) const = 0;

      virtual void compute_jacobian_adjointgradient_adjointhessian(
          ConstVectorBaseRef<InType> x,
          ConstVectorBaseRef<OutType> fx_,
          ConstMatrixBaseRef<JacType> jx_,
          ConstVectorBaseRef<AdjGradType> adjgrad_,
          ConstMatrixBaseRef<AdjHessType> adjhess_,
          ConstVectorBaseRef<AdjVarType> adjvars) const = 0;

      virtual void compute_jacobian_adjointgradient_adjointhessian(
          ConstVectorBaseRef<SuperInType> x,
          ConstVectorBaseRef<SuperOutType> fx_,
          ConstMatrixBaseRef<SuperJacType> jx_,
          ConstVectorBaseRef<SuperAdjGradType> adjgrad_,
          ConstMatrixBaseRef<SuperAdjHessType> adjhess_,
          ConstVectorBaseRef<SuperAdjVarType> adjvars) const = 0;

      virtual void scale_jacobian(ConstMatrixBaseRef<JacType> target_, double s) const = 0;
      /*virtual void scale_gradient(ConstMatrixBaseRef<AdjGradType> target_,
                                  double s) const = 0;
      virtual void scale_hessian(ConstMatrixBaseRef<AdjHessType> target_,
                                 double s) const = 0;*/

      virtual void accumulate_jacobian(ConstMatrixBaseRef<JacType> target_,
                                       ConstMatrixBaseRef<JacType> right,
                                       DirectAssignment assign) const = 0;
      virtual void accumulate_jacobian(ConstMatrixBaseRef<JacType> target_,
                                       ConstMatrixBaseRef<JacType> right,
                                       PlusEqualsAssignment assign) const = 0;
      virtual void accumulate_jacobian(ConstMatrixBaseRef<JacType> target_,
                                       ConstMatrixBaseRef<JacType> right,
                                       MinusEqualsAssignment assign) const = 0;

      /* virtual void accumulate_gradient(ConstMatrixBaseRef<AdjGradType> target_,
                                        ConstMatrixBaseRef<AdjGradType> right,
                                        DirectAssignment assign) const = 0;
       virtual void accumulate_gradient(ConstMatrixBaseRef<AdjGradType> target_,
                                        ConstMatrixBaseRef<AdjGradType> right,
                                        PlusEqualsAssignment assign) const = 0;
       virtual void accumulate_gradient(ConstMatrixBaseRef<AdjGradType> target_,
                                        ConstMatrixBaseRef<AdjGradType> right,
                                        MinusEqualsAssignment assign) const = 0;

       virtual void accumulate_hessian(ConstMatrixBaseRef<AdjHessType> target_,
                                       ConstMatrixBaseRef<AdjHessType> right,
                                       DirectAssignment assign) const = 0;
       virtual void accumulate_hessian(ConstMatrixBaseRef<AdjHessType> target_,
                                       ConstMatrixBaseRef<AdjHessType> right,
                                       PlusEqualsAssignment assign) const = 0;
       virtual void accumulate_hessian(ConstMatrixBaseRef<AdjHessType> target_,
                                       ConstMatrixBaseRef<AdjHessType> right,
                                       MinusEqualsAssignment assign) const = 0;*/

      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          DirectAssignment assign,
                                          std::bool_constant<true> aliased) const = 0;
      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          PlusEqualsAssignment assign,
                                          std::bool_constant<true> aliased) const = 0;
      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          MinusEqualsAssignment assign,
                                          std::bool_constant<true> aliased) const = 0;

      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          DirectAssignment assign,
                                          std::bool_constant<false> aliased) const = 0;
      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          PlusEqualsAssignment assign,
                                          std::bool_constant<false> aliased) const = 0;
      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          MinusEqualsAssignment assign,
                                          std::bool_constant<false> aliased) const = 0;

      /*virtual void right_jacobian_product(
          ConstMatrixBaseRef<RightJacTarget> target_,
          ConstEigenBaseRef<LeftDiagMatrix> left,
          ConstEigenBaseRef<JacType> right, DirectAssignment assign,
          std::bool_constant<true> aliased) const = 0;
      virtual void right_jacobian_product(
          ConstMatrixBaseRef<RightJacTarget> target_,
          ConstEigenBaseRef<LeftDiagMatrix> left,
          ConstEigenBaseRef<JacType> right, PlusEqualsAssignment assign,
          std::bool_constant<true> aliased) const = 0;
      virtual void right_jacobian_product(
          ConstMatrixBaseRef<RightJacTarget> target_,
          ConstEigenBaseRef<LeftDiagMatrix> left,
          ConstEigenBaseRef<JacType> right, MinusEqualsAssignment assign,
          std::bool_constant<true> aliased) const = 0;*/
    };
    template<class Holder>
    struct Model : public Holder, public virtual Concept {
      using Holder::Holder;
      // Pass through to encapsulated value.

      /*virtual void rpt(Eigen::VectorXd x, int n) const override {
        return rubber_types::model_get(this).rpt(x, n);
      }*/
      virtual DomainMatrix input_domain() const override {
        return rubber_types::model_get(this).input_domain();
      }
      virtual bool is_linear() const override {
        return rubber_types::model_get(this).is_linear();
      }
      virtual void enable_vectorization(bool b) const override {
        return rubber_types::model_get(this).enable_vectorization(b);
      }
      virtual void compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const override {
        return rubber_types::model_get(this).compute(x, fx_);
      }

      virtual void compute(ConstVectorBaseRef<SuperInType> x,
                           ConstVectorBaseRef<SuperOutType> fx_) const override {
        return rubber_types::model_get(this).compute(x, fx_);
      }

      virtual void compute_jacobian(ConstVectorBaseRef<InType> x,
                                    ConstVectorBaseRef<OutType> fx_,
                                    ConstMatrixBaseRef<JacType> jx_) const override {
        return rubber_types::model_get(this).compute_jacobian(x, fx_, jx_);
      }

      virtual void compute_jacobian(ConstVectorBaseRef<SuperInType> x,
                                    ConstVectorBaseRef<SuperOutType> fx_,
                                    ConstMatrixBaseRef<SuperJacType> jx_) const override {
        return rubber_types::model_get(this).compute_jacobian(x, fx_, jx_);
      }

      virtual void compute_jacobian_adjointgradient_adjointhessian(
          ConstVectorBaseRef<InType> x,
          ConstVectorBaseRef<OutType> fx_,
          ConstMatrixBaseRef<JacType> jx_,
          ConstVectorBaseRef<AdjGradType> adjgrad_,
          ConstMatrixBaseRef<AdjHessType> adjhess_,
          ConstVectorBaseRef<AdjVarType> adjvars) const override {
        return rubber_types::model_get(this).compute_jacobian_adjointgradient_adjointhessian(
            x, fx_, jx_, adjgrad_, adjhess_, adjvars);
      }

      virtual void compute_jacobian_adjointgradient_adjointhessian(
          ConstVectorBaseRef<SuperInType> x,
          ConstVectorBaseRef<SuperOutType> fx_,
          ConstMatrixBaseRef<SuperJacType> jx_,
          ConstVectorBaseRef<SuperAdjGradType> adjgrad_,
          ConstMatrixBaseRef<SuperAdjHessType> adjhess_,
          ConstVectorBaseRef<SuperAdjVarType> adjvars) const override {
        return rubber_types::model_get(this).compute_jacobian_adjointgradient_adjointhessian(
            x, fx_, jx_, adjgrad_, adjhess_, adjvars);
      }

      virtual void scale_jacobian(ConstMatrixBaseRef<JacType> target_, double s) const override {
        return rubber_types::model_get(this).scale_jacobian(target_, s);
      }
      /*virtual void scale_gradient(ConstMatrixBaseRef<AdjGradType> target_,
                                  double s) const override {
        return rubber_types::model_get(this).scale_gradient(target_, s);
      }
      virtual void scale_hessian(ConstMatrixBaseRef<AdjHessType> target_,
                                 double s) const override {
        return rubber_types::model_get(this).scale_hessian(target_, s);
      }*/

      virtual void accumulate_jacobian(ConstMatrixBaseRef<JacType> target_,
                                       ConstMatrixBaseRef<JacType> right,
                                       DirectAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_jacobian(target_, right, assign);
      }
      virtual void accumulate_jacobian(ConstMatrixBaseRef<JacType> target_,
                                       ConstMatrixBaseRef<JacType> right,
                                       PlusEqualsAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_jacobian(target_, right, assign);
      }
      virtual void accumulate_jacobian(ConstMatrixBaseRef<JacType> target_,
                                       ConstMatrixBaseRef<JacType> right,
                                       MinusEqualsAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_jacobian(target_, right, assign);
      }

      /*virtual void accumulate_gradient(ConstMatrixBaseRef<AdjGradType> target_,
                                       ConstMatrixBaseRef<AdjGradType> right,
                                       DirectAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_gradient(target_, right,
                                                                 assign);
      }
      virtual void accumulate_gradient(
          ConstMatrixBaseRef<AdjGradType> target_,
          ConstMatrixBaseRef<AdjGradType> right,
          PlusEqualsAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_gradient(target_, right,
                                                                 assign);
      }
      virtual void accumulate_gradient(
          ConstMatrixBaseRef<AdjGradType> target_,
          ConstMatrixBaseRef<AdjGradType> right,
          MinusEqualsAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_gradient(target_, right,
                                                                 assign);
      }

      virtual void accumulate_hessian(ConstMatrixBaseRef<AdjHessType> target_,
                                      ConstMatrixBaseRef<AdjHessType> right,
                                      DirectAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_hessian(target_, right,
                                                                assign);
      }
      virtual void accumulate_hessian(
          ConstMatrixBaseRef<AdjHessType> target_,
          ConstMatrixBaseRef<AdjHessType> right,
          PlusEqualsAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_hessian(target_, right,
                                                                assign);
      }
      virtual void accumulate_hessian(
          ConstMatrixBaseRef<AdjHessType> target_,
          ConstMatrixBaseRef<AdjHessType> right,
          MinusEqualsAssignment assign) const override {
        return rubber_types::model_get(this).accumulate_hessian(target_, right,
                                                                assign);
      }*/

      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          DirectAssignment assign,
                                          std::bool_constant<true> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(target_, left, right, assign, aliased);
      }
      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          PlusEqualsAssignment assign,
                                          std::bool_constant<true> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(target_, left, right, assign, aliased);
      }
      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          MinusEqualsAssignment assign,
                                          std::bool_constant<true> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(target_, left, right, assign, aliased);
      }

      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          DirectAssignment assign,
                                          std::bool_constant<false> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(target_, left, right, assign, aliased);
      }
      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          PlusEqualsAssignment assign,
                                          std::bool_constant<false> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(target_, left, right, assign, aliased);
      }
      virtual void right_jacobian_product(ConstMatrixBaseRef<RightJacTarget> target_,
                                          ConstEigenBaseRef<LeftJacMatrix> left,
                                          ConstEigenBaseRef<JacType> right,
                                          MinusEqualsAssignment assign,
                                          std::bool_constant<false> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(target_, left, right, assign, aliased);
      }

      /*virtual void right_jacobian_product(
          ConstMatrixBaseRef<RightJacTarget> target_,
          ConstEigenBaseRef<LeftDiagMatrix> left,
          ConstEigenBaseRef<JacType> right, DirectAssignment assign,
          std::bool_constant<true> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(
            target_, left, right, assign, aliased);
      }
      virtual void right_jacobian_product(
          ConstMatrixBaseRef<RightJacTarget> target_,
          ConstEigenBaseRef<LeftDiagMatrix> left,
          ConstEigenBaseRef<JacType> right, PlusEqualsAssignment assign,
          std::bool_constant<true> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(
            target_, left, right, assign, aliased);
      }
      virtual void right_jacobian_product(
          ConstMatrixBaseRef<RightJacTarget> target_,
          ConstEigenBaseRef<LeftDiagMatrix> left,
          ConstEigenBaseRef<JacType> right, MinusEqualsAssignment assign,
          std::bool_constant<true> aliased) const override {
        return rubber_types::model_get(this).right_jacobian_product(
            target_, left, right, assign, aliased);
      }*/
    };
    template<class Container>
    struct ExternalInterface : public Container {
      using Container_ = Container;
      using Container_::Container_;


      /*
      Retemplating and then explicitly casting inputs to Eigen Ref Types. This
      allows the most flexibilty with input arg types
      */

      /*void rpt(Eigen::VectorXd x, int n) const {
        return rubber_types::interface_get(this).rpt(x, n);
      }*/

      DomainMatrix input_domain() const {
        return rubber_types::interface_get(this).input_domain();
      }
      bool is_linear() const {
        return rubber_types::interface_get(this).is_linear();
      }
      void enable_vectorization(bool b) const {
        return rubber_types::interface_get(this).enable_vectorization(b);
      }
      template<class InTypeTT, class OutTypeTT>
      void compute(ConstVectorBaseRef<InTypeTT> x, ConstVectorBaseRef<OutTypeTT> fx_) const {
        typedef typename InTypeTT::Scalar Scalar;

        if constexpr (std::is_same<Scalar, double>::value) {
          InType xt(x.derived());
          OutType fxt(fx_.const_cast_derived());
          return rubber_types::interface_get(this).compute(xt, fxt);
        } else {
          SuperInType xt(x.derived());
          SuperOutType fxt(fx_.const_cast_derived());
          return rubber_types::interface_get(this).compute(xt, fxt);
        }
      }
      template<class InTypeTT, class OutTypeTT, class JacTypeTT>
      void compute_jacobian(ConstVectorBaseRef<InTypeTT> x,
                            ConstVectorBaseRef<OutTypeTT> fx_,
                            ConstMatrixBaseRef<JacTypeTT> jx_) const {
        typedef typename InTypeTT::Scalar Scalar;

        if constexpr (std::is_same<Scalar, double>::value) {
          InType xt(x.derived());
          OutType fxt(fx_.const_cast_derived());
          JacType jxt(jx_.const_cast_derived());
          rubber_types::interface_get(this).compute_jacobian(xt, fxt, jxt);
        } else {
          SuperInType xt(x.derived());
          SuperOutType fxt(fx_.const_cast_derived());
          SuperJacType jxt(jx_.const_cast_derived());
          rubber_types::interface_get(this).compute_jacobian(xt, fxt, jxt);
        }
      }
      template<class InTypeTT,
               class OutTypeTT,
               class JacTypeTT,
               class AdjGradTypeTT,
               class AdjHessTypeTT,
               class AdjVarTypeTT>
      void compute_jacobian_adjointgradient_adjointhessian(ConstVectorBaseRef<InTypeTT> x,
                                                           ConstVectorBaseRef<OutTypeTT> fx_,
                                                           ConstMatrixBaseRef<JacTypeTT> jx_,
                                                           ConstVectorBaseRef<AdjGradTypeTT> adjgrad_,
                                                           ConstMatrixBaseRef<AdjHessTypeTT> adjhess_,
                                                           ConstVectorBaseRef<AdjVarTypeTT> adjvars) const {
        typedef typename InTypeTT::Scalar Scalar;

        if constexpr (std::is_same<Scalar, double>::value) {
          InType xt(x.derived());
          OutType fxt(fx_.const_cast_derived());
          JacType jxt(jx_.const_cast_derived());
          AdjGradType adjgradt(adjgrad_.const_cast_derived());
          AdjHessType adjhesst(adjhess_.const_cast_derived());
          AdjVarType adjvarst(adjvars.derived());

          return rubber_types::interface_get(this).compute_jacobian_adjointgradient_adjointhessian(
              xt, fxt, jxt, adjgradt, adjhesst, adjvarst);
        } else {
          SuperInType xt(x.derived());
          SuperOutType fxt(fx_.const_cast_derived());
          SuperJacType jxt(jx_.const_cast_derived());
          SuperAdjGradType adjgradt(adjgrad_.const_cast_derived());
          SuperAdjHessType adjhesst(adjhess_.const_cast_derived());
          SuperAdjVarType adjvarst(adjvars.derived());
          return rubber_types::interface_get(this).compute_jacobian_adjointgradient_adjointhessian(
              xt, fxt, jxt, adjgradt, adjhesst, adjvarst);
        }
      }

      template<class JacTypeTT>
      void scale_jacobian(ConstMatrixBaseRef<JacTypeTT> target_, double s) const {
        JacType jxt(target_.const_cast_derived());
        return rubber_types::interface_get(this).scale_jacobian(jxt, s);
      }
      /*template <class AdjGradTypeTT>
      void scale_gradient(ConstMatrixBaseRef<AdjGradTypeTT> target_,
                          double s) const {
        AdjGradType adjgradt(target_.const_cast_derived());
        return rubber_types::interface_get(this).scale_gradient(adjgradt, s);
      }
      template <class AdjHessTypeTT>
      void scale_hessian(ConstMatrixBaseRef<AdjHessTypeTT> target_,
                         double s) const {
        AdjHessType adjhesst(target_.const_cast_derived());
        return rubber_types::interface_get(this).scale_hessian(adjhesst, s);
      }*/

      template<class TargetTT, class JacTypeTT, class Assignment>
      void accumulate_jacobian(ConstMatrixBaseRef<TargetTT> target_,
                               ConstMatrixBaseRef<JacTypeTT> right,
                               Assignment assign) const {
        JacType jtarg(target_.const_cast_derived());
        JacType jright(right.const_cast_derived());
        return rubber_types::interface_get(this).accumulate_jacobian(jtarg, jright, assign);
      }
      /* template <class TargetTT, class AdjGradTypeTT, class Assignment>
       void accumulate_gradient(ConstMatrixBaseRef<TargetTT> target_,
                                ConstMatrixBaseRef<AdjGradTypeTT> right,
                                Assignment assign) const {
         AdjGradType gtarg(target_.const_cast_derived());
         AdjGradType gright(right.const_cast_derived());
         return rubber_types::interface_get(this).accumulate_gradient(
             gtarg, gright, assign);
       }
       template <class TargetTT, class AdjHessTypeTT, class Assignment>
       void accumulate_hessian(ConstMatrixBaseRef<TargetTT> target_,
                               ConstMatrixBaseRef<AdjHessTypeTT> right,
                               Assignment assign) const {
         AdjHessType htarg(target_.const_cast_derived());
         AdjHessType hright(right.const_cast_derived());
         return rubber_types::interface_get(this).accumulate_hessian(htarg, hright,
                                                                     assign);
       }*/

      template<class TargetTT, class LeftTT, class RightTT, class Assignment, bool Aliased>
      void right_jacobian_product(ConstMatrixBaseRef<TargetTT> target_,
                                  ConstMatrixBaseRef<LeftTT> left,
                                  ConstMatrixBaseRef<RightTT> right,
                                  Assignment assign,
                                  std::bool_constant<Aliased> aliased) const {
        RightJacTarget target(target_.const_cast_derived());
        LeftJacMatrix leftt(left.derived());
        JacType rightt(right.const_cast_derived());
        return rubber_types::interface_get(this).right_jacobian_product(
            target, leftt, rightt, assign, aliased);
      }

      /* template <class TargetTT, class LeftTT, class RightTT, class Assignment,
                 bool Aliased>
       void right_jacobian_product(ConstMatrixBaseRef<TargetTT> target_,
                                   ConstDiagonalBaseRef<LeftTT> left,
                                   ConstMatrixBaseRef<RightTT> right,
                                   Assignment assign,
                                   std::bool_constant<Aliased> aliased) const {
         RightJacTarget target(target_.const_cast_derived());
         LeftDiagMatrix leftt = left;
         JacType rightt(right.const_cast_derived());
         return rubber_types::interface_get(this).right_jacobian_product(
             target, leftt, rightt, assign, std::bool_constant<true>());
       }*/

      /// Input Rows at Compile Time (-1 if Dynamic)
      static const int IRC = IR;
      /// Output Rows at Compile Time (-1 if Dynamic)
      static const int ORC = OR;
      static const bool InputIsDynamic = (IR < 0);
      static const bool OutputIsDynamic = (OR < 0);
      static const bool JacobianIsDynamic = (IR < 0 || OR < 0);
      static const bool FullyDynamic = (IR < 0 && OR < 0);
      static const bool IsLinearFunction = false;
      static const bool IsGenericFunction = true;
      static const bool IsVectorizable = true;
      static const bool IsConditional = false;

      using INPUT_DOMAIN = SingleDomain<IR, 0, IR>;

      /////////////////////////////////////////////////////////////
      template<class Scalar>
      using Output = Eigen::Matrix<Scalar, OR, 1>;
      template<class Scalar>
      using Input = Eigen::Matrix<Scalar, IR, 1>;
      template<class Scalar>
      using Gradient = Eigen::Matrix<Scalar, IR, 1>;
      template<class Scalar>
      using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
      template<class Scalar>
      using Hessian = Eigen::Matrix<Scalar, IR, IR>;
      //////////////////////////////////////////////////////////////
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
      //////////////////////////////////////////////////////////////
    };
  };

}  // namespace ASSET

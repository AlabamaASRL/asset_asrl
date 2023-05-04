#pragma once

#include "VectorFunction.h"

namespace ASSET {


  template<class OuterFunc, class InnerFunc1, class... InnerFuncs>
  struct NestedCallAndAppendChain2
      : VectorFunction<NestedCallAndAppendChain2<OuterFunc, InnerFunc1, InnerFuncs...>,
                       InnerFunc1::IRC,
                       OuterFunc::ORC> {
    using Base = VectorFunction<NestedCallAndAppendChain2<OuterFunc, InnerFunc1, InnerFuncs...>,
                                InnerFunc1::IRC,
                                OuterFunc::ORC>;
    using Base::compute;

    DENSE_FUNCTION_BASE_TYPES(Base);

    OuterFunc outer_func;
    InnerFunc1 inner_func1;

    std::tuple<InnerFuncs...> inner_funcs;
    SUB_FUNCTION_IO_TYPES(OuterFunc);
    SUB_FUNCTION_IO_TYPES(InnerFunc1);

    using INPUT_DOMAIN = typename InnerFunc1::INPUT_DOMAIN;

    static const bool IsVectorizable = InnerFunc1::IsVectorizable && OuterFunc::IsVectorizable;

    static const int SizeInnerFuncs = sizeof...(InnerFuncs);

    NestedCallAndAppendChain2() {
    }

    NestedCallAndAppendChain2(OuterFunc outer_func, std::tuple<InnerFunc1, InnerFuncs...> inner_funct)
        : outer_func(std::move(outer_func)) {
      this->inner_func1 = std::get<0>(inner_funct);
      ASSET::constexpr_for_loop(
          std::integral_constant<int, 0>(),
          std::integral_constant<int, sizeof...(InnerFuncs)>(),
          [&](auto i) { std::get<i.value>(this->inner_funcs) = std::get<i.value + 1>(inner_funct); });


      this->setIORows(this->inner_func1.IRows(), this->outer_func.ORows());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;

      auto Impl = [&](auto& xchain) {
        xchain.template head<Base::IRC>(this->IRows()) = x;
        this->inner_func1.compute(
            x, xchain.template segment<InnerFunc1::ORC>(this->IRows(), this->inner_func1.ORows()));
        ASSET::tuple_for_each(this->inner_funcs, [&](const auto& func_i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
          func_i.compute(xchain.template head<FTtype::IRC>(func_i.IRows()),
                         xchain.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()));
        });
        this->outer_func.compute(xchain, fx_);
      };
      MemoryManager::allocate_run(
          this->outer_func.IRows(), Impl, TempSpec<OuterFunc_Input<Scalar>>(this->outer_func.IRows(), 1));
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      // VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      // OuterFunc_Input<Scalar> xchain(this->outer_func.IRows());
      // InnerFunc1_jacobian<Scalar> jx1;
      // std::tuple<typename InnerFuncs::template Jacobian<Scalar>...> jxi;
      // OuterFunc_jacobian<Scalar> jxO;


      auto Impl = [&](auto& xchain, auto& jx1, auto& jxi, auto& jxO) {
        xchain.template head<Base::IRC>(this->IRows()) = x;
        this->inner_func1.compute_jacobian(
            x, xchain.template segment<InnerFunc1::ORC>(this->IRows(), this->inner_func1.ORows()), jx1);


        ASSET::tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
          func_i.compute_jacobian(xchain.template head<FTtype::IRC>(func_i.IRows()),
                                  xchain.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()),
                                  std::get<i.value>(jxi));
        });

        this->outer_func.compute_jacobian(xchain, fx_, jxO);

        ASSET::reverse_tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
          func_i.right_jacobian_product(jxO.template leftCols<FTtype::IRC>(func_i.IRows()),
                                        jxO.template middleCols<FTtype::ORC>(func_i.IRows(), func_i.ORows()),
                                        std::get<i.value>(jxi),
                                        PlusEqualsAssignment(),
                                        std::bool_constant<false>());
        });

        this->inner_func1.right_jacobian_product(
            jxO.template leftCols<InnerFunc1::IRC>(this->inner_func1.IRows()),
            jxO.template middleCols<InnerFunc1::ORC>(this->inner_func1.IRows(), this->inner_func1.ORows()),
            jx1,
            PlusEqualsAssignment(),
            std::bool_constant<false>());

        jx.template leftCols<Base::IRC>(this->IRows()) = jxO.template leftCols<Base::IRC>(this->IRows());
      };


      auto make_temp_tuple = [&](auto f) {
        auto app = [&](const auto&... func_i) { return std::tuple {f(func_i)...}; };
        return std::apply(app, this->inner_funcs);
      };
      auto jis = [&](const auto& func_i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
        using JType = typename FTtype::template Jacobian<Scalar>;
        return TempSpec<JType>(func_i.ORows(), func_i.IRows());
      };
      auto JITemps =
          TupleOfTempSpecs<typename InnerFuncs::template Jacobian<Scalar>...> {make_temp_tuple(jis)};

      MemoryManager::allocate_run(
          this->outer_func.IRows(),
          Impl,
          TempSpec<OuterFunc_Input<Scalar>>(this->outer_func.IRows(), 1),
          TempSpec<InnerFunc1_jacobian<Scalar>>(this->inner_func1.ORows(), this->inner_func1.IRows()),
          JITemps,
          TempSpec<OuterFunc_jacobian<Scalar>>(this->outer_func.ORows(), this->outer_func.IRows())

      );
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
      // VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      // OuterFunc_Input<Scalar> xchain(this->outer_func.IRows());

      // OuterFunc_jacobian<Scalar> jxO;  // = OuterFunc_jacobian<Scalar>::Zero();
      // OuterFunc_hessian<Scalar> hxO;   // = OuterFunc_hessian <Scalar>::Zero();
      // OuterFunc_gradient<Scalar> gxO;  // = OuterFunc_gradient<Scalar>::Zero();

      // InnerFunc1_jacobian<Scalar> jx1;  // = InnerFunc1_jacobian<Scalar>::Zero();
      // InnerFunc1_gradient<Scalar> gx1;  // = InnerFunc1_gradient<Scalar>::Zero();
      // InnerFunc1_hessian<Scalar> hx1;   // = InnerFunc1_hessian<Scalar>::Zero();

      // std::tuple<typename InnerFuncs::template Jacobian<Scalar>...> jxi;
      // std::tuple<typename InnerFuncs::template Hessian<Scalar>...> hxi;
      // std::tuple<typename InnerFuncs::template Gradient<Scalar>...> gxi;

      // Eigen::Matrix<Scalar, OuterFunc::IRC, Base::IRC> j0s;


      // xchain.setZero();
      // xchain.template head<Base::IRC>() = x;

      auto Impl = [&](auto& xchain,
                      auto& jx1,
                      auto& gx1,
                      auto& hx1,
                      auto& jxi,
                      auto& gxi,
                      auto& hxi,
                      auto& jxO,
                      auto& gxO,
                      auto& hxO,
                      auto& j0s) {
        xchain.template head<Base::IRC>(this->IRows()) = x;
        this->inner_func1.compute(
            x, xchain.template segment<InnerFunc1::ORC>(this->IRows(), this->inner_func1.ORows()));

        ASSET::tuple_for_each(this->inner_funcs, [&](const auto& func_i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
          func_i.compute(xchain.template head<FTtype::IRC>(func_i.IRows()),
                         xchain.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()));
        });


        this->outer_func.compute_jacobian_adjointgradient_adjointhessian(xchain, fx_, jxO, gxO, hxO, adjvars);

        ASSET::reverse_tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;


          func_i.compute_jacobian_adjointgradient_adjointhessian(
              xchain.template head<FTtype::IRC>(func_i.IRows()),
              xchain.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()),
              std::get<i.value>(jxi),
              std::get<i.value>(gxi),
              std::get<i.value>(hxi),
              gxO.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()));

          func_i.accumulate_gradient(
              gxO.template head<FTtype::IRC>(func_i.IRows()), std::get<i.value>(gxi), PlusEqualsAssignment());

          func_i.right_jacobian_product(jxO.template leftCols<FTtype::IRC>(func_i.IRows()),
                                        jxO.template middleCols<FTtype::ORC>(func_i.IRows(), func_i.ORows()),
                                        std::get<i.value>(jxi),
                                        PlusEqualsAssignment(),
                                        std::bool_constant<false>());
        });

        ////////////////////////////////////////////////////////////////////

        this->inner_func1.compute_jacobian_adjointgradient_adjointhessian(
            x,
            xchain.template segment<InnerFunc1::ORC>(this->inner_func1.IRows(), this->inner_func1.ORows()),
            jx1,
            adjgrad,
            adjhess,
            gxO.template segment<InnerFunc1::ORC>(this->inner_func1.IRows()));

        this->inner_func1.right_jacobian_product(
            jxO.template leftCols<InnerFunc1::IRC>(this->inner_func1.IRows()),
            jxO.template middleCols<InnerFunc1::ORC>(this->inner_func1.IRows(), this->inner_func1.ORows()),
            jx1,
            PlusEqualsAssignment(),
            std::bool_constant<false>());

        jx.template leftCols<Base::IRC>(this->IRows()) = jxO.template leftCols<Base::IRC>(this->IRows());
        adjgrad += gxO.template head<InnerFunc1::IRC>(this->inner_func1.IRows());

        /////////////////////

        //////////////////////

        j0s.template topRows<Base::IRC>(this->IRows()).setIdentity();
        j0s.template middleRows<InnerFunc1::ORC>(this->inner_func1.IRows(), this->inner_func1.ORows()) = jx1;

        ASSET::tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;

          constexpr int Ev = SZ_DIFF<FTtype::IRC, Base::IRC>::value;  // FTtype::IRC - Base::IRC
          const int ev = func_i.IRows() - this->IRows();

          j0s.template middleRows<FTtype::ORC>(func_i.IRows(), func_i.ORows()) =
              std::get<i.value>(jxi).template leftCols<Base::IRC>(this->IRows())
              + std::get<i.value>(jxi).template rightCols<Ev>(func_i.IRows() - this->IRows())
                    * j0s.template middleRows<Ev>(this->IRows(), func_i.IRows() - this->IRows());
        });

        ASSET::tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;

          func_i.accumulate_hessian(
              hxO.template topLeftCorner<FTtype::IRC, FTtype::IRC>(func_i.IRows(), func_i.IRows()),
              std::get<i.value>(hxi),
              PlusEqualsAssignment());
        });


        adjhess.template topLeftCorner<Base::IRC, Base::IRC>(this->IRows(), this->IRows()) +=
            hxO.template topLeftCorner<Base::IRC, Base::IRC>(this->IRows(), this->IRows());

        constexpr int Ev = SZ_DIFF<OuterFunc::IRC, Base::IRC>::value;  // OuterFunc::IRC - Base::IRC;
        const int ev = this->outer_func.IRows() - this->IRows();
        hxO.template leftCols<Base::IRC>(this->IRows()).noalias() =
            hxO.template rightCols<Ev>(ev) * j0s.template bottomRows<Ev>(ev);

        adjhess.template topLeftCorner<Base::IRC, Base::IRC>(this->IRows(), this->IRows()) +=
            hxO.template topLeftCorner<Base::IRC, Base::IRC>(this->IRows(), this->IRows())
            + hxO.template topLeftCorner<Base::IRC, Base::IRC>(this->IRows(), this->IRows()).transpose();

        adjhess.noalias() += j0s.template bottomRows<Ev>(ev).transpose()
                             * hxO.template leftCols<Base::IRC>(this->IRows()).template bottomRows<Ev>(ev);
      };


      auto make_temp_tuple = [&](auto f) {
        auto app = [&](const auto&... func_i) { return std::tuple {f(func_i)...}; };
        return std::apply(app, this->inner_funcs);
      };

      auto jis = [&](const auto& func_i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
        using JType = typename FTtype::template Jacobian<Scalar>;
        return TempSpec<JType>(func_i.ORows(), func_i.IRows());
      };
      auto gis = [&](const auto& func_i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
        using GType = typename FTtype::template Gradient<Scalar>;
        return TempSpec<GType>(func_i.IRows(), 1);
      };
      auto his = [&](const auto& func_i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
        using HType = typename FTtype::template Hessian<Scalar>;
        return TempSpec<HType>(func_i.IRows(), func_i.IRows());
      };


      auto JITemps =
          TupleOfTempSpecs<typename InnerFuncs::template Jacobian<Scalar>...> {make_temp_tuple(jis)};
      auto GITemps =
          TupleOfTempSpecs<typename InnerFuncs::template Gradient<Scalar>...> {make_temp_tuple(gis)};
      auto HITemps =
          TupleOfTempSpecs<typename InnerFuncs::template Hessian<Scalar>...> {make_temp_tuple(his)};


      MemoryManager::allocate_run(
          this->outer_func.IRows(),
          Impl,
          TempSpec<OuterFunc_Input<Scalar>>(this->outer_func.IRows(), 1),
          TempSpec<InnerFunc1_jacobian<Scalar>>(this->inner_func1.ORows(), this->inner_func1.IRows()),
          TempSpec<InnerFunc1_gradient<Scalar>>(this->inner_func1.IRows(), 1),
          TempSpec<InnerFunc1_hessian<Scalar>>(this->inner_func1.IRows(), this->inner_func1.IRows()),
          JITemps,
          GITemps,
          HITemps,
          TempSpec<OuterFunc_jacobian<Scalar>>(this->outer_func.ORows(), this->outer_func.IRows()),
          TempSpec<OuterFunc_gradient<Scalar>>(this->outer_func.IRows(), 1),
          TempSpec<OuterFunc_hessian<Scalar>>(this->outer_func.IRows(), this->outer_func.IRows()),
          TempSpec<Eigen::Matrix<Scalar, OuterFunc::IRC, Base::IRC>>(this->outer_func.IRows(),
                                                                     this->IRows()));
    }
  };


  template<class OuterFunc, class InnerFunc1, class... InnerFuncs>
  struct NestedCallAndAppendChain
      : VectorFunction<NestedCallAndAppendChain<OuterFunc, InnerFunc1, InnerFuncs...>,
                       InnerFunc1::IRC,
                       OuterFunc::ORC> {
    using Base = VectorFunction<NestedCallAndAppendChain<OuterFunc, InnerFunc1, InnerFuncs...>,
                                InnerFunc1::IRC,
                                OuterFunc::ORC>;
    using Base::compute;

    DENSE_FUNCTION_BASE_TYPES(Base);

    OuterFunc outer_func;
    InnerFunc1 inner_func1;

    std::tuple<InnerFuncs...> inner_funcs;
    SUB_FUNCTION_IO_TYPES(OuterFunc);
    SUB_FUNCTION_IO_TYPES(InnerFunc1);

    using INPUT_DOMAIN = typename InnerFunc1::INPUT_DOMAIN;

    static const bool ReverseAlg = false;
    static const int SizeInnerFuncs = sizeof...(InnerFuncs);

    NestedCallAndAppendChain() {
    }
    NestedCallAndAppendChain(OuterFunc outer_func, InnerFunc1 inner_func1, InnerFuncs... inner_funcs)
        : outer_func(std::move(outer_func)),
          inner_func1(std::move(inner_func1)),
          inner_funcs(inner_funcs...) {
    }
    NestedCallAndAppendChain(OuterFunc outer_func,
                             InnerFunc1 inner_func1,
                             std::tuple<InnerFuncs...> inner_funcs)
        : outer_func(std::move(outer_func)), inner_func1(std::move(inner_func1)), inner_funcs(inner_funcs) {
    }

    NestedCallAndAppendChain(OuterFunc outer_func, std::tuple<InnerFunc1, InnerFuncs...> inner_funct)
        : outer_func(std::move(outer_func)) {
      this->inner_func1 = std::get<0>(inner_funct);
      ASSET::constexpr_for_loop(
          std::integral_constant<int, 0>(),
          std::integral_constant<int, sizeof...(InnerFuncs)>(),
          [&](auto i) { std::get<i.value>(this->inner_funcs) = std::get<i.value + 1>(inner_funct); });
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      // VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      OuterFunc_Input<Scalar> xchain(this->outer_func.IRows());
      xchain.setZero();
      xchain.template head<Base::IRC>() = x;
      // int start = this->IRows();

      this->inner_func1.compute(
          x, xchain.template segment<InnerFunc1::ORC>(this->IRows(), this->inner_func1.ORows()));

      ASSET::tuple_for_each(this->inner_funcs, [&](const auto& func_i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
        func_i.compute(xchain.template head<FTtype::IRC>(func_i.IRows()),
                       xchain.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()));
      });

      this->outer_func.compute(xchain, fx_);
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      // VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      OuterFunc_Input<Scalar> xchain(this->outer_func.IRows());

      xchain.setZero();
      xchain.template head<Base::IRC>() = x;

      InnerFunc1_jacobian<Scalar> jx1;  // jx1.setZero();

      this->inner_func1.compute_jacobian(
          x, xchain.template segment<InnerFunc1::ORC>(this->IRows(), this->inner_func1.ORows()), jx1);

      std::tuple<typename InnerFuncs::template Jacobian<Scalar>...> jxi;

      ASSET::tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
        // std::get<i.value>(jxi).setZero();
        func_i.compute_jacobian(xchain.template head<FTtype::IRC>(func_i.IRows()),
                                xchain.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()),
                                std::get<i.value>(jxi));
      });

      OuterFunc_jacobian<Scalar> jxO;

      // Eigen::Matrix<Scalar, -1, -1> jxO;
      // jxO.resize(this->outer_func.ORows(), this->outer_func.IRows());
      // std::vector<OuterFunc_jacobian<Scalar>> jxtt(1);
      // Eigen::Ref< Eigen::Matrix<Scalar,-1,-1>> jxO(jxOt);
      // Eigen::Ref< OuterFunc_jacobian<Scalar>> jxO(jxOt);
      // Eigen::Map<OuterFunc_jacobian<Scalar>> jxO(jxOt.data());
      // Eigen::Map<OuterFunc_jacobian<Scalar>> jxO(jxtt[0].data(),
      // this->outer_func.ORows(), this->outer_func.IRows());

      this->outer_func.compute_jacobian(xchain, fx_, jxO);

      ASSET::reverse_tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
        func_i.right_jacobian_product(jxO.template leftCols<FTtype::IRC>(func_i.IRows()),
                                      jxO.template middleCols<FTtype::ORC>(func_i.IRows(), func_i.ORows()),
                                      std::get<i.value>(jxi),
                                      PlusEqualsAssignment(),
                                      std::bool_constant<false>());
      });

      this->inner_func1.right_jacobian_product(
          jxO.template leftCols<InnerFunc1::IRC>(this->inner_func1.IRows()),
          jxO.template middleCols<InnerFunc1::ORC>(this->inner_func1.IRows(), this->inner_func1.ORows()),
          jx1,
          PlusEqualsAssignment(),
          std::bool_constant<false>());

      jx.template leftCols<Base::IRC>(this->IRows()) = jxO.template leftCols<Base::IRC>(this->IRows());
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
      // VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      OuterFunc_Input<Scalar> xchain(this->outer_func.IRows());
      xchain.setZero();
      xchain.template head<Base::IRC>() = x;
      int start = this->IRows();

      this->inner_func1.compute(
          x, xchain.template segment<InnerFunc1::ORC>(this->IRows(), this->inner_func1.ORows()));

      ASSET::tuple_for_each(this->inner_funcs, [&](const auto& func_i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
        func_i.compute(xchain.template head<FTtype::IRC>(func_i.IRows()),
                       xchain.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()));
      });

      OuterFunc_jacobian<Scalar> jxO;  // = OuterFunc_jacobian<Scalar>::Zero();
      OuterFunc_hessian<Scalar> hxO;   // = OuterFunc_hessian <Scalar>::Zero();
      OuterFunc_gradient<Scalar> gxO;  // = OuterFunc_gradient<Scalar>::Zero();

      InnerFunc1_jacobian<Scalar> jx1;  // = InnerFunc1_jacobian<Scalar>::Zero();
      InnerFunc1_gradient<Scalar> gx1;  // = InnerFunc1_gradient<Scalar>::Zero();
      InnerFunc1_hessian<Scalar> hx1;   // = InnerFunc1_hessian<Scalar>::Zero();

      std::tuple<typename InnerFuncs::template Jacobian<Scalar>...> jxi;
      std::tuple<typename InnerFuncs::template Hessian<Scalar>...> hxi;
      std::tuple<typename InnerFuncs::template Gradient<Scalar>...> gxi;

      this->outer_func.compute_jacobian_adjointgradient_adjointhessian(xchain, fx_, jxO, gxO, hxO, adjvars);

      ASSET::reverse_tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
        using FTtype =
            typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;

        // std::get<i.value>(gxi).setZero();
        // std::get<i.value>(hxi).setZero();
        // std::get<i.value>(jxi).setZero();

        func_i.compute_jacobian_adjointgradient_adjointhessian(
            xchain.template head<FTtype::IRC>(func_i.IRows()),
            xchain.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()),
            std::get<i.value>(jxi),
            std::get<i.value>(gxi),
            std::get<i.value>(hxi),
            gxO.template segment<FTtype::ORC>(func_i.IRows(), func_i.ORows()));

        func_i.accumulate_gradient(
            gxO.template head<FTtype::IRC>(func_i.IRows()), std::get<i.value>(gxi), PlusEqualsAssignment());

        func_i.right_jacobian_product(jxO.template leftCols<FTtype::IRC>(func_i.IRows()),
                                      jxO.template middleCols<FTtype::ORC>(func_i.IRows(), func_i.ORows()),
                                      std::get<i.value>(jxi),
                                      PlusEqualsAssignment(),
                                      std::bool_constant<false>());

        ///////////
        // hxO.template topLeftCorner<FTtype::IRC, FTtype::IRC>(func_i.IRows(),
        // func_i.IRows())
        //	+= std::get<i.value>(jxi).transpose()*hxO.template
        // block<FTtype::ORC,FTtype::ORC>(func_i.IRows(), func_i.IRows()) *
        // std::get<i.value>(jxi);

        if constexpr (ReverseAlg) {
          typename FTtype::template Jacobian<Scalar> jt;

          func_i.right_jacobian_product(
              jt,
              hxO.template block<FTtype::ORC, FTtype::ORC>(func_i.IRows(), func_i.IRows()),
              std::get<i.value>(jxi),
              DirectAssignment(),
              std::bool_constant<false>());

          func_i.right_jacobian_product(
              hxO.template topLeftCorner<FTtype::IRC, FTtype::IRC>(func_i.IRows(), func_i.IRows()),
              jt.transpose(),
              std::get<i.value>(jxi),
              PlusEqualsAssignment(),
              std::bool_constant<false>());

          func_i.accumulate_hessian(
              hxO.template topLeftCorner<FTtype::IRC, FTtype::IRC>(func_i.IRows(), func_i.IRows()),
              std::get<i.value>(hxi),
              PlusEqualsAssignment());

          std::get<i.value>(hxi).setZero();

          func_i.right_jacobian_product(std::get<i.value>(hxi),
                                        hxO.template block<FTtype::IRC, FTtype::ORC>(0, func_i.IRows()),
                                        std::get<i.value>(jxi),
                                        DirectAssignment(),
                                        std::bool_constant<false>());

          hxO.template topLeftCorner<FTtype::IRC, FTtype::IRC>(func_i.IRows(), func_i.IRows()) +=
              std::get<i.value>(hxi) + std::get<i.value>(hxi).transpose();
        }

        /////////////////
      });

      ////////////////////////////////////////////////////////////////////

      this->inner_func1.compute_jacobian_adjointgradient_adjointhessian(
          x,
          xchain.template segment<InnerFunc1::ORC>(this->inner_func1.IRows(), this->inner_func1.ORows()),
          jx1,
          adjgrad,
          adjhess,
          gxO.template segment<InnerFunc1::ORC>(this->inner_func1.IRows()));

      this->inner_func1.right_jacobian_product(
          jxO.template leftCols<InnerFunc1::IRC>(this->inner_func1.IRows()),
          jxO.template middleCols<InnerFunc1::ORC>(this->inner_func1.IRows(), this->inner_func1.ORows()),
          jx1,
          PlusEqualsAssignment(),
          std::bool_constant<false>());

      jx.template leftCols<Base::IRC>(this->IRows()) = jxO.template leftCols<Base::IRC>(this->IRows());
      adjgrad += gxO.template head<InnerFunc1::IRC>(this->inner_func1.IRows());

      ///////////////
      if constexpr (ReverseAlg) {
        //  hxO.template topLeftCorner<InnerFunc1::IRC, InnerFunc1::IRC>(
        //      inner_func1.IRows(), inner_func1.IRows()) +=
        //     jx1.transpose() *
        //     hxO.template block<InnerFunc1::ORC, InnerFunc1::ORC>(
        //         inner_func1.IRows(), inner_func1.IRows()) *
        //     jx1;

        using FTtype = InnerFunc1;
        typename FTtype::template Jacobian<Scalar> jt;

        inner_func1.right_jacobian_product(
            jt,
            hxO.template block<FTtype::ORC, FTtype::ORC>(inner_func1.IRows(), inner_func1.IRows()),
            jx1,
            DirectAssignment(),
            std::bool_constant<false>());

        inner_func1.right_jacobian_product(
            hxO.template topLeftCorner<FTtype::IRC, FTtype::IRC>(inner_func1.IRows(), inner_func1.IRows()),
            jt.transpose(),
            jx1,
            PlusEqualsAssignment(),
            std::bool_constant<false>());

        this->inner_func1.right_jacobian_product(
            hx1,
            hxO.template block<InnerFunc1::IRC, InnerFunc1::ORC>(0, inner_func1.IRows()),
            jx1,
            DirectAssignment(),
            std::bool_constant<false>());

        adjhess += hxO.template topLeftCorner<InnerFunc1::IRC, InnerFunc1::IRC>(inner_func1.IRows(),
                                                                                inner_func1.IRows())
                   + hx1 + hx1.transpose();

      } else {
        //////////////////////
        Eigen::Matrix<Scalar, OuterFunc::IRC, Base::IRC> j0s;

        j0s.template topRows<Base::IRC>(this->IRows()).setIdentity();

        j0s.template middleRows<InnerFunc1::ORC>(this->inner_func1.IRows(), this->inner_func1.ORows()) = jx1;

        ASSET::tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;

          j0s.template middleRows<FTtype::ORC>(func_i.IRows(), func_i.ORows()) =
              std::get<i.value>(jxi).template leftCols<Base::IRC>(this->IRows())
              + std::get<i.value>(jxi).template rightCols<FTtype::IRC - Base::IRC>(func_i.IRows()
                                                                                   - this->IRows())
                    * j0s.template middleRows<FTtype::IRC - Base::IRC>(this->IRows(),
                                                                       func_i.IRows() - this->IRows());
        });

        ASSET::tuple_for_loop(this->inner_funcs, [&](const auto& func_i, auto i) {
          using FTtype =
              typename std::remove_const<typename std::remove_reference<decltype(func_i)>::type>::type;
          // if constexpr (!FTtype::IsLinearFunction) adjhess.noalias() +=
          // j0s.template topRows<FTtype::IRC>(func_i.IRows()).transpose() *
          // std::get<i.value>(hxi) * j0s.template
          // topRows<FTtype::IRC>(func_i.IRows());
          func_i.accumulate_hessian(
              hxO.template topLeftCorner<FTtype::IRC, FTtype::IRC>(func_i.IRows(), func_i.IRows()),
              std::get<i.value>(hxi),
              PlusEqualsAssignment());
        });

        // std::cout << std::setprecision(4)<< hxO << std::endl << std::endl;

        // adjhess.noalias() += j0s.transpose() * hxO * j0s;

        adjhess.template topLeftCorner<Base::IRC, Base::IRC>() +=
            hxO.template topLeftCorner<Base::IRC, Base::IRC>();

        constexpr int Ev = OuterFunc::IRC - Base::IRC;
        // Eigen::Matrix<Scalar, Ev, Base::IRC> j0s2 = j0s.template
        // bottomRows<Ev>();

        hxO.template leftCols<Base::IRC>().noalias() =
            hxO.template rightCols<Ev>() * j0s.template bottomRows<Ev>();

        adjhess.template topLeftCorner<Base::IRC, Base::IRC>() +=
            hxO.template topLeftCorner<Base::IRC, Base::IRC>()
            + hxO.template topLeftCorner<Base::IRC, Base::IRC>().transpose();

        adjhess.noalias() += j0s.template bottomRows<Ev>().transpose()
                             * hxO.template leftCols<Base::IRC>().template bottomRows<Ev>();
      }
    }
  };

}  // namespace ASSET

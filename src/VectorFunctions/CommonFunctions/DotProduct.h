#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, class Func1, class Func2>
  struct FunctionDotProduct_Impl;

  template<class Func1, class Func2>
  struct FunctionDotProduct : FunctionDotProduct_Impl<FunctionDotProduct<Func1, Func2>, Func1, Func2> {
    using Base = FunctionDotProduct_Impl<FunctionDotProduct<Func1, Func2>, Func1, Func2>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
  };

  template<class Derived, class Func1, class Func2>
  struct FunctionDotProduct_Impl : VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, 1> {
    using Base = VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, 1>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);
    using Base::compute;

    static const bool IsSegmentOp = Is_Segment<Func1>::value && Is_Segment<Func2>::value;
    static const bool IsVectorizable = Func1::IsVectorizable && Func2::IsVectorizable;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;


    Func1 func1;
    Func2 func2;


    FunctionDotProduct_Impl() {
    }
    FunctionDotProduct_Impl(Func1 f1, Func2 f2) : func1(std::move(f1)), func2(std::move(f2)) {
      int irtemp = std::max(this->func1.IRows(), this->func2.IRows());
      if (this->func1.ORows() != this->func2.ORows()) {
        fmt::print(
            fmt::fg(fmt::color::red),
            "Math Error in FunctionDotProduct/.dot method !!!\n"
            "Output Size of Func1 (ORows = {0:}) does not match Output Size of Func2 (ORows = {1:}).\n",
            this->func1.ORows(),
            this->func2.ORows());

        throw std::invalid_argument("");
      }
      if (this->func1.IRows() != this->func2.IRows()) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in FunctionDotProduct/.dot method !!!\n"
                   "Input Size of Func1 (IRows = {0:}) does not match Input Size of Func2 (IRows = {1:}).\n",
                   this->func1.IRows(),
                   this->func2.IRows());
        throw std::invalid_argument("");
      }

      this->setIORows(irtemp, 1);
      this->set_input_domain(this->IRows(), {this->func1.input_domain(), this->func2.input_domain()});
    }
    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<Derived>(m, name);
      obj.def(py::init<Func1, Func2>());
      Base::DenseBaseBuild(obj);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;

      VectorBaseRef<OutType> fx = fx_.const_cast_derived();


      if constexpr (IsSegmentOp) {
        fx[0] = x.template segment<Func1::ORC>(this->func1.SegStart, this->func1.ORows())
                    .dot(x.template segment<Func2::ORC>(this->func2.SegStart, this->func2.ORows()));
      } else {

        auto Impl = [&](auto& fx1, auto& fx2) {
          this->func1.compute(x, fx1);
          this->func2.compute(x, fx2);
          fx[0] = fx1.dot(fx2);
        };


        const int orows = this->func1.ORows();
        const int crit_size = orows;

        using FType1 = Func1_Output<Scalar>;
        using FType2 = Func2_Output<Scalar>;

        MemoryManager::allocate_run(crit_size, Impl, TempSpec<FType1>(orows, 1), TempSpec<FType2>(orows, 1));
      }
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();


      if constexpr (IsSegmentOp) {
        fx[0] = x.template segment<Func1::ORC>(this->func1.SegStart, this->func1.ORows())
                    .dot(x.template segment<Func2::ORC>(this->func2.SegStart, this->func2.ORows()));

        jx.template block<1, Func1::ORC>(0, this->func1.SegStart, 1, this->func1.ORows()) +=
            x.template segment<Func2::ORC>(this->func2.SegStart, this->func2.ORows()).transpose();

        jx.template block<1, Func2::ORC>(0, this->func2.SegStart, 1, this->func2.ORows()) +=
            x.template segment<Func1::ORC>(this->func1.SegStart, this->func1.ORows()).transpose();

      } else {

        auto Impl = [&](auto& fx1, auto& fx2, auto& jx1, auto& jx2) {
          this->func1.compute_jacobian(x, fx1, jx1);
          this->func2.compute_jacobian(x, fx2, jx2);
          fx[0] = fx1.dot(fx2);

          this->func1.right_jacobian_product(
              jx_, fx2.transpose(), jx1, DirectAssignment(), std::bool_constant<false>());
          this->func2.right_jacobian_product(
              jx_, fx1.transpose(), jx2, PlusEqualsAssignment(), std::bool_constant<false>());
        };


        const int orows = this->func1.ORows();
        const int irows = this->func1.IRows();
        const int crit_size = std::max({irows, orows});

        using FType1 = Func1_Output<Scalar>;
        using JType1 = Func2_jacobian<Scalar>;

        using FType2 = Func2_Output<Scalar>;
        using JType2 = Func2_jacobian<Scalar>;

        MemoryManager::allocate_run(crit_size,
                                    Impl,
                                    TempSpec<FType1>(orows, 1),
                                    TempSpec<FType2>(orows, 1),
                                    TempSpec<JType1>(orows, irows),
                                    TempSpec<JType2>(orows, irows));
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


      if constexpr (IsSegmentOp) {
        fx[0] = x.template segment<Func1::ORC>(this->func1.SegStart, this->func1.ORows())
                    .dot(x.template segment<Func2::ORC>(this->func2.SegStart, this->func2.ORows()));

        jx.template block<1, Func1::ORC>(0, this->func1.SegStart, 1, this->func1.ORows()) +=
            x.template segment<Func2::ORC>(this->func2.SegStart, this->func2.ORows()).transpose();

        jx.template block<1, Func2::ORC>(0, this->func2.SegStart, 1, this->func2.ORows()) +=
            x.template segment<Func1::ORC>(this->func1.SegStart, this->func1.ORows()).transpose();

        adjgrad.template segment<Func1::ORC>(this->func1.SegStart, this->func1.ORows()) +=
            x.template segment<Func2::ORC>(this->func2.SegStart, this->func2.ORows()) * adjvars[0];

        adjgrad.template segment<Func2::ORC>(this->func2.SegStart, this->func2.ORows()) +=
            x.template segment<Func1::ORC>(this->func1.SegStart, this->func1.ORows()) * adjvars[0];

        for (int i = 0; i < this->func1.ORows(); i++) {
          adjhess(this->func1.SegStart + i, this->func2.SegStart + i) += adjvars[0];
          adjhess(this->func2.SegStart + i, this->func1.SegStart + i) += adjvars[0];
        }

      } else {


        auto Impl = [&](auto& fx1, auto& fx2, auto& jx1, auto& jx2, auto& gx2, auto& hx2, auto& adjtemp) {
          this->func2.compute(x, adjtemp);

          adjtemp *= adjvars[0];

          this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx1, jx1, adjgrad, adjhess, adjtemp);

          adjtemp = fx1 * adjvars[0];

          this->func2.compute_jacobian_adjointgradient_adjointhessian(x, fx2, jx2, gx2, hx2, adjtemp);

          fx[0] = fx1.dot(fx2);
          if constexpr (!Func2::IsLinearFunction) {
            this->func2.accumulate_hessian(adjhess, hx2, PlusEqualsAssignment());
            this->func2.zero_matrix_domain(hx2);
          }

          this->func2.accumulate_gradient(adjgrad, gx2, PlusEqualsAssignment());

          this->func1.right_jacobian_product(
              hx2, jx2.transpose(), jx1, DirectAssignment(), std::bool_constant<false>());


          if constexpr (Func1::InputIsDynamic) {
            const int sds = this->func1.SubDomains.cols();
            if (sds == 0) {
              adjhess += hx2 + hx2.transpose();
            } else {
              for (int i = 0; i < sds; i++) {
                int start = this->func1.SubDomains(0, i);
                int size = this->func1.SubDomains(1, i);
                adjhess.middleCols(start, size) += hx2.middleCols(start, size) * adjvars[0];
                adjhess.middleRows(start, size) += hx2.middleCols(start, size).transpose() * adjvars[0];
              }
            }
          } else {
            constexpr int sds = Func1::INPUT_DOMAIN::SubDomains.size();
            ASSET::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                  constexpr int start = Func1::INPUT_DOMAIN::SubDomains[i.value][0];
                  constexpr int size = Func1::INPUT_DOMAIN::SubDomains[i.value][1];

                  adjhess.template middleCols<size>(start, size) +=
                      hx2.template middleCols<size>(start, size) * adjvars[0];
                  adjhess.template middleRows<size>(start, size) +=
                      hx2.template middleCols<size>(start, size).transpose() * adjvars[0];
                });
          }


          this->func1.right_jacobian_product(
              jx_, fx2.transpose(), jx1, DirectAssignment(), std::bool_constant<false>());
          this->func2.right_jacobian_product(
              jx_, fx1.transpose(), jx2, PlusEqualsAssignment(), std::bool_constant<false>());
        };


        const int orows = this->func1.ORows();
        const int irows = this->func1.IRows();
        const int crit_size = std::max({irows, orows});

        using FType1 = Func1_Output<Scalar>;
        using JType1 = Func2_jacobian<Scalar>;

        using FType2 = Func2_Output<Scalar>;
        using JType2 = Func2_jacobian<Scalar>;
        using GType2 = Func2_gradient<Scalar>;
        using HType2 = Func2_hessian<Scalar>;

        MemoryManager::allocate_run(crit_size,
                                    Impl,
                                    TempSpec<FType1>(orows, 1),
                                    TempSpec<FType2>(orows, 1),
                                    TempSpec<JType1>(orows, irows),
                                    TempSpec<JType2>(orows, irows),
                                    TempSpec<GType2>(irows, 1),
                                    TempSpec<HType2>(irows, irows),
                                    TempSpec<FType1>(orows, 1));
      }
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };

}  // namespace ASSET

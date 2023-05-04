#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Func, int IRC, int ORC>
  struct ParsedInput : VectorFunction<ParsedInput<Func, IRC, ORC>, IRC, ORC, Analytic> {
    using Base = VectorFunction<ParsedInput<Func, IRC, ORC>, IRC, ORC, Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);

    SUB_FUNCTION_IO_TYPES(Func);
    Func func;
    Func_Input<int> varlocs;
    bool contiguous = false;

    ParsedInput() {
    }
    ParsedInput(Func f, const Func_Input<int>& varlocs, int irr) : func(std::move(f)), varlocs(varlocs) {
      this->setIORows(irr, this->func.ORows());

      this->contiguous = true;
      for (int i = 0; i < varlocs.size() - 1; i++) {
        int delta = varlocs[i + 1] - varlocs[i];
        if (delta != 1)
          this->contiguous = false;
      }
      // this->set_input_domain(irr, func.input_domain());
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<ParsedInput<Func, IRC, ORC>>(m, name);
      obj.def(py::init<Func, const Func_Input<int>&, int>());
      Base::DenseBaseBuild(obj);
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      using Scalar = typename InType::Scalar;

      if (this->contiguous) {
        this->func.compute(x.segment(varlocs[0], this->func.IRows()), fx_);

      } else {
        Func_Input<Scalar> xin(this->func.IRows());
        for (int i = 0; i < this->func.IRows(); i++) {
          xin[i] = x[this->varlocs[i]];
        }
        this->func.compute(xin, fx_);
      }
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();

      using Scalar = typename InType::Scalar;

      if (this->contiguous) {
        this->func.compute_jacobian(
            x.segment(varlocs[0], this->func.IRows()), fx_, jx.middleCols(varlocs[0], this->func.IRows()));
      } else {
        Func_Input<Scalar> xin(this->func.IRows());
        Func_jacobian<Scalar> jxin(this->func.ORows(), this->func.IRows());
        jxin.setZero();
        for (int i = 0; i < this->func.IRows(); i++) {
          xin[i] = x[this->varlocs[i]];
        }
        this->func.compute_jacobian(xin, fx_, jxin);
        for (int i = 0; i < this->func.IRows(); i++) {
          jx.col(this->varlocs[i]) = jxin.col(i);
        }
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
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      Eigen::MatrixBase<AdjHessType>& adjhess = adjhess_.const_cast_derived();

      using Scalar = typename InType::Scalar;

      if (this->contiguous) {
        this->func.compute_jacobian_adjointgradient_adjointhessian(
            x.segment(varlocs[0], this->func.IRows()),
            fx_,
            jx.middleCols(varlocs[0], this->func.IRows()),
            adjgrad.segment(varlocs[0], this->func.IRows()),
            adjhess.block(varlocs[0], varlocs[0], this->func.IRows(), this->func.IRows()),
            adjvars);

      } else {
        Func_Input<Scalar> xin(this->func.IRows());
        Func_jacobian<Scalar> jxin(this->func.ORows(), this->func.IRows());
        Func_gradient<Scalar> gxin(this->func.IRows());
        Func_hessian<Scalar> hxin(this->func.IRows(), this->func.IRows());
        jxin.setZero();
        hxin.setZero();
        gxin.setZero();

        for (int i = 0; i < this->func.IRows(); i++) {
          xin[i] = x[this->varlocs[i]];
        }

        this->func.compute_jacobian_adjointgradient_adjointhessian(xin, fx_, jxin, gxin, hxin, adjvars);

        for (int i = 0; i < this->func.IRows(); i++) {
          jx.col(this->varlocs[i]) = jxin.col(i);
          adjgrad[this->varlocs[i]] = gxin[i];
        }
        for (int i = 0; i < this->func.IRows(); i++) {
          for (int j = 0; j < this->func.IRows(); j++) {
            adjhess(this->varlocs[j], this->varlocs[i]) = hxin(j, i);
          }
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };

}  // namespace ASSET

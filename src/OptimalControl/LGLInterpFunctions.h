#pragma once

#include "LGLInterpTable.h"

namespace ASSET {

  template<int OR>
  struct InterpFunction : VectorFunction<InterpFunction<OR>, 1, OR, Analytic, Analytic> {
    using Base = VectorFunction<InterpFunction<OR>, 1, OR, Analytic, Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    static constexpr int TempSize = SZ_SUM<OR, 1>::value;
    std::shared_ptr<LGLInterpTable> table;
    Eigen::VectorXi vars;
    InterpFunction() {
    }

    InterpFunction(std::shared_ptr<LGLInterpTable> tab, Eigen::VectorXi v) {
      this->table = tab;
      this->vars = v;

      if (v.maxCoeff() + 1 > tab->XtUVars) {
        throw std::invalid_argument("Interpolation table has incorrect dimensions");
      }

      this->setIORows(1, this->vars.size());
      // this->setHessFDSteps(tab->DeltaT/10.0);
    }
    InterpFunction(std::shared_ptr<LGLInterpTable> tab) {
      this->table = tab;
      if (this->table->XVars != OR || this->table->UVars > 0) {
        throw std::invalid_argument("Interpolation table has incorrect dimensions");
      }
      this->vars.resize(tab->XVars);
      for (int i = 0; i < this->vars.size(); i++)
        this->vars[i] = i;
      this->setIORows(1, this->vars.size());
      // this->setHessFDSteps(tab->DeltaT / 10.0);
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      auto Impl = [&](auto maxsize) {
        Scalar t = x[0];
        Eigen::Matrix<Scalar, TempSize, 1, 0, maxsize.value, 1> state;
        state.resize(this->table->XtUVars);
        this->table->InterpolateRef(t, state);
        for (int i = 0; i < this->ORows(); i++) {
          fx[i] = state[this->vars[i]];
        }
      };

      if constexpr (OR > 0)
        Impl(std::integral_constant<int, TempSize>());
      else
        LambdaJumpTable<4, 8, 16>::run(Impl, this->table->XtUVars);
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      auto Impl = [&](auto maxsize) {
        Scalar t = x[0];
        Eigen::Matrix<Scalar, TempSize, 2, 0, maxsize.value, 2> state;
        state.resize(this->table->XtUVars, 2);
        this->table->InterpolateDerivRef(t, state);
        for (int i = 0; i < this->ORows(); i++) {
          fx[i] = state(this->vars[i], 0);
          jx(i, 0) = state(this->vars[i], 1);
        }
      };

      if constexpr (OR > 0)
        Impl(std::integral_constant<int, TempSize>());
      else
        LambdaJumpTable<4, 8, 16>::run(Impl, this->table->XtUVars);
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

      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      MatrixBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> hx = adjhess_.const_cast_derived();

      auto Impl = [&](auto maxsize) {
        if (this->table->Method == TranscriptionModes::LGL3) {
          Scalar t = x[0];
          Eigen::Matrix<Scalar, TempSize, 3, 0, maxsize.value, 3> state;
          state.resize(this->table->XtUVars, 3);
          this->table->Interpolate2ndDerivRef(t, state);
          for (int i = 0; i < this->ORows(); i++) {
            fx[i] = state(this->vars[i], 0);
            jx(i, 0) = state(this->vars[i], 1);
            adjgrad[0] += jx(i, 0) * adjvars[i];
            hx(0, 0) += (state(this->vars[i], 2)) * adjvars[i];
          }
        } else {
          Scalar t = x[0];
          Eigen::Matrix<Scalar, TempSize, 2, 0, maxsize.value, 2> state;
          state.resize(this->table->XtUVars, 2);
          this->table->InterpolateDerivRef(t, state);
          for (int i = 0; i < this->ORows(); i++) {
            fx[i] = state(this->vars[i], 0);
            jx(i, 0) = state(this->vars[i], 1);
            adjgrad[0] += jx(i, 0) * adjvars[i];
          }
          state.setZero();
          Scalar h = Scalar(this->table->DeltaT / 10.0);
          this->table->InterpolateDerivRef(t + h, state);
          for (int i = 0; i < this->ORows(); i++) {
            hx(0, 0) += (state(this->vars[i], 1) - jx(i, 0)) * adjvars[i] / h;
          }
        }
      };

      if constexpr (OR > 0)
        Impl(std::integral_constant<int, TempSize>());
      else
        LambdaJumpTable<4, 8, 16>::run(Impl, this->table->XtUVars);
    }


    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<InterpFunction<OR>>(m, name);
      if (OR == -1) {
        obj.def(py::init<std::shared_ptr<LGLInterpTable>, Eigen::VectorXi>());
      } else {
        obj.def(py::init<std::shared_ptr<LGLInterpTable>>());
      }

      Base::DenseBaseBuild(obj);

      obj.def("__call__", [](const InterpFunction<OR>& self, const GenericFunction<-1, 1>& t) {
        return GenericFunction<-1, -1>(self.eval(t));
      });
      obj.def("__call__", [](const InterpFunction<OR>& self, const Segment<-1, 1, -1>& t) {
        return GenericFunction<-1, -1>(self.eval(t));
      });
    }
  };


}  // namespace ASSET

#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Func>
  struct IOScaled : VectorFunction<IOScaled<Func>, Func::IRC, Func::ORC, Analytic> {
    using Base = VectorFunction<IOScaled<Func>, Func::IRC, Func::ORC, Analytic> ;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    Func func;
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static const bool IsLinearFunction = Func::IsLinearFunction;
    static const bool IsVectorizable = Func::IsVectorizable;

    Input<double> input_scales;
    Output<double> output_scales;

    IOScaled() { }

    IOScaled(Func f, const Input<double>& input_scales, const Output<double>& output_scales)
        : func(std::move(f)) {
      this->setIORows(this->func.IRows(), this->func.ORows());
      this->set_input_domain(this->IRows(), {this->func.input_domain()});

      this->input_scales = input_scales;
      this->output_scales = output_scales;

      this->EnableVectorization = this->func.EnableVectorization;
    }
    
    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<IOScaled<Func>>(m, name);
      obj.def(py::init<Func , const Input<double> & , const Output<double> & >());
      Base::DenseBaseBuild(obj);
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      auto Impl = [&](auto& x_scaled) {
            for (int i = 0; i < this->IRows(); i++) {
                 x_scaled[i] = this->input_scales[i] * x[i];
            }

        this->func.compute(x_scaled, fx);
        
        for (int i = 0; i < this->ORows(); i++) {
                 fx[i] *= this->output_scales[i];
        }
      };


      MemoryManager::allocate_run(this->IRows(),
                                  Impl, TempSpec<Input<Scalar>>(this->IRows(), 1));


    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {


     typedef typename InType::Scalar Scalar;
     VectorBaseRef<OutType> fx = fx_.const_cast_derived();
     MatrixBaseRef<JacType> jx = jx_.const_cast_derived();


      auto Impl = [&](auto& x_scaled) {
        for (int i = 0; i < this->IRows(); i++) {
          x_scaled[i] = this->input_scales[i] * x[i];
        }

      this->func.compute_jacobian(x_scaled, fx, jx);

        for (int i = 0; i < this->ORows(); i++) {
          fx[i] *= this->output_scales[i];
          jx.row(i) *= Scalar(this->output_scales[i]);
        }
        for (int i = 0; i < this->IRows(); i++) {
          jx.col(i) *= Scalar(this->input_scales[i]);
        }

      };

      MemoryManager::allocate_run(this->IRows(), Impl, TempSpec<Input<Scalar>>(this->IRows(), 1));

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


        auto Impl = [&](auto& x_scaled,auto & l_scaled) {
        for (int i = 0; i < this->IRows(); i++) {
          x_scaled[i] = this->input_scales[i] * x[i];
        }
        for (int i = 0; i < this->ORows(); i++) {
          l_scaled[i] = this->output_scales[i] * adjvars[i];
        }
        this->func.compute_jacobian_adjointgradient_adjointhessian(
            x_scaled, fx, jx, adjgrad, adjhess, l_scaled);



        for (int i = 0; i < this->ORows(); i++) {
          fx[i] *= this->output_scales[i];
          jx.row(i) *= Scalar(this->output_scales[i]);
        }
        for (int i = 0; i < this->IRows(); i++) {
          jx.col(i) *= Scalar(this->input_scales[i]);
          adjhess.col(i) *= Scalar(this->input_scales[i]);
          adjgrad[i] *=this->input_scales[i];
        }

        for (int i = 0; i < this->IRows(); i++) {
            adjhess.row(i) *= Scalar(this->input_scales[i]);
        }

      };

      MemoryManager::allocate_run(this->IRows(),
                                    Impl,
                                    TempSpec<Input<Scalar>>(this->IRows(), 1),
                                    TempSpec<Output<Scalar>>(this->ORows(), 1));




    }

   
    
  };

}  // namespace ASSET
#pragma once

#include "DenseDerivatives.h"

namespace ASSET {

  //! First derivatives using forward finite difference
  /*!
    \tparam IR Input Rows
    \tparam OR Output Rows
  */
  template<class Derived, int IR, int OR>
  struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeModes::FDiffFwd>
      : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    DenseFirstDerivatives() {
      this->setJacFDSteps(1.0e-7);
    }

    //! Set step size for each input dimension
    void setJacFDSteps(const Input<double>& steps) {
      this->jacFDSteps = steps;
    }
    //! Set step size for all input dimensions
    void setJacFDSteps(double step) {
      this->jacFDSteps.resize(this->IRows());
      this->jacFDSteps.setConstant(step);
    }

    //! Jacobian implementation
    /*!
      \tparam InType Eigen type of input x
      \tparam OutType Eigen type of output fx
      \tparam JacType Eigen type of jacobian jx
      \param x const reference to input vector
      \param fx_ const reference to output vector
      \param jx_ const reference to jacobian matrix
      Calculates function and jacobian and stores them in fx_ and jx_. Requires
      (IR+1) function calls.
    */
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      this->derived().compute(x, fx_);
      Input<Scalar> xi = x;
      Output<Scalar> fi(this->ORows());
      fi.setZero();
      for (int i = 0; i < this->IRows(); i++) {
        xi[i] += this->jacFDSteps[i];
        this->derived().compute(xi, fi);

        jx.col(i) = (fi - fx) / Scalar(this->jacFDSteps[i]);

        fi.setZero();
        xi[i] = x[i];
      }
    }

   protected:
    Eigen::VectorXd jacFDSteps;
  };

  ////////////////////////////////////////////////////////////////////////////////

  //! Second derivatives using forward finite difference
  /*!
    \tparam IR Input Rows
    \tparam OR Output Rows
    \tparam JMode Jacobian Mode (enumerator)
  */
  template<class Derived, int IR, int OR, int JMode>
  struct DenseSecondDerivatives<Derived, IR, OR, JMode, DenseDerivativeModes::FDiffFwd>
      : DenseFirstDerivatives<Derived, IR, OR, JMode> {
    using Base = DenseFirstDerivatives<Derived, IR, OR, JMode>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    DenseSecondDerivatives() {
      this->setHessFDSteps(1.0e-7);
    }
    using Base::adjointhessian;
    //! Set step size for each input dimension
    void setHessFDSteps(const Input<double>& steps) {
      this->hessFDSteps = steps;
    }
    //! Set step size for all input dimensions
    void setHessFDSteps(double step) {
      this->hessFDSteps = Input<double>::Constant(this->IRows(), step);
    }

    //! Adjoint hessian implementation
    /*!
      \tparam InType Eigen type of input x
      \tparam AdjHessType Eigen type of output adjoint hessian matrix
      \tparam AdjVarType Eigen type of adjoint coefficient vector
      \param x const reference to input vector
      \param adjhess_ const reference to adjoint hessian matrix
      \param adjvars const reference to adjoint coefficient vector
      Calculates adjoint hessian matrix by taking the derivative of the adjoint
      gradient vector.
    */
    template<class InType, class AdjHessType, class AdjVarType>
    inline void adjointhessian(ConstVectorBaseRef<InType> x,
                               ConstMatrixBaseRef<AdjHessType> adjhess_,
                               ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      Gradient<Scalar> ag(this->IRows());
      Gradient<Scalar> agi(this->IRows());
      ag.setZero();
      agi.setZero();
      this->adjointgradient(x, ag, adjvars);

      Input<Scalar> xi = x;
      for (int i = 0; i < this->IRows(); i++) {
        // if (this->VariableTypes[i] == VarTypes::Linear) {
        if (false) {
          for (int j = 0; j < this->IRows(); j++) {
            adjhess(j, i) = Scalar(0.0);
          }
        } else {
          xi[i] += this->hessFDSteps[i];
          this->adjointgradient(xi, agi, adjvars);
          for (int j = 0; j < this->IRows(); j++) {
            adjhess(j, i) = (agi[j] - ag[j]) / Scalar(this->hessFDSteps[i]);
          }
          agi.setZero();
          xi[i] = x[i];
        }
      }
      adjhess = (adjhess + adjhess.transpose()).eval() * Scalar(0.5);
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
      this->derived().compute_jacobian_adjointgradient(x, fx_, jx_, adjgrad_, adjvars);
      adjointhessian(x, adjhess_, adjvars);
    }

   protected:
    Input<double> hessFDSteps;
  };

}  // namespace ASSET

#pragma once

#include "DenseDerivatives.h"

namespace ASSET {

  //! First derivatives using forward finite difference
  /*!
    \tparam IR Input Rows
    \tparam OR Output Rows
  */
  template<class Derived, int IR, int OR>
  struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeModes::FDiffCentArray>
      : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    DenseFirstDerivatives() {
      this->setJacFDSteps(1.0e-5);
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

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      this->derived().compute(x, fx_);
      using ScalArray = Eigen::Array<Scalar, 2, 1>;
      Input<Eigen::Array<Scalar, 2, 1>> xi = x.template cast<Eigen::Array<Scalar, 2, 1>>();
      Output<Eigen::Array<Scalar, 2, 1>> fi(this->ORows());
      fi.setZero();
      for (int i = 0; i < this->IRows(); i++) {
        xi[i][0] += this->jacFDSteps[i];
        xi[i][1] -= this->jacFDSteps[i];
        this->derived().compute(xi, fi);
        for (int j = 0; j < this->ORows(); j++) {
          jx(j, i) = (fi[j][0] - fi[j][1]) / (2.0 * jacFDSteps[i]);
        }
        fi.setZero();
        xi[i] = x[i];
      }
    }

   protected:
    Eigen::VectorXd jacFDSteps;
  };
}  // namespace ASSET
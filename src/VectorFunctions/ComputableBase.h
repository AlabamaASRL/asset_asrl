/*
File Name: ComputableBase.h

File Description:

Implements Base class of all dense and sparse expressions with a compute method.
Template parameters are the derived class (Derived), and the compile time value of the input (IR) and
output (OR) rows of the vectorfunction.

Inherits from CRTP to gain derived casting capabilities.
Inherits from InputOutputSize<IR, OR> to gain fields for holding input and outsizes if necessary
for dynamic sized functions (IR=OR=-1).

Defines the default set of constexpr boolean info about functions that are used by the expression system,
such as IsVectorizable. Derived classes are intended to overide these as neccessary.


Adds functions for getting and setting the input and output rows.

Defines the .compute fuction in terms of derived().compute_impl which must be implemented by a derived class.

Also defines implements the constraints function in terms of the compute function. The constraints functions
is part of a vector functions interface to the non-linear optimzier PSIOPT.

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
#include "DetectSuperScalar.h"
#include "FunctionalFlags.h"
#include "IndexingData.h"
#include "InputOuputSize.h"
#include "Utils/CRTPBase.h"
#include "Utils/SizingHelpers.h"
#include "pch.h"

#if defined(ASSET_MEMORYMAN)
  #include "Utils/MemoryManagement.h"
#endif


namespace ASSET {

  /*!
   * @brief A computable is anything with a \code compute \endcode function
   *
   * @tparam Derived CRTP Derived class
   * @tparam IR Input Rows
   * @tparam OR Output Rows
   */
  template<class Derived, int IR, int OR>
  struct ComputableBase : CRTPBase<Derived>, InputOutputSize<IR, OR> {
    ///////////////////////TypeDefs////////////////////////////////////////////
    template<class Scalar>
    using Output = Eigen::Matrix<Scalar, OR, 1>;
    template<class Scalar>
    using Input = Eigen::Matrix<Scalar, IR, 1>;
    template<class Scalar>
    using Gradient = Eigen::Matrix<Scalar, IR, 1>;

    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar> &;
    template<class Scalar>
    using VectorBaseRef = Eigen::MatrixBase<Scalar> &;

    /// Input Rows at Compile Time (-1 if Dynamic)
    static const int IRC = IR;
    /// Output Rows at Compile Time (-1 if Dynamic)
    static const int ORC = OR;

    static const bool InputIsDynamic = (IR < 0);
    static const bool OutputIsDynamic = (OR < 0);
    static const bool JacobianIsDynamic = (IR < 0 || OR < 0);
    static const bool FullyDynamic = (IR < 0 && OR < 0);

    static const bool IsVectorizable = false;
    static const bool IsLinearFunction = false;
    static const bool HasDiagonalJacobian = false;
    static const bool HasDiagonalHessian = false;
    static const bool IsCwiseOperator = false;
    static const bool IsGenericFunction = false;
    static const bool IsConditional = false;

    mutable bool EnableVectorization = false;

    void enable_vectorization(bool b) const {
      this->EnableVectorization = b;
    }

    constexpr bool is_linear() const {
      return Derived::IsLinearFunction;
    }

    void setInputRows(int inputrows) {
      if constexpr (IR < 0) {
        this->InputRows = inputrows;
      }
    }
    void setOutputRows(int outputrows) {
      if constexpr (OR < 0) {
        this->OutputRows = outputrows;
      }
    }
    void setIORows(int inputrows, int outputrows) {
      this->setInputRows(inputrows);
      this->setOutputRows(outputrows);
    }
    inline int IRows() const {
      if constexpr (IR < 0) {
        return this->InputRows;
      } else {
        return IR;
      }
    }
    inline int ORows() const {
      if constexpr (OR < 0) {
        return this->OutputRows;
      } else {
        return OR;
      }
    }

    bool thread_safe() const {
      return true;
    }
    //////////////////////////////////////////////////////////////////////////////
    ComputableBase() {
    }
    ComputableBase(int inputrows, int outputrows) {
      this->setIORows(inputrows, outputrows);
    }

    /*!
     * @brief Calls compute on the derived class.
     *
     * @tparam InType Eigen type of input vector
     * @tparam OutType Eigen type of output vector
     * @param x Input vector
     * @param fx_ Output vector
     */
    template<class InType, class OutType>
    inline void compute(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      if constexpr (!Derived::IsVectorizable) {
        if constexpr (Is_SuperScalar<Scalar>::value) {
          VectorBaseRef<OutType> fx = fx_.const_cast_derived();

          typedef typename Scalar::Scalar RealScalar;

          Input<RealScalar> x_r;
          Output<RealScalar> fx_r;

          const int IRR = this->IRows();
          const int ORR = this->ORows();

          if constexpr (InputIsDynamic)
            x_r.resize(IRR);
          if constexpr (OutputIsDynamic)
            fx_r.resize(ORR);

          for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
            for (int j = 0; j < IRR; j++)
              x_r[j] = x[j][i];
            this->derived().compute_impl(x_r, fx_r);
            for (int j = 0; j < ORR; j++)
              fx[j][i] = fx_r[j];
            fx_r.setZero();
          }

        } else {
          this->derived().compute_impl(x, fx_);
        }
      } else {
        this->derived().compute_impl(x, fx_);
      }
    }

    /*!
     * @brief Returns the output of compute on the derived class
     *
     * @tparam InType Eigen type of input vector
     * @param x Input vecter
     * @return Output<typename InType::Scalar> Output type
     */
    template<class InType>
    inline Output<typename InType::Scalar> compute(ConstVectorBaseRef<InType> x) const {
      typedef typename InType::Scalar Scalar;
      Output<Scalar> fx(this->ORows());
      fx.setZero();
      this->derived().compute(x, fx);
      return fx;
    }

    template<class InType, class OutType, class AdjGradType, class AdjVarType>
    inline void compute_adjointgradient(ConstVectorBaseRef<InType> x,
                                        ConstVectorBaseRef<OutType> fx_,
                                        ConstVectorBaseRef<AdjGradType> adjgrad_,
                                        ConstVectorBaseRef<AdjVarType> adjvars) const {
      this->derived().compute_adjointgradient(x, fx_, adjgrad_, adjvars);
    }

    template<class InType, class AdjGradType, class AdjVarType>
    inline void adjointgradient(ConstVectorBaseRef<InType> x,
                                ConstVectorBaseRef<AdjGradType> adjgrad_,
                                ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      Output<Scalar> fx(this->ORows());
      fx.setZero();
      this->derived().compute_adjointgradient(x, fx, adjgrad_, adjvars);
    }

    template<class InType, class AdjVarType>
    inline Gradient<typename InType::Scalar> adjointgradient(ConstVectorBaseRef<InType> x,
                                                             ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      Gradient<Scalar> adjgrad(this->IRows());
      adjgrad.setZero();
      this->derived().adjointgradient(x, adjgrad, adjvars);
      return adjgrad;
    }


    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    /*
    * This function is the interface allowing the vector function to be used as a constraint inside psiopt.
    * Vector X is the total variables vector for the full optimization problem, and FX is the total eqaility
      or inequality constraints vector for the problem. Data pertaining to the loction of the input variables
    in X for each distinct call as well as the target location for the output variables in FX is stored in the
    SolverIndexingData object. In general users should not directly overload this function unless they have a
    very good reason.
    */
    void constraints(ConstEigenRef<Eigen::VectorXd> X,
                     EigenRef<Eigen::VectorXd> FX,
                     const SolverIndexingData &data) const {

      Input<double> x(this->IRows());
      Eigen::Map<Output<double>> fx(NULL, this->ORows());


      const int IRR = this->IRows();
      const int ORR = this->ORows();

      // Scalar non-vectorized call to the funtion
      auto ScalarImpl = [&](int start, int stop) {
        for (int V = start; V < stop; V++) {
          this->gatherInput(X, x, V, data);
          new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V], this->ORows());
          fx.setZero();
          this->derived().compute(x, fx);
        }
      };

      /*
      Super Scalar vectorized call to the function.
      Does as many fully packed vectorized calls as possible
      then reverts to the scalar implementation.
      */

      auto VectorImpl = [&]() {
        using SuperScalar = ASSET::DefaultSuperScalar;
        constexpr int vsize = SuperScalar::SizeAtCompileTime;
        int Packs = data.NumAppl() / vsize;
        ;

        Input<SuperScalar> x_vect(this->IRows());
        Output<SuperScalar> fx_vect(this->ORows());

        for (int i = 0; i < Packs; i++) {
          for (int j = 0; j < vsize; j++) {
            int V = i * vsize + j;
            this->gatherInput(X, x, V, data);
            for (int k = 0; k < IRR; k++) {
              x_vect[k][j] = x[k];
            }
          }
          fx_vect.setZero();
          this->derived().compute(x_vect, fx_vect);
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

      // Only try vectorized impl if Derived allows and it is enabled
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

    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L,
                                     EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const SolverIndexingData &data) const {
      Input<double> x(this->IRows());
      Output<double> l(this->ORows());

      Eigen::Map<Output<double>> fx(NULL, this->ORows());
      Eigen::Map<Input<double>> agx(NULL, this->IRows());

      for (int V = 0; V < data.NumAppl(); V++) {
        this->gatherInput(X, x, V, data);
        this->gatherMult(L, l, V, data);
        new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V], this->ORows());
        new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.InnerGradientStarts[V], this->IRows());
        fx.setZero();
        agx.setZero();
        this->derived().compute_adjointgradient(x, fx, agx, l);
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////


   protected:
    /*
     * This helper is used to gather the required variables for the Vth function call
     * from the optimizer variables vector, X, into the local input vector of the function, xt.
     * The solver indexing data structure constains the locations of the variables for the Vth
     * function call as well as precomputed metadata indicating whether these variables are stored
     * contiguously. In the case of contiguous storage the varibels are retrieved in a single call
     * This is implemented here to potentially take advantage of the fact the the input size is a
     * known compile time constant. This was observed to provide a modest but measurable perf improvement.
     */
    inline void gatherInput(ConstEigenRef<Eigen::VectorXd> X,
                            Input<double> &xt,
                            int V,
                            const SolverIndexingData &data) const {
      if (data.VindexContinuity[V] == ParsedIOFlags::Contiguous) {
        xt = X.template segment<IR>(data.VLoc(0, V), this->IRows());
      } else {
        for (int i = 0; i < this->IRows(); i++)
          xt(i) = X(data.VLoc(i, V));
      }
    }

    /*
     * This helper is used to gather the required lagrange multipliers for the Vth function call
     * from the optimizer multipler vector, L, into the local multiplier vector of the function, lt.
     * See above for justification.
     */
    inline void gatherMult(ConstEigenRef<Eigen::VectorXd> L,
                           Output<double> &l,
                           int V,
                           const SolverIndexingData &data) const {
      if (data.CindexContinuity[V] == ParsedIOFlags::Contiguous) {
        l = L.template segment<OR>(data.CLoc(0, V), this->ORows());
      } else {
        for (int i = 0; i < this->ORows(); i++)
          l(i) = L(data.CLoc(i, V));
      }
    }
  };

}  // namespace ASSET

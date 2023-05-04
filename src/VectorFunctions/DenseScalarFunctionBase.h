#pragma once

#include "DenseFunctionBase.h"

namespace ASSET {

  template<class Derived, int IR>
  struct DenseScalarFunctionBase : DenseFunctionBase<Derived, IR, 1> {
    using Base = DenseFunctionBase<Derived, IR, 1>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    void objective(double ObjScale,
                   ConstEigenRef<Eigen::VectorXd> X,
                   double& Val,
                   const SolverIndexingData& data) const {
      Input<double> x(this->IRows());
      Output<double> fx(1);

      for (int V = 0; V < data.NumAppl(); V++) {
        this->gatherInput(X, x, V, data);
        fx.setZero();
        this->derived().compute(x, fx);
        Val += fx[0] * ObjScale;
      }
    }
    void objective_gradient(double ObjScale,
                            ConstEigenRef<Eigen::VectorXd> X,
                            double& Val,
                            EigenRef<Eigen::VectorXd> GX,
                            const SolverIndexingData& data) const {
      Input<double> x(this->IRows());
      Output<double> fx(1);
      Jacobian<double> jx(1, this->IRows());
      Eigen::Map<Input<double>> gx(NULL, this->IRows());

      for (int V = 0; V < data.NumAppl(); V++) {
        this->gatherInput(X, x, V, data);
        new (&gx) Eigen::Map<Input<double>>(GX.data() + data.InnerGradientStarts[V], this->IRows());
        fx.setZero();
        jx.setZero();
        gx.setZero();

        this->derived().compute_jacobian(x, fx, jx);
        Val += fx[0] * ObjScale;
        gx = jx.transpose() * ObjScale;
      }
    }
    void objective_gradient_hessian(double ObjScale,
                                    ConstEigenRef<Eigen::VectorXd> X,
                                    double& Val,
                                    EigenRef<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                                    EigenRef<Eigen::VectorXi> KKTLocations,
                                    EigenRef<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex>& KKTLocks,
                                    const SolverIndexingData& data) const {
      Input<double> x(this->IRows());
      Output<double> fx(1);
      Jacobian<double> jx(1, this->IRows());
      Eigen::Map<Input<double>> gx(NULL, this->IRows());
      Hessian<double> hx(this->IRows(), this->IRows());
      Output<double> lm(1);
      lm[0] = ObjScale;

      for (int V = 0; V < data.NumAppl(); V++) {
        this->gatherInput(X, x, V, data);
        new (&gx) Eigen::Map<Input<double>>(GX.data() + data.InnerGradientStarts[V], this->IRows());

        fx.setZero();
        jx.setZero();
        gx.setZero();
        hx.setZero();

        this->derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);

        Val += fx[0] * ObjScale;
        this->KKTFillHess(V, hx, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
      }
    }
    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////

   protected:
    // double Scale = 1.0;
    void KKTFillHess(int Apl,
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

      const int IRR = (Base::IRC > 0) ? Base::IRC : this->IRows();

      for (int i = 0; i < IRR; i++) {
        ActiveVar = data.VLoc(i, Apl);
        Lock(ActiveVar);
        ///// insert hessian column symetrically
        for (int j = i; j < IRR; j++) {
          this->derived().AddHessianElem(hx(j, i), j, i, mpt, lpt, freeloc);
        }
        ///////////////////////////////////////////////////////////////////////////////
        UnLock(ActiveVar);
      }
    }
  };
}  // namespace ASSET

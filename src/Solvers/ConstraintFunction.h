/*
File Name: ConstraintFunction.h

File Description: Implements the ConstraintFunction class.
Holds an ConstraintInterface type erasure class and SolverIndexingData struct.
Interfaces directly with NonLinearProgram and PSIOPT.

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

#include "SolverFunctionBase.h"
#include "pch.h"

namespace ASSET {

  struct ConstraintFunction : SolverFunctionBase<ConstraintInterface> {
    using Base = SolverFunctionBase<ConstraintInterface>;
    using Base::function;
    using Base::index_data;
    using MatrixXi = Eigen::MatrixXi;
    using VectorXi = Eigen::VectorXi;

    ConstraintFunction() {
    }

    ConstraintFunction(const ConstraintInterface& f, const MatrixXi& vindex, const MatrixXi& cindex) {
      this->function = f;
      this->index_data = SolverIndexingData(f.IRows(), f.ORows(), vindex, cindex);
    }

    ConstraintFunction(const ConstraintInterface& f, const SolverIndexingData& data) {
      this->function = f;
      this->index_data = data;
    }

    /*
    Partitions multiple calls to this function into seperate ConstraintFunction instances that
    will be called on multiple threads.
    */
    std::vector<ConstraintFunction> thread_split(int Thr) const {
      std::vector<SolverIndexingData> idat = this->index_data.thread_split(Thr);
      std::vector<ConstraintFunction> split(idat.size());
      for (int i = 0; i < idat.size(); i++) {
        split[i] = ConstraintFunction(this->function, idat[i]);
      }
      return split;
    }

    /*
    Interface for calling the underlying type erased function's .constraints method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void constraints(ConstEigenRef<Eigen::VectorXd> X, Eigen::Ref<Eigen::VectorXd> FX) const {
      this->function.constraints(X, FX, this->index_data);
    }


    /*
    Interface for calling the underlying type erased function's .constraints_adjointgradient method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L,
                                     EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX) const {
      this->function.constraints_adjointgradient(X, L, FX, AGX, this->index_data);
    }


    /*
    Interface for calling the underlying type erased function's .constraints_jacobian method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X,
                              EigenRef<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                              EigenRef<Eigen::VectorXi> KKTLocations,
                              EigenRef<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex>& KKTLocks) const {
      this->function.constraints_jacobian(
          X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, this->index_data);
    }


    /*
    Interface for calling the underlying type erased function's .constraints_jacobian_adjointgradient method.
    Passes the arguments from PSIOPT and NonLinearProgram as well as the indexing data struct to the
    underlying vector function.
    */
    void constraints_jacobian_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                              ConstEigenRef<Eigen::VectorXd> L,
                                              EigenRef<Eigen::VectorXd> FX,
                                              EigenRef<Eigen::VectorXd> AGX,
                                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                                              EigenRef<Eigen::VectorXi> KKTLocations,
                                              EigenRef<Eigen::VectorXi> KKTClashes,
                                              std::vector<std::mutex>& KKTLocks) const {
      this->function.constraints_jacobian_adjointgradient(
          X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, this->index_data);
    }

    /*
    Interface for calling the underlying type erased function's
    .constraints_jacobian_adjointgradient_adjointhessian method. Passes the arguments from PSIOPT and
    NonLinearProgram as well as the indexing data struct to the underlying vector function.
    */
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X,
        ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX,
        EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations,
        EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks) const {
      this->function.constraints_jacobian_adjointgradient_adjointhessian(
          X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, this->index_data);
    }
  };

}  // namespace ASSET

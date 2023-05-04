/*
File Name: ConstraintFunction.h

File Description: Implements the SolverFunctionBase class which is the base class to
ConstraintFunction and ObjectiveFunction. Holds an Constraint/ObjectiveInterface type erasure class and
SolverIndexingData struct. Defines methods for the function to request and reserve KKT and RHS space from
the solver, and passes relevant arguments to the underlying type erased function or index data structure.
The two Derived classes ( Constraint/ObjectiveInterface) then define the rest of the interface
to the type-erased functions constraints and objective methods.

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

#include "VectorFunctions/FunctionalFlags.h"
#include "VectorFunctions/IndexingData.h"
#include "VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h"
#include "pch.h"

namespace ASSET {

  template<class FuncType>
  struct SolverFunctionBase {
    using MatrixXi = Eigen::MatrixXi;
    using VectorXi = Eigen::VectorXi;

    FuncType function;
    SolverIndexingData index_data;
    int ThreadMode = ThreadingFlags::ByApplication;

    SolverFunctionBase() {
    }

    void print_data() {
      using std::cout;
      using std::endl;

      cout << "Name: " << this->function.name() << endl << endl;
      cout << "Input  Rows:" << this->function.IRows() << endl << endl;
      cout << "Output Rows:" << this->function.ORows() << endl << endl;
      cout << "Thread Policy:" << ThreadMode << endl << endl;

      cout << "Vindex: " << endl << this->index_data.getVindex() << endl << endl;
      if (this->index_data.cindex_init) {
        cout << "Cindex: " << endl << this->index_data.getCindex() << endl << endl;
      }
    }

    int numKKTEles(bool dojac, bool dohess) {
      return this->function.numKKTEles(dojac, dohess) * this->index_data.NumAppl();
    }
    int numConEles() const {
      return this->function.ORows() * this->index_data.NumAppl();
    }
    int numGradEles() const {
      return this->function.IRows() * this->index_data.NumAppl();
    }
    int getThreadMode() const {
      return this->ThreadMode;
    }
    void getKKTSpace(EigenRef<VectorXi> KKTrows,
                     EigenRef<VectorXi> KKTcols,
                     int& freeloc,
                     int conoffset,
                     bool dojac,
                     bool dohess) {
      this->function.getKKTSpace(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, this->index_data);
    }
    void getGradientSpace(EigenRef<VectorXi> GXrows, int& freeloc) {
      this->index_data.getGradientSpace(GXrows, freeloc);
    }
    void getConstraintSpace(EigenRef<VectorXi> FXrows, int& freeloc) {
      this->index_data.getConstraintSpace(FXrows, freeloc);
    }
  };

}  // namespace ASSET

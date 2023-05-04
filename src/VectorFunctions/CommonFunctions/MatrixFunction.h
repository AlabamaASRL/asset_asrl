#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<int MRows, int MCols>
  struct MatrixRowsCols;

  template<class Func, int MRows, int MCols, int MMajor>
  struct MatrixFunctionView : Func, MatrixRowsCols<MRows, MCols> {
    static const int Major = MMajor;
    static const int MROWS = MRows;
    static const int MCOLS = MCols;

    MatrixFunctionView(Func f, int rows, int cols) : Func(f), MatrixRowsCols<MRows, MCols>(rows, cols) {
      // make sure row col consistent with orows
    }
  };

  template<class Func1, class... Funcs>
  struct ColMajorMatrix : MatrixFunctionView<StackedOutputs<Func1, Funcs...>,
                                             Func1::ORC,
                                             1 + sizeof...(Funcs),
                                             Eigen::ColMajor> {
    using Base = MatrixFunctionView<StackedOutputs<Func1, Funcs...>,
                                    Func1::ORC,
                                    1 + sizeof...(Funcs),
                                    Eigen::ColMajor>;
    ColMajorMatrix(Func1 f1, Funcs... fs)
        : Base(StackedOutputs {f1, fs...}, f1.ORows(), 1 + sizeof...(Funcs)) {
    }
  };

  template<class Func1, class... Funcs>
  struct RowMajorMatrix : MatrixFunctionView<StackedOutputs<Func1, Funcs...>,
                                             1 + sizeof...(Funcs),
                                             Func1::ORC,
                                             Eigen::RowMajor> {
    using Base = MatrixFunctionView<StackedOutputs<Func1, Funcs...>,
                                    1 + sizeof...(Funcs),
                                    Func1::ORC,
                                    Eigen::RowMajor>;
    RowMajorMatrix(Func1 f1, Funcs... fs)
        : Base(StackedOutputs {f1, fs...}, 1 + sizeof...(Funcs), f1.ORows()) {
    }
  };

  template<int MRows, int MCols>
  struct MatrixRowsCols {
    static const int MatrixRows = MRows;
    static const int MatrixCols = MCols;

    MatrixRowsCols(int rows, int cols) {
    }
  };

  template<>
  struct MatrixRowsCols<-1, -1> {
    int MatrixRows = 0;
    int MatrixCols = 0;

    MatrixRowsCols(int rows, int cols) : MatrixRows(rows), MatrixCols(cols) {
    }
  };

  template<int MCols>
  struct MatrixRowsCols<-1, MCols> {
    int MatrixRows = 0;
    static const int MatrixCols = MCols;
    MatrixRowsCols(int rows, int cols) : MatrixRows(rows) {
    }
  };

  template<int MRows>
  struct MatrixRowsCols<MRows, -1> {
    static const int MatrixRows = MRows;
    int MatrixCols = 0;
    MatrixRowsCols(int rows, int cols) : MatrixCols(cols) {
    }
  };


}  // namespace ASSET

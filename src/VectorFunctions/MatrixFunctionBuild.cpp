#include "ASSET_VectorFunctions.h"

namespace ASSET {


  void MatrixFunctionBuild(py::module& m) {

    using Gen = GenericFunction<-1, -1>;
    using Func = GenericFunction<-1, -1>;

    using colmattype = MatrixFunctionView<Func, -1, -1, Eigen::ColMajor>;
    using colvectype = MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>;
    using rowmattype = MatrixFunctionView<Func, -1, -1, Eigen::RowMajor>;
    using rowvectype = MatrixFunctionView<Func, 1, -1, Eigen::RowMajor>;


    auto ColMat = py::class_<colmattype>(m, "ColMatrix");


    ColMat.def(py::init<Func, int, int>());
    ColMat.def(py::init([](const std::vector<Func>& colfuns) {
      int cols = colfuns.size();
      if (cols == 0) {
        throw std::invalid_argument("List must contain at least one function.");
      }
      int rows = colfuns[0].ORows();
      for (auto& fun: colfuns)
        if (fun.ORows() != rows)
          throw std::invalid_argument("Column Functions must have same output size");
      auto tmp = DynamicStack(colfuns);
      return std::make_unique<colmattype>(tmp, rows, cols);
    }));

    ColMat.def("__mul__", [](const colmattype& m1, const colmattype& m2) {
      auto tmp = MatrixFunctionProduct<colmattype, colmattype>(m1, m2);
      return colmattype(tmp, m1.MatrixRows, m2.MatrixCols);
    });

    ColMat.def("__mul__", [](const colmattype& m1, const rowmattype& m2) {
      auto tmp = MatrixFunctionProduct<colmattype, rowmattype>(m1, m2);
      return colmattype(tmp, m1.MatrixRows, m2.MatrixCols);
    });

    ColMat.def("__mul__", [](const colmattype& m1, double scale) {
      return colmattype(m1 * scale, m1.MatrixRows, m1.MatrixCols);
    });
    ColMat.def("__rmul__", [](const colmattype& m1, double scale) {
      return colmattype(m1 * scale, m1.MatrixRows, m1.MatrixCols);
    });
    ColMat.def("__add__", [](const colmattype& m1, const Eigen::MatrixXd& mshift) {
      if (m1.MatrixRows != mshift.rows() || m1.MatrixCols != mshift.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions to be added.");
      }
      Eigen::VectorXd v = mshift.reshaped(mshift.rows() * mshift.cols(), 1);
      return colmattype(m1 + v, m1.MatrixRows, m1.MatrixCols);
    });
    ColMat.def("__radd__", [](const colmattype& m1, const Eigen::MatrixXd& mshift) {
      if (m1.MatrixRows != mshift.rows() || m1.MatrixCols != mshift.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions to be added.");
      }
      Eigen::VectorXd v = mshift.reshaped(mshift.rows() * mshift.cols(), 1);
      return colmattype(m1 + v, m1.MatrixRows, m1.MatrixCols);
    });

    ColMat.attr("__array_ufunc__") = py::none();

    ColMat.def("__add__", [](const colmattype& m1, const colmattype& m2) {
      if (m1.MatrixRows != m2.MatrixRows || m1.MatrixCols != m2.MatrixCols) {
        throw std::invalid_argument("Matrices must have the same dimensions to be added.");
      }

      return colmattype(m1 + m2, m1.MatrixRows, m1.MatrixCols);
    });

    ColMat.def("inverse", [](const colmattype& m1) {
      if (m1.MatrixRows != m1.MatrixCols) {
        throw std::invalid_argument("Matrix must be square to be invertible");
      }

      int size = m1.MatrixRows;

      GenericFunction<-1, -1> invfunc;

      if (size == 2) {
        invfunc = MatrixInverse<2, Eigen::ColMajor>(size);
      } else if (size == 3) {
        invfunc = MatrixInverse<3, Eigen::ColMajor>(size);
      } else {
        invfunc = MatrixInverse<-1, Eigen::ColMajor>(size);
      }

      GenericFunction<-1, -1> minv(invfunc.eval(m1));


      return colmattype(minv, size, size);
    });

    ColMat.def("vf", [](const colmattype& m) { return GenericFunction<-1, -1>(m); });

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    auto RowMat = py::class_<rowmattype>(m, "RowMatrix");

    RowMat.def(py::init<Func, int, int>());
    RowMat.def(py::init([](const std::vector<Func>& rowfuns) {
      int rows = rowfuns.size();

      if (rows == 0) {
        throw std::invalid_argument("List must contain at least one function.");
      }
      int cols = rowfuns[0].ORows();
      for (auto& fun: rowfuns)
        if (fun.ORows() != cols)
          throw std::invalid_argument("Row Functions must have same output size");
      auto tmp = DynamicStack(rowfuns);
      return std::make_unique<rowmattype>(tmp, rows, cols);
    }));

    RowMat.def("__mul__", [](const rowmattype& m1, const colmattype& m2) {
      auto tmp = MatrixFunctionProduct<rowmattype, colmattype>(m1, m2);
      return colmattype(tmp, m1.MatrixRows, m2.MatrixCols);
    });

    RowMat.def("__mul__", [](const rowmattype& m1, const rowmattype& m2) {
      auto tmp = MatrixFunctionProduct<rowmattype, rowmattype>(m1, m2);
      return colmattype(tmp, m1.MatrixRows, m2.MatrixCols);
    });


    RowMat.def("__mul__", [](const rowmattype& m1, double scale) {
      return rowmattype(m1 * scale, m1.MatrixRows, m1.MatrixCols);
    });
    RowMat.def("__rmul__", [](const rowmattype& m1, double scale) {
      return rowmattype(m1 * scale, m1.MatrixRows, m1.MatrixCols);
    });

    RowMat.def("__add__", [](const rowmattype& m1, const rowmattype& m2) {
      if (m1.MatrixRows != m2.MatrixRows || m1.MatrixCols != m2.MatrixCols) {
        throw std::invalid_argument("Matrices must have the same dimensions to be added.");
      }

      return rowmattype(m1 + m2, m1.MatrixRows, m1.MatrixCols);
    });

    RowMat.attr("__array_ufunc__") = py::none();

    RowMat.def("__add__", [](const rowmattype& m1, const Eigen::MatrixXd& mshift) {
      if (m1.MatrixRows != mshift.rows() || m1.MatrixCols != mshift.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions to be added.");
      }
      Eigen::MatrixXd tmp = mshift.transpose();
      Eigen::VectorXd v = tmp.reshaped(mshift.rows() * mshift.cols(), 1);
      return rowmattype(m1 + v, m1.MatrixRows, m1.MatrixCols);
    });
    RowMat.def("__radd__", [](const rowmattype& m1, const Eigen::MatrixXd& mshift) {
      if (m1.MatrixRows != mshift.rows() || m1.MatrixCols != mshift.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions to be added.");
      }
      Eigen::MatrixXd tmp = mshift.transpose();
      Eigen::VectorXd v = tmp.reshaped(mshift.rows() * mshift.cols(), 1);
      return rowmattype(m1 + v, m1.MatrixRows, m1.MatrixCols);
    });

    RowMat.def("inverse", [](const rowmattype& m1) {
      if (m1.MatrixRows != m1.MatrixCols) {
        throw std::invalid_argument("Matrix must be square to be invertible.");
      }

      int size = m1.MatrixRows;

      GenericFunction<-1, -1> invfunc;

      if (size == 2) {
        invfunc = MatrixInverse<2, Eigen::RowMajor>(size);
      } else if (size == 3) {
        invfunc = MatrixInverse<3, Eigen::RowMajor>(size);
      } else {
        invfunc = MatrixInverse<-1, Eigen::RowMajor>(size);
      }

      GenericFunction<-1, -1> minv(invfunc.eval(m1));


      return rowmattype(minv, size, size);
    });

    RowMat.def("vf", [](const rowmattype& m) { return GenericFunction<-1, -1>(m); });


    /*
    These two must come last.
    */

    ColMat.def("__mul__", [](const colmattype& m1, const Func& m2) {
      auto tmp = MatrixFunctionProduct<colmattype, colvectype>(m1, colvectype(m2, m2.ORows(), 1));
      return GenericFunction<-1, -1>(tmp);  // Result must be vector
    });

    RowMat.def("__mul__", [](const rowmattype& m1, const Func& m2) {
      auto tmp = MatrixFunctionProduct<rowmattype, colvectype>(m1, colvectype(m2, m2.ORows(), 1));
      return GenericFunction<-1, -1>(tmp);
    });

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////


    m.def("matmul", [](const colmattype& m1, const colmattype& m2) {
      auto tmp = MatrixFunctionProduct<colmattype, colmattype>(m1, m2);
      return colmattype(tmp, m1.MatrixRows, m2.MatrixCols);
    });

    m.def("matmul", [](const colmattype& m1, const rowmattype& m2) {
      auto tmp = MatrixFunctionProduct<colmattype, rowmattype>(m1, m2);
      return colmattype(tmp, m1.MatrixRows, m2.MatrixCols);
    });

    m.def("matmul", [](const colmattype& m1, const Func& m2) {
      auto tmp = MatrixFunctionProduct<colmattype, colvectype>(m1, colvectype(m2, m2.ORows(), 1));
      return GenericFunction<-1, -1>(tmp);
    });

    m.def("matmul", [](const colmattype& m1, const Eigen::VectorXd& v) {
      auto m2 = Constant<-1, -1>(m1.IRows(), v);
      auto tmp = MatrixFunctionProduct<colmattype, colvectype>(m1, colvectype(m2, m2.ORows(), 1));
      return GenericFunction<-1, -1>(tmp);
    });

    m.def("matmul", [](const rowmattype& m1, const colmattype& m2) {
      auto tmp = MatrixFunctionProduct<rowmattype, colmattype>(m1, m2);
      return colmattype(tmp, m1.MatrixRows, m2.MatrixCols);
    });

    m.def("matmul", [](const rowmattype& m1, const rowmattype& m2) {
      auto tmp = MatrixFunctionProduct<rowmattype, rowmattype>(m1, m2);
      return colmattype(tmp, m1.MatrixRows, m2.MatrixCols);
    });

    m.def("matmul", [](const rowmattype& m1, const Func& m2) {
      auto tmp = MatrixFunctionProduct<rowmattype, colvectype>(m1, colvectype(m2, m2.ORows(), 1));
      return GenericFunction<-1, -1>(tmp);
    });

    m.def("matmul", [](const rowmattype& m1, const Eigen::VectorXd& v) {
      auto m2 = Constant<-1, -1>(m1.IRows(), v);
      auto tmp = MatrixFunctionProduct<rowmattype, colvectype>(m1, colvectype(m2, m2.ORows(), 1));
      return GenericFunction<-1, -1>(tmp);
    });


    m.def("matmul", [](const Eigen::Matrix<double, 2, 2>& mat, const Gen& vec) {
      return Gen(MatrixScaled<Gen, 2>(vec, mat));
    });

    m.def("matmul", [](const Eigen::Matrix<double, 3, 3>& mat, const Gen& vec) {
      return Gen(MatrixScaled<Gen, 3>(vec, mat));
    });

    m.def("matmul",
          [](const Eigen::MatrixXd& mat, const Gen& vec) { return Gen(MatrixScaled<Gen, -1>(vec, mat)); });
  }


}  // namespace ASSET
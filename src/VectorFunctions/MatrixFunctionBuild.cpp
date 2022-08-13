#include "ASSET_VectorFunctions.h"

namespace ASSET {
    

    

    void MatrixFunctionBuild(py::module & m) {

        using Gen = GenericFunction<-1, -1>;
        using Func = GenericFunction<-1, -1>;

        using colmattype = MatrixFunctionView<Func, -1, -1, Eigen::ColMajor>;
        using colvectype = MatrixFunctionView<Func, -1, 1, Eigen::ColMajor>;
        using rowmattype = MatrixFunctionView<Func, -1, -1, Eigen::RowMajor>;
        using rowvectype = MatrixFunctionView<Func, 1, -1, Eigen::RowMajor>;

        
        auto cm = py::class_<colmattype, Func>(m, "ColMatrix");
        cm.def(py::init<Func, int, int>());
        cm.def(py::init([](const std::vector<Func>& colfuns) {
            int cols = colfuns.size();
            int rows = colfuns[0].ORows();
            for (auto& fun : colfuns)
                if (fun.ORows() != rows)
                    throw std::invalid_argument(
                        "Column Functions must have same output size");
            auto tmp = DynamicStack(colfuns);
            return std::make_unique<colmattype>(tmp, rows, cols);
            }));

        cm.def("__mul__", [](const colmattype& m1, const colmattype& m2) {
            auto tmp = MatrixFunctionProduct<colmattype, colmattype>(m1, m2);
            return GenericFunction<-1, -1>(tmp);
            });

        cm.def("inverse", [](const colmattype& m1) {
           
            if (m1.MatrixRows != m1.MatrixCols) {
                throw std::invalid_argument("Matrix must be square to be invertible");
            }

            int size = m1.MatrixRows;

            GenericFunction<-1, -1> invfunc;

            if (size == 2) {
                invfunc = MatrixInverse<2,  Eigen::ColMajor>(size);
            }
            else if (size == 3) {
                invfunc = MatrixInverse<3,  Eigen::ColMajor>(size);
            }
            else {
                invfunc = MatrixInverse<-1, Eigen::ColMajor>(size);
            }
            
            GenericFunction<-1, -1> minv(invfunc.eval(m1));


            return colmattype(minv,size,size);
            });

        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////

        auto rm = py::class_<rowmattype, Func>(m, "RowMatrix");
        rm.def(py::init<Func, int, int>());
        rm.def(py::init([](const std::vector<Func>& rowfuns) {
            int rows = rowfuns.size();
            int cols = rowfuns[0].ORows();
            for (auto& fun : rowfuns)
                if (fun.ORows() != cols)
                    throw std::invalid_argument("Row Functions must have same output size");
            auto tmp = DynamicStack(rowfuns);
            return std::make_unique<rowmattype>(tmp, rows, cols);
            }));

        rm.def("__mul__", [](const rowmattype& m1, const colmattype& m2) {
            auto tmp = MatrixFunctionProduct<rowmattype, colmattype>(m1, m2);
            return GenericFunction<-1, -1>(tmp);
            });

        rm.def("inverse", [](const rowmattype& m1) {

            if (m1.MatrixRows != m1.MatrixCols) {
                throw std::invalid_argument("Matrix must be square to be invertible");
            }

            int size = m1.MatrixRows;

            GenericFunction<-1, -1> invfunc;

            if (size == 2) {
                invfunc = MatrixInverse<2, Eigen::RowMajor>(size);
            }
            else if (size == 3) {
                invfunc = MatrixInverse<3, Eigen::RowMajor>(size);
            }
            else {
                invfunc = MatrixInverse<-1, Eigen::RowMajor>(size);
            }

            GenericFunction<-1, -1> minv(invfunc.eval(m1));


            return rowmattype(minv, size, size);
            });

        

        //////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////

        cm.def("__mul__", [](const colmattype& m1, const rowmattype& m2) {
            //throw std::invalid_argument("ColumnMajor on RowMajor Matmul not implemented yet");
            fmt::print("Sss");
            auto tmp = MatrixFunctionProduct<colmattype,rowmattype>(m1, m2);


            return GenericFunction<-1, -1>(tmp);
            });

        rm.def("__mul__", [](const rowmattype& m1, const rowmattype& m2) {
            //throw std::invalid_argument("RowMajor on RowMajor Matmul not implemented yed");

            auto tmp = MatrixFunctionProduct<rowmattype, rowmattype>(m1, m2);
            return GenericFunction<-1, -1>(tmp);
            });



        cm.def("__mul__", [](const colmattype& m1, const Func& m2) {
            auto tmp = MatrixFunctionProduct<colmattype, colvectype>(
                m1, colvectype(m2, m2.ORows(), 1));
            return GenericFunction<-1, -1>(tmp);
            });
        rm.def("__mul__", [](const rowmattype& m1, const Func& m2) {
            auto tmp = MatrixFunctionProduct<rowmattype, colvectype>(
                m1, colvectype(m2, m2.ORows(), 1));
            return GenericFunction<-1, -1>(tmp);
            });



        m.def("matmul", [](const Eigen::Matrix<double, 2, 2>& mat, const Gen& vec) {
            
            return Gen(MatrixScaled<Gen, 2>(vec, mat));
            });

        m.def("matmul", [](const Eigen::Matrix<double, 3, 3>& mat, const Gen& vec) {
            
            return Gen(MatrixScaled<Gen, 3>(vec, mat));
            });

        m.def("matmul", [](const Eigen::MatrixXd& mat, const Gen& vec) {
           
            return Gen(MatrixScaled<Gen, -1>(vec, mat));
            });

        

    }


}  // na
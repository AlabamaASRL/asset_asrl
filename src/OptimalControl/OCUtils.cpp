#include "OCUtils.h"


namespace ASSET {


    void OCUtilsBuild(py::module& oc)
    {
        oc.def("jump_function", &jump_function);
        oc.def("jump_function_mmod", &jump_function_mmod);

    }

    Eigen::VectorXd jump_function(const Eigen::VectorXd& tsin, const Eigen::VectorXd& usin, const Eigen::VectorXd& tsout, int m) {

        int size = tsin.size();
        m = std::min(size, m);

        std::vector < std::array < double, 2>> tupars;

        double fact = 1;
        for (int i = 1; i <= (m - 1); i++)
            fact = fact * i;

        Eigen::VectorXd jmp(tsout.size());

        for (int i = 0, start = 0; i < tsout.size(); i++) {

            double t = tsout[i];

            auto it = std::upper_bound(tsin.cbegin() + start, tsin.cend(), t);
            int elem = int(it - tsin.begin());
            start = elem - 1;
            start = std::clamp(start, 0, size - 1);

            int j0 = std::max(0, elem - m - 1);
            int jf = std::min(elem + m + 1, size - 1);
            tupars.resize(jf - j0 + 1);

            for (int j = j0, k = 0; j <= jf; j++, k++) {
                tupars[k] = { tsin[j],usin[j] };
            }

            //fmt::print("{0},\n", tupars.size());

            std::sort(tupars.begin(), tupars.end(),
                [t](auto t1, auto t2) {
                    return std::abs(t1[0] - t) < std::abs(t2[0] - t);
                });


            double q = 0;
            double fs = 0;


            for (int j = 0; j < m; j++) {
                double cj = fact;

                for (int k = 0; k < m; k++) {
                    if (k != j) {
                        cj *= 1.0 / (tupars[j][0] - tupars[k][0]);
                    }
                }
                fs += cj * tupars[j][1];

                if (tupars[j][0] > t) {

                    q += cj;
                    // fmt::print("{0},\n", q);

                }
            }


            jmp[i] = fs / q;


        }

        return jmp;

    }


    Eigen::VectorXd jump_function_mmod(const Eigen::VectorXd& tsin, const Eigen::VectorXd& usin, const Eigen::VectorXd& tsout, Eigen::VectorXi ms) {
        Eigen::VectorXd jf(tsout.size());
        Eigen::MatrixXd js(tsout.size(), ms.size());
        for (int i = 0; i < ms.size(); i++) {
            js.col(i) = jump_function(tsin, usin, tsout, ms[i]).cwiseAbs();
        }
        jf = js.rowwise().minCoeff();
        return jf;
    }


}


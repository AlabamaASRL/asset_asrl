#include "LambertSolvers.h"


void ASSET::LambertSolversBuild(FunctionRegistry& reg, py::module& m) {


  m.def("lambert_izzo",
        [](const Vector3<double>& R1, const Vector3<double>& R2, double dt, double mu, bool longway) {
          return lambert_izzo(R1, R2, dt, mu, longway);
        });

  m.def("lambert_izzo",
        [](const Vector3<double>& R1,
           const Vector3<double>& R2,
           double dt,
           double mu,
           bool longway,
           int Nrevs,
           bool rightbranch) { return lambert_izzo(R1, R2, dt, mu, longway, Nrevs, rightbranch); });

  m.def("lambert_izzo_multirev",
        [](const Vector3<double>& R1,
           const Vector3<double>& R2,
           double dt,
           double mu,
           bool longway,
           int Nrevs,
           bool rightbranch) { return lambert_izzo(R1, R2, dt, mu, longway, Nrevs, rightbranch); });


  using NumpyMat = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;


  m.def("lambert_izzo",
        [](ConstEigenRef<NumpyMat> R1s,
           ConstEigenRef<NumpyMat> R2s,
           ConstEigenRef<VectorX<double>> dts,
           double mu,
           const std::vector<bool>& longways,
           EigenRef<NumpyMat> V1s,
           EigenRef<NumpyMat> V2s,
           int axis,
           bool vectorize) {
          using SuperScalar = Eigen::Array<double, 8, 1>;
          constexpr int vsize = SuperScalar::SizeAtCompileTime;
          int NumCalls;
          if (axis == 0) {
            NumCalls = R1s.cols();
          } else {
            NumCalls = R1s.rows();
          }
          int Packs = vectorize ? NumCalls / vsize : 0;

          Eigen::VectorXi exitcodes(NumCalls);
          Vector3<SuperScalar> R1ss;
          Vector3<SuperScalar> R2ss;
          std::array<bool, vsize> lwss;
          std::array<int, vsize> ecodess;

          Vector3<SuperScalar> V1ss;
          Vector3<SuperScalar> V2ss;

          for (int i = 0; i < Packs; i++) {
            int V = i * vsize;
            if (axis == 0) {
              R1ss[0] = R1s.row(0).segment<vsize>(V, vsize).transpose();
              R1ss[1] = R1s.row(1).segment<vsize>(V, vsize).transpose();
              R1ss[2] = R1s.row(2).segment<vsize>(V, vsize).transpose();

              R2ss[0] = R2s.row(0).segment<vsize>(V, vsize).transpose();
              R2ss[1] = R2s.row(1).segment<vsize>(V, vsize).transpose();
              R2ss[2] = R2s.row(2).segment<vsize>(V, vsize).transpose();
            } else {

              R1ss[0] = R1s.col(0).segment<vsize>(V, vsize);
              R1ss[1] = R1s.col(1).segment<vsize>(V, vsize);
              R1ss[2] = R1s.col(2).segment<vsize>(V, vsize);

              R2ss[0] = R2s.col(0).segment<vsize>(V, vsize);
              R2ss[1] = R2s.col(1).segment<vsize>(V, vsize);
              R2ss[2] = R2s.col(2).segment<vsize>(V, vsize);
            }


            SuperScalar dtss = dts.segment<vsize>(V);
            for (int j = 0; j < vsize; j++) {
              lwss[j] = longways[V + j];
            }

            lambert_izzo_impl(R1ss, R2ss, dtss, mu, lwss, 0, false, V1ss, V2ss, ecodess);


            if (axis == 0) {
              V1s.row(0).segment<vsize>(V, vsize) = V1ss[0].transpose();
              V1s.row(1).segment<vsize>(V, vsize) = V1ss[1].transpose();
              V1s.row(2).segment<vsize>(V, vsize) = V1ss[2].transpose();

              V2s.row(0).segment<vsize>(V, vsize) = V2ss[0].transpose();
              V2s.row(1).segment<vsize>(V, vsize) = V2ss[1].transpose();
              V2s.row(2).segment<vsize>(V, vsize) = V2ss[2].transpose();
            } else {
              V1s.col(0).segment<vsize>(V, vsize) = V1ss[0];
              V1s.col(1).segment<vsize>(V, vsize) = V1ss[1];
              V1s.col(2).segment<vsize>(V, vsize) = V1ss[2];

              V2s.col(0).segment<vsize>(V, vsize) = V2ss[0];
              V2s.col(1).segment<vsize>(V, vsize) = V2ss[1];
              V2s.col(2).segment<vsize>(V, vsize) = V2ss[2];
            }

            for (int j = 0; j < vsize; j++) {
              exitcodes[V + j] = ecodess[j];
            }
          }

          Vector3<double> R1;
          Vector3<double> R2;
          Vector3<double> V1;
          Vector3<double> V2;

          for (int i = Packs * vsize; i < NumCalls; i++) {

            if (axis == 0) {
              R1 = R1s.col(i);
              R2 = R2s.col(i);
            } else {
              R1 = R1s.row(i);
              R2 = R2s.row(i);
            }

            double dt = dts[i];
            bool longway = longways[i];
            int excode;

            lambert_izzo_impl(R1, R2, dt, mu, longway, 0, false, V1, V2, excode);

            exitcodes[i] = excode;

            if (axis == 0) {
              V1s.col(i) = V1;
              V2s.col(i) = V2;
            } else {
              V1s.row(i) = V1.transpose();
              V2s.row(i) = V2.transpose();
            }
          }

          return exitcodes;
        });
}

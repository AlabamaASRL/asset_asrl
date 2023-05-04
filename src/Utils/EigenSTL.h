#pragma once
#include <Eigen/Core>
#include <vector>


namespace ASSET {

  template<class Scalar, int Size>
  std::vector<Scalar> eigenvector_to_stdvector(const Eigen::Matrix<Scalar, Size, 1>& eigvec) {
    int size = eigvec.size();
    std::vector<double> stdvec(size);
    for (int i = 0; i < size; i++) {
      stdvec[i] = eigvec[i];
    }
    return stdvec;
  }

  template<class Scalar>
  Eigen::Matrix<Scalar, -1, 1> stdvector_to_eigenvector(const std::vector<Scalar>& stdvec) {
    int size = stdvec.size();
    Eigen::Matrix<Scalar, -1, 1> eigvec(size);
    for (int i = 0; i < size; i++) {
      eigvec[i] = stdvec[i];
    }
    return eigvec;
  }


}  // namespace ASSET
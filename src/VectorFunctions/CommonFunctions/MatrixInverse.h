#pragma once

#include "VectorFunction.h"


namespace ASSET {

  template<int Size, int Major>
  struct MatrixInverse : VectorFunction<MatrixInverse<Size, Major>,
                                        SZ_PROD<Size, Size>::value,
                                        SZ_PROD<Size, Size>::value,
                                        Analytic,
                                        Analytic> {

    using Base = VectorFunction<MatrixInverse<Size, Major>,
                                SZ_PROD<Size, Size>::value,
                                SZ_PROD<Size, Size>::value,
                                Analytic,
                                Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    static const bool IsVectorizable = false;
    template<class Scalar>
    using Mat = Eigen::Matrix<Scalar, Size, Size>;

    int size;


    MatrixInverse() {};
    MatrixInverse(int size) : size(size) {
      this->setIORows(size * size, size * size);
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      auto Impl = [&](auto& M, auto& MInv) {
        for (int i = 0; i < this->size; i++) {
          if constexpr (Major == Eigen::ColMajor) {
            M.col(i) = x.template segment<Size>(i * this->size, this->size);
          } else {
            M.row(i) = x.template segment<Size>(i * this->size, this->size);
          }
        }
        MInv = M.inverse();
        for (int i = 0; i < this->size; i++) {
          if constexpr (Major == Eigen::ColMajor) {
            fx.template segment<Size>(i * this->size, this->size) = MInv.col(i);
          } else {
            fx.template segment<Size>(i * this->size, this->size) = MInv.row(i);
          }
        }
      };

      MemoryManager::allocate_run(this->size,
                                  Impl,
                                  TempSpec<Mat<Scalar>>(this->size, this->size),
                                  TempSpec<Mat<Scalar>>(this->size, this->size));
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();


      auto Impl = [&](auto& M, auto& MInv) {
        for (int i = 0; i < this->size; i++) {
          if constexpr (Major == Eigen::ColMajor) {
            M.col(i) = x.template segment<Size>(i * this->size, this->size);
          } else {
            M.row(i) = x.template segment<Size>(i * this->size, this->size);
          }
        }
        MInv = M.inverse();
        for (int i = 0; i < this->size; i++) {
          if constexpr (Major == Eigen::ColMajor) {
            fx.template segment<Size>(i * this->size, this->size) = MInv.col(i);
          } else {
            fx.template segment<Size>(i * this->size, this->size) = MInv.row(i);
          }
        }

        for (int i = 0; i < this->size; i++) {
          for (int j = 0; j < this->size; j++) {
            M = MInv.col(j) * MInv.row(i);
            for (int k = 0; k < this->size; k++) {
              if constexpr (Major == Eigen::ColMajor) {
                jx.col(i * this->size + j).template segment<Size>(k * this->size, this->size) = -M.col(k);
              } else {
                jx.col(j * this->size + i).template segment<Size>(k * this->size, this->size) = -M.row(k);
              }
            }
          }
        }
      };

      MemoryManager::allocate_run(this->size,
                                  Impl,
                                  TempSpec<Mat<Scalar>>(this->size, this->size),
                                  TempSpec<Mat<Scalar>>(this->size, this->size));
    }
    template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_,
        ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_,
        ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      auto Impl = [&](auto& M, auto& MInv, auto& Madj) {
        for (int i = 0; i < this->size; i++) {

          if constexpr (Major == Eigen::ColMajor) {
            M.col(i) = x.template segment<Size>(i * this->size, this->size);
          } else {
            M.row(i) = x.template segment<Size>(i * this->size, this->size);
          }
        }

        MInv = M.inverse();

        for (int i = 0; i < this->size; i++) {

          if constexpr (Major == Eigen::ColMajor) {
            fx.template segment<Size>(i * this->size, this->size) = MInv.col(i);
          } else {
            fx.template segment<Size>(i * this->size, this->size) = MInv.row(i);
          }
        }

        for (int i = 0; i < size; i++) {
          if constexpr (Major == Eigen::ColMajor) {
            Madj.col(i) = adjvars.template segment<Size>(i * size, size);
          } else {
            Madj.row(i) = adjvars.template segment<Size>(i * size, size);
          }
        }

        for (int i = 0; i < this->size; i++) {

          for (int j = 0; j < this->size; j++) {
            M = MInv.col(j) * MInv.row(i);
            for (int k = 0; k < this->size; k++) {
              if constexpr (Major == Eigen::ColMajor) {
                jx.col(i * this->size + j).template segment<Size>(k * this->size, this->size) = -M.col(k);
              } else {
                jx.col(j * this->size + i).template segment<Size>(k * this->size, this->size) = -M.row(k);
              }
            }
            for (int k = 0; k < this->size; k++) {
              for (int l = 0; l < this->size; l++) {

                if constexpr (Major == Eigen::ColMajor) {
                  Scalar hv = ((M.col(l) * MInv.row(k)).cwiseProduct(Madj)).sum();
                  adjhess(i * this->size + j, k * this->size + l) += hv;
                  adjhess(k * this->size + l, i * this->size + j) += hv;
                } else {
                  Scalar hv = ((MInv.col(l) * M.row(k)).cwiseProduct(Madj)).sum();
                  adjhess(j * this->size + i, l * this->size + k) += hv;
                  adjhess(l * this->size + k, j * this->size + i) += hv;
                }
              }
            }
          }
        }
        adjgrad.noalias() = (adjvars.transpose() * jx_).transpose();
      };

      MemoryManager::allocate_run(this->size,
                                  Impl,
                                  TempSpec<Mat<Scalar>>(this->size, this->size),
                                  TempSpec<Mat<Scalar>>(this->size, this->size),
                                  TempSpec<Mat<Scalar>>(this->size, this->size));
    }
  };
}  // namespace ASSET
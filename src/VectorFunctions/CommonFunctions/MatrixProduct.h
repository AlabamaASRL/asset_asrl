#pragma once

#include "VectorFunction.h"

namespace ASSET {


  template<class Derived, class MatFunc1, class MatFunc2>
  struct MatrixFunctionProduct_Impl;


  template<class MatFunc1, class MatFunc2>
  struct MatrixFunctionProduct
      : MatrixFunctionProduct_Impl<MatrixFunctionProduct<MatFunc1, MatFunc2>, MatFunc1, MatFunc2> {
    using Base = MatrixFunctionProduct_Impl<MatrixFunctionProduct<MatFunc1, MatFunc2>, MatFunc1, MatFunc2>;
    using Base::Base;
  };

  template<class Derived, class MatFunc1, class MatFunc2>
  struct MatrixFunctionProduct_Impl : VectorFunction<Derived,
                                                     SZ_MAX<MatFunc1::IRC, MatFunc2::IRC>::value,
                                                     SZ_PROD<MatFunc1::MROWS, MatFunc2::MCOLS>::value,
                                                     Analytic,
                                                     Analytic> {
    using Base = VectorFunction<Derived,
                                SZ_MAX<MatFunc1::IRC, MatFunc2::IRC>::value,
                                SZ_PROD<MatFunc1::MROWS, MatFunc2::MCOLS>::value,
                                Analytic,
                                Analytic>;

    static const int M1Rows = MatFunc1::MROWS;
    static const int M1Cols_M2Rows = MatFunc2::MROWS;
    static const int M2Cols = MatFunc2::MCOLS;

    static const int M1Major = MatFunc1::Major;
    static const int M2Major = MatFunc2::Major;

    int m1rows = 0;
    int m1cols_m2rows = 0;
    int m2cols = 0;

    template<class Scalar>
    using MatrixOne = Eigen::Matrix<Scalar, M1Rows, M1Cols_M2Rows, M1Major>;
    template<class Scalar>
    using MatrixTwo = Eigen::Matrix<Scalar, M1Cols_M2Rows, M2Cols, M2Major>;
    template<class Scalar>
    using MatrixResult = Eigen::Matrix<Scalar, M1Rows, M2Cols>;

    SUB_FUNCTION_IO_TYPES(MatFunc1);
    SUB_FUNCTION_IO_TYPES(MatFunc2);

    MatFunc1 matrix_func1;
    MatFunc2 matrix_func2;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename MatFunc1::INPUT_DOMAIN, typename MatFunc2::INPUT_DOMAIN>;

    static const bool IsVectorizable = MatFunc1::IsVectorizable && MatFunc2::IsVectorizable;
    // static const bool IsVectorizable = false;

    DENSE_FUNCTION_BASE_TYPES(Base);

    MatrixFunctionProduct_Impl() {
    }
    MatrixFunctionProduct_Impl(MatFunc1 mf1, MatFunc2 mf2) : matrix_func1(mf1), matrix_func2(mf2) {
      m1rows = this->matrix_func1.MatrixRows;
      m1cols_m2rows = this->matrix_func2.MatrixRows;
      m2cols = this->matrix_func2.MatrixCols;

      if (this->matrix_func1.MatrixCols != this->matrix_func2.MatrixRows) {
        throw std::invalid_argument("Invalid matrix product. Number of columns in matrix 1 does not match "
                                    "number of rows in matrix 2.");
      }

      this->setIORows(this->matrix_func1.IRows(), m1rows * m2cols);

      this->set_input_domain(this->IRows(), {matrix_func1.input_domain(), matrix_func2.input_domain()});
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      auto Impl = [&](auto& fxm1, auto& fxm2, auto& fxmd) {
        this->matrix_func1.compute(x, fxm1);
        this->matrix_func2.compute(x, fxm2);

        Eigen::Map<const MatrixOne<Scalar>> m1(fxm1.data(), m1rows, m1cols_m2rows);
        Eigen::Map<const MatrixTwo<Scalar>> m2(fxm2.data(), m1cols_m2rows, m2cols);
        Eigen::Map<MatrixResult<Scalar>> fxm(fxmd.data(), m1rows, m2cols);

        fxm.noalias() = m1 * m2;

        fx = fxmd;
      };


      const int o1 = this->matrix_func1.ORows();
      const int o2 = this->matrix_func2.ORows();
      const int orows = this->ORows();

      const int crit_size = std::max({o1, o2, orows});

      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  TempSpec<MatFunc1_Output<Scalar>>(o1, 1),
                                  TempSpec<MatFunc2_Output<Scalar>>(o2, 1),
                                  TempSpec<Output<Scalar>>(orows, 1));
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      ///////////////////////////
      auto Impl = [&](auto& fxm1, auto& jxm1, auto& fxm2, auto& jxm2, auto& fxmd, auto& dvec) {
        this->matrix_func1.compute_jacobian(x, fxm1, jxm1);
        this->matrix_func2.compute_jacobian(x, fxm2, jxm2);
        Eigen::Map<const MatrixOne<Scalar>> m1(fxm1.data(), m1rows, m1cols_m2rows);
        Eigen::Map<const MatrixTwo<Scalar>> m2(fxm2.data(), m1cols_m2rows, m2cols);
        Eigen::Map<MatrixResult<Scalar>> fxm(fxmd.data(), m1rows, m2cols);
        fxm.noalias() = m1 * m2;
        fx = fxmd;

        for (int i = 0; i < m2cols; i++) {
          if constexpr (M1Major == Eigen::ColMajor) {
            for (int j = 0; j < m1cols_m2rows; j++) {
              dvec.setConstant(m2(j, i));
              this->matrix_func1.right_jacobian_domain_product(
                  jx.template middleRows<M1Rows>(m1rows * i, m1rows),
                  dvec.asDiagonal(),
                  jxm1.template middleRows<M1Rows>(m1rows * j, m1rows),
                  PlusEqualsAssignment(),
                  std::bool_constant<false>());
            }
          } else {
            for (int j = 0; j < m1rows; j++) {
              this->matrix_func1.right_jacobian_domain_product(
                  jx.row(m1rows * i + j),
                  m2.col(i).transpose(),
                  jxm1.template middleRows<M1Cols_M2Rows>(m1cols_m2rows * j, m1cols_m2rows),
                  PlusEqualsAssignment(),
                  std::bool_constant<false>());
            }
          }

          if constexpr (M2Major == Eigen::ColMajor) {
            this->matrix_func2.right_jacobian_domain_product(
                jx.template middleRows<M1Rows>(m1rows * i, m1rows),
                m1,
                jxm2.template middleRows<M1Cols_M2Rows>(m1cols_m2rows * i, m1cols_m2rows),
                PlusEqualsAssignment(),
                std::bool_constant<false>());
          }
        }
        if constexpr (M2Major == Eigen::RowMajor) {
          for (int i = 0; i < m1cols_m2rows; i++) {
            dvec = m1.col(i);
            for (int j = 0; j < m2cols; j++) {
              this->matrix_func2.right_jacobian_domain_product(
                  jx.template middleRows<M1Rows>(m1rows * j, m1rows),
                  dvec,
                  jxm2.row(m2cols * i + j),
                  PlusEqualsAssignment(),
                  std::bool_constant<false>());
            }
          }
        }
      };


      const int o1 = this->matrix_func1.ORows();
      const int o2 = this->matrix_func2.ORows();
      const int irows = this->IRows();
      const int orows = this->ORows();

      const int crit_size = std::max({o1, o2, orows, irows, m1rows});

      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  TempSpec<MatFunc1_Output<Scalar>>(o1, 1),
                                  TempSpec<MatFunc1_jacobian<Scalar>>(o1, irows),
                                  TempSpec<MatFunc2_Output<Scalar>>(o2, 1),
                                  TempSpec<MatFunc2_jacobian<Scalar>>(o2, irows),
                                  TempSpec<Output<Scalar>>(orows, 1),
                                  TempSpec<Eigen::Matrix<Scalar, M1Rows, 1>>(m1rows, 1));
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

      ///////////////////////////
      auto Impl = [&](auto& fxm1,
                      auto& adjv1,
                      auto& jxm1,
                      auto& fxm2,
                      auto& adjv2,
                      auto& jxm2,
                      auto& gxm2,
                      auto& hxm2,
                      auto& fxmd,
                      auto& dvec,
                      auto& jttemp) {
        Eigen::Map<MatrixOne<Scalar>> m1(fxm1.data(), m1rows, m1cols_m2rows);
        Eigen::Map<MatrixTwo<Scalar>> m2(fxm2.data(), m1cols_m2rows, m2cols);
        Eigen::Map<MatrixResult<Scalar>> fxm(fxmd.data(), m1rows, m2cols);
        this->matrix_func2.compute(x, fxm2);

        if constexpr (M1Major == Eigen::ColMajor) {
          for (int i = 0; i < m2cols; i++) {
            for (int j = 0; j < m1cols_m2rows; j++) {
              adjv1.template segment<M1Rows>(m1rows * j, m1rows) +=
                  adjvars.template segment<M1Rows>(m1rows * i, m1rows) * m2(j, i);
            }
          }
        } else {
          for (int i = 0; i < m2cols; i++) {
            for (int j = 0; j < m1rows; j++) {
              adjv1.template segment<M1Cols_M2Rows>(m1cols_m2rows * j, m1cols_m2rows) +=
                  adjvars[m1rows * i + j] * m2.col(i);
            }
          }
        }

        this->matrix_func1.compute_jacobian_adjointgradient_adjointhessian(
            x, fxm1, jxm1, adjgrad, adjhess, adjv1);
        //////////////////////////////////////////////

        if constexpr (M2Major == Eigen::ColMajor) {
          for (int i = 0; i < m2cols; i++) {
            adjv2.template segment<M1Cols_M2Rows>(m1cols_m2rows * i, m1cols_m2rows) +=
                m1.transpose() * adjvars.template segment<M1Rows>(m1rows * i, m1rows);
          }
        } else {
          for (int i = 0; i < m1cols_m2rows; i++) {
            for (int j = 0; j < m2cols; j++) {
              adjv2[m2cols * i + j] += m1.col(i).dot(adjvars.template segment<M1Rows>(m1rows * j, m1rows));
            }
          }
        }

        fxm2.setZero();

        this->matrix_func2.compute_jacobian_adjointgradient_adjointhessian(x, fxm2, jxm2, gxm2, hxm2, adjv2);

        this->matrix_func2.accumulate_hessian(adjhess, hxm2, PlusEqualsAssignment());
        this->matrix_func2.accumulate_gradient(adjgrad, gxm2, PlusEqualsAssignment());

        fxm.noalias() = m1 * m2;
        fx = fxmd;

        for (int i = 0; i < m2cols; i++) {
          if constexpr (M1Major == Eigen::ColMajor) {
            for (int j = 0; j < m1cols_m2rows; j++) {
              dvec.setConstant(m2(j, i));
              this->matrix_func1.right_jacobian_domain_product(
                  jx.template middleRows<M1Rows>(m1rows * i, m1rows),
                  dvec.asDiagonal(),
                  jxm1.template middleRows<M1Rows>(m1rows * j, m1rows),
                  PlusEqualsAssignment(),
                  std::bool_constant<false>());
            }
          } else {
            for (int j = 0; j < m1rows; j++) {
              this->matrix_func1.right_jacobian_domain_product(
                  jx.row(m1rows * i + j),
                  m2.col(i).transpose(),
                  jxm1.template middleRows<M1Cols_M2Rows>(m1cols_m2rows * j, m1cols_m2rows),
                  PlusEqualsAssignment(),
                  std::bool_constant<false>());
            }
          }

          if constexpr (M2Major == Eigen::ColMajor) {
            this->matrix_func2.right_jacobian_domain_product(
                jx.template middleRows<M1Rows>(m1rows * i, m1rows),
                m1,
                jxm2.template middleRows<M1Cols_M2Rows>(m1cols_m2rows * i, m1cols_m2rows),
                PlusEqualsAssignment(),
                std::bool_constant<false>());
          }
        }
        if constexpr (M2Major == Eigen::RowMajor) {
          for (int i = 0; i < m1cols_m2rows; i++) {
            dvec = m1.col(i);
            for (int j = 0; j < m2cols; j++) {
              this->matrix_func2.right_jacobian_domain_product(
                  jx.template middleRows<M1Rows>(m1rows * j, m1rows),
                  dvec,
                  jxm2.row(m2cols * i + j),
                  PlusEqualsAssignment(),
                  std::bool_constant<false>());
            }
          }
        }

        ////////////////////////////////////////////////////////////////////
        // typedef typename std::remove_reference<decltype(jxm2)>::type Jac2type;

        // Jac2type jttemp;
        // jttemp.resize(this->matrix_func2.ORows(), this->IRows());

        if constexpr (M2Major == Eigen::ColMajor) {
          if constexpr (M1Major == Eigen::ColMajor) {
            for (int i = 0; i < m2cols; i++) {
              for (int j = 0; j < m1cols_m2rows; j++) {
                this->matrix_func1.right_jacobian_domain_product(
                    jttemp.row(m1cols_m2rows * i + j),
                    adjvars.template segment<M1Rows>(m1rows * i, m1rows).transpose(),
                    jxm1.template middleRows<M1Rows>(m1rows * j, m1rows),
                    PlusEqualsAssignment(),
                    std::bool_constant<false>());
              }
            }
          } else {
            Eigen::Matrix<Scalar, M1Cols_M2Rows, 1> lvec;
            lvec.resize(m1cols_m2rows);
            for (int i = 0; i < m2cols; i++) {
              for (int j = 0; j < m1rows; j++) {
                lvec.setConstant(adjvars.template segment<M1Rows>(m1rows * i, m1rows)[j]);

                this->matrix_func1.right_jacobian_domain_product(
                    jttemp.template middleRows<M1Cols_M2Rows>(m1cols_m2rows * i, m1cols_m2rows),
                    lvec.asDiagonal(),
                    jxm1.template middleRows<M1Cols_M2Rows>(m1cols_m2rows * j, m1cols_m2rows),
                    PlusEqualsAssignment(),
                    std::bool_constant<false>());
              }
            }
          }
        } else {

          if constexpr (M1Major == Eigen::ColMajor) {
            fxmd = adjvars;
            Eigen::Map<const Eigen::Matrix<Scalar, M2Cols, M1Rows, Eigen::RowMajor>> lmat(
                fxmd.data(), m2cols, m1rows);

            for (int i = 0; i < m1cols_m2rows; i++) {
              this->matrix_func1.right_jacobian_domain_product(
                  jttemp.template middleRows<M2Cols>(m2cols * i, m2cols),
                  lmat,
                  jxm1.template middleRows<M1Rows>(m1rows * i, m1rows),
                  PlusEqualsAssignment(),
                  std::bool_constant<false>());
            }
          } else {
            fxmd = adjvars;
            Eigen::Map<const Eigen::Matrix<Scalar, M2Cols, M1Rows, Eigen::RowMajor>> lmat(
                fxmd.data(), m2cols, m1rows);

            for (int i = 0; i < m1rows; i++) {

              for (int j = 0; j < m1cols_m2rows; j++) {

                this->matrix_func1.right_jacobian_domain_product(
                    jttemp.template middleRows<M2Cols>(m2cols * j, m2cols),
                    lmat.col(i),
                    jxm1.row(m1cols_m2rows * i + j),
                    PlusEqualsAssignment(),
                    std::bool_constant<false>());
              }
            }
          }
        }

        hxm2.setZero();
        this->matrix_func2.right_jacobian_product(
            hxm2, jttemp.transpose(), jxm2, DirectAssignment(), std::bool_constant<false>());

        if constexpr (MatFunc2::InputIsDynamic) {
          const int sds = this->matrix_func2.SubDomains.cols();
          if (sds == 0) {
            adjhess += hxm2 + hxm2.transpose();
          } else {
            for (int i = 0; i < sds; i++) {
              int Start1 = this->matrix_func2.SubDomains(0, i);
              int Size1 = this->matrix_func2.SubDomains(1, i);
              adjhess.middleCols(Start1, Size1) += hxm2.middleCols(Start1, Size1);
              adjhess.middleRows(Start1, Size1) += hxm2.middleCols(Start1, Size1).transpose();
            }
          }
        } else {
          constexpr int sds = MatFunc2::INPUT_DOMAIN::SubDomains.size();
          ASSET::constexpr_for_loop(
              std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                constexpr int Start1 = MatFunc2::INPUT_DOMAIN::SubDomains[i.value][0];
                constexpr int Size1 = MatFunc2::INPUT_DOMAIN::SubDomains[i.value][1];
                adjhess.template middleCols<Size1>(Start1, Size1) +=
                    hxm2.template middleCols<Size1>(Start1, Size1);
                adjhess.template middleRows<Size1>(Start1, Size1) +=
                    hxm2.template middleCols<Size1>(Start1, Size1).transpose();
              });
        }
      };


      const int o1 = this->matrix_func1.ORows();
      const int o2 = this->matrix_func2.ORows();
      const int irows = this->IRows();
      const int orows = this->ORows();

      const int crit_size = std::max({o1, o2, orows, irows, m1rows});

      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  TempSpec<MatFunc1_Output<Scalar>>(o1, 1),
                                  TempSpec<MatFunc1_Output<Scalar>>(o1, 1),
                                  TempSpec<MatFunc1_jacobian<Scalar>>(o1, irows),

                                  TempSpec<MatFunc2_Output<Scalar>>(o2, 1),
                                  TempSpec<MatFunc2_Output<Scalar>>(o2, 1),
                                  TempSpec<MatFunc2_jacobian<Scalar>>(o2, irows),
                                  TempSpec<MatFunc2_gradient<Scalar>>(irows, 1),
                                  TempSpec<MatFunc2_hessian<Scalar>>(irows, irows),

                                  TempSpec<Output<Scalar>>(orows, 1),
                                  TempSpec<Eigen::Matrix<Scalar, M1Rows, 1>>(m1rows, 1),
                                  TempSpec<MatFunc2_jacobian<Scalar>>(o2, irows));
    }
  };


}  // namespace ASSET

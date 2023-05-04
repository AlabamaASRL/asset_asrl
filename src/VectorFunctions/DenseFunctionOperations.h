#pragma once

#include "AssigmentTypes.h"
#include "pch.h"

namespace ASSET {

  template<class Target, class Left, class Right, class Assignment, bool Aliased>
  void right_jacobian_product_impl(const Eigen::MatrixBase<Target>& target_,
                                   const Eigen::EigenBase<Left>& left,
                                   const Eigen::EigenBase<Right>& right,
                                   Assignment assign,
                                   std::bool_constant<Aliased> aliased) {
    Eigen::MatrixBase<Target>& target = target_.const_cast_derived();
    typedef typename Target::Scalar Scalar;
    if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
      if constexpr (Aliased) {
        target = left.derived() * right.derived();
      } else {
        target.noalias() = left.derived() * right.derived();
      }
    } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
      if constexpr (Aliased) {
        target += left.derived() * right.derived();
      } else {
        target.noalias() += left.derived() * right.derived();
      }
    } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
      if constexpr (Aliased) {
        target -= left.derived() * right.derived();
      } else {
        target.noalias() -= left.derived() * right.derived();
      }
    } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
      if constexpr (Aliased) {
        target = assign.value * left.derived() * right.derived();
      } else {
        target.noalias() = assign.value * left.derived() * right.derived();
      }
    } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
      if constexpr (Aliased) {
        target += assign.value * left.derived() * right.derived();
      } else {
        target.noalias() += assign.value * left.derived() * right.derived();
      }
    } else {
      std::cout << "right_jacobian_product has not been implemented for: "

                << std::endl;
    }
  }

  template<class INPUT_DOMAIN, class Target, class Left, class Right, class Assignment, bool Aliased>
  void right_jacobian_product_constant_impl(const INPUT_DOMAIN& SubDomains,
                                            const Eigen::MatrixBase<Target>& target_,
                                            const Eigen::EigenBase<Left>& left,
                                            const Eigen::EigenBase<Right>& right,
                                            Assignment assign,
                                            std::bool_constant<Aliased> aliased) {
    constexpr int sds = INPUT_DOMAIN::SubDomains.size();

    const Eigen::MatrixBase<Right>& right_ref(right.derived());
    Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

    ASSET::constexpr_for_loop(
        std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
          constexpr int Start1 = INPUT_DOMAIN::SubDomains[i.value][0];
          constexpr int Size1 = INPUT_DOMAIN::SubDomains[i.value][1];
          ASSET::right_jacobian_product_impl(target_ref.template middleCols<Size1>(Start1, Size1),
                                             left,
                                             right_ref.template middleCols<Size1>(Start1, Size1),
                                             assign,
                                             aliased);
        });
  }

  template<class Target, class Left, class Right, class Assignment, bool Aliased>
  void right_jacobian_product_dynamic_impl(const DomainMatrix& SubDomains,
                                           const Eigen::MatrixBase<Target>& target_,
                                           const Eigen::EigenBase<Left>& left,
                                           const Eigen::EigenBase<Right>& right,
                                           Assignment assign,
                                           std::bool_constant<Aliased> aliased) {
    const int sds = SubDomains.cols();

    if (sds == 0) {
      // ASSET::right_jacobian_product_impl(target_, left, right, assign, aliased);
    } else {
      const Eigen::MatrixBase<Right>& right_ref(right.derived());
      Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

      for (int i = 0; i < sds; i++) {
        int start = SubDomains(0, i);
        int size = SubDomains(1, i);

        ASSET::right_jacobian_product_impl(
            target_ref.middleCols(start, size), left, right_ref.middleCols(start, size), assign, aliased);
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////

  template<class Target, class Left, class Right, class Assignment, bool Aliased>
  void symetric_jacobian_product_impl(const Eigen::MatrixBase<Target>& target_,
                                      const Eigen::EigenBase<Left>& left,
                                      const Eigen::EigenBase<Right>& right,
                                      Assignment assign,
                                      std::bool_constant<Aliased> aliased) {
    Eigen::MatrixBase<Target>& target = target_.const_cast_derived();
    typedef typename Target::Scalar Scalar;
    if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
      if constexpr (Aliased) {
        target = right.derived().transpose() * left.derived() * right.derived();
      } else {
        target.noalias() = right.derived().transpose() * left.derived() * right.derived();
      }
    } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
      if constexpr (Aliased) {
        target += right.derived().transpose() * left.derived() * right.derived();
      } else {
        if constexpr (Left::MaxRowsAtCompileTime == 1 && Left::MaxColsAtCompileTime == 1) {
          target.noalias() += right.derived().transpose() * (Scalar(left.derived()[0]) * right.derived());

        } else {
          target.noalias() += right.derived().transpose() * left.derived() * right.derived();
        }
      }
    } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
      if constexpr (Aliased) {
        target -= right.derived().transpose() * left.derived() * right.derived();
      } else {
        target.noalias() -= right.derived().transpose() * left.derived() * right.derived();
      }
    } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
      if constexpr (Aliased) {
        target = assign.value * right.derived().transpose() * left.derived() * right.derived();
      } else {
        target.noalias() = assign.value * right.derived().transpose() * left.derived() * right.derived();
      }
    } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
      if constexpr (Aliased) {
        target += assign.value * right.derived().transpose() * left.derived() * right.derived();
      } else {
        target.noalias() += assign.value * right.derived().transpose() * left.derived() * right.derived();
      }
    } else {
      std::cout << "symetric_jacobian_product has not been implemented for: "

                << std::endl;
    }
  }

  template<class INPUT_DOMAIN, class Target, class Left, class Right, class Assignment, bool Aliased>
  void symetric_jacobian_product_constant_impl(const INPUT_DOMAIN& SubDomains,
                                               const Eigen::MatrixBase<Target>& target_,
                                               const Eigen::EigenBase<Left>& left,
                                               const Eigen::EigenBase<Right>& right,
                                               Assignment assign,
                                               std::bool_constant<Aliased> aliased) {
    constexpr int sds = INPUT_DOMAIN::SubDomains.size();

    const Eigen::MatrixBase<Right>& right_ref(right.derived());
    Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

    if constexpr (sds != 1) {
      ASSET::symetric_jacobian_product_impl(target_, left, right, assign, aliased);

    } else {
      ASSET::constexpr_for_loop(
          std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
            constexpr int Start1 = INPUT_DOMAIN::SubDomains[i.value][0];
            constexpr int Size1 = INPUT_DOMAIN::SubDomains[i.value][1];
            ASSET::symetric_jacobian_product_impl(
                target_ref.template block<Size1, Size1>(Start1, Start1, Size1, Size1),
                left,
                right_ref.template middleCols<Size1>(Start1, Size1),
                assign,
                aliased);
          });
    }
  }

  template<class Target, class Left, class Right, class Assignment, bool Aliased>
  void symetric_jacobian_product_dynamic_impl(const DomainMatrix& SubDomains,
                                              const Eigen::MatrixBase<Target>& target_,
                                              const Eigen::EigenBase<Left>& left,
                                              const Eigen::EigenBase<Right>& right,
                                              Assignment assign,
                                              std::bool_constant<Aliased> aliased) {
    const int sds = SubDomains.cols();

    if (sds == 0 || sds > 1) {
      ASSET::symetric_jacobian_product_impl(target_, left, right, assign, aliased);
    } else {
      const Eigen::MatrixBase<Right>& right_ref(right.derived());
      Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

      for (int i = 0; i < sds; i++) {
        int start = SubDomains(0, i);
        int size = SubDomains(1, i);


        ASSET::symetric_jacobian_product_impl(target_ref.block(start, start, size, size),
                                              left,
                                              right_ref.middleCols(start, size),
                                              assign,
                                              aliased);

        /*for (int j = i+1; j < sds; j++) {
            int start2 = SubDomains(0, j);
            int size2 = SubDomains(1, j);

            target_ref.block(start, start, size, size) +=
                right_ref.middleCols(start2, size2).transpose() * left.derived() *
        right_ref.middleCols(start, size);

        }*/
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////

  template<class Target, class JacType, class Assignment>
  void accumulate_impl(const Eigen::MatrixBase<Target>& target_,
                       const Eigen::MatrixBase<JacType>& right,
                       Assignment assign) {
    Eigen::MatrixBase<Target>& target = target_.const_cast_derived();
    typedef typename Target::Scalar Scalar;

    if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
      target = right;
    } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
      target += right;
    } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
      target -= right;
    } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
      target = assign.value * right.derived();
    } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
      target += assign.value * right.derived();
    }
  }

  template<class Target, class JacType, class Assignment>
  void accumulate_matrix_dynamic_domain_impl(const DomainMatrix& SubDomains,
                                             const Eigen::MatrixBase<Target>& target_,
                                             const Eigen::MatrixBase<JacType>& right,
                                             Assignment assign) {
    int sds = SubDomains.cols();
    if (sds == 0) {
      // ASSET::accumulate_impl(target_, right, assign);
    } else {
      const Eigen::MatrixBase<JacType>& right_ref(right.derived());
      Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

      for (int i = 0; i < sds; i++) {
        int start = SubDomains(0, i);
        int size = SubDomains(1, i);

        ASSET::accumulate_impl(target_ref.middleCols(start, size), right_ref.middleCols(start, size), assign);
      }
    }
  }

  template<class Target, class JacType, class Assignment>
  void accumulate_symetric_matrix_dynamic_domain_impl(const DomainMatrix& SubDomains,
                                                      const Eigen::MatrixBase<Target>& target_,
                                                      const Eigen::MatrixBase<JacType>& right,
                                                      Assignment assign) {
    int sds = SubDomains.cols();
    const Eigen::MatrixBase<JacType>& right_ref(right.derived());
    Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

    if (sds == 0) {
      // ASSET::accumulate_impl(target_, right, assign);
    } else if (sds == 1) {

      int Start1 = SubDomains(0, 0);
      int Size1 = SubDomains(1, 0);
      ASSET::accumulate_impl(target_ref.block(Start1, Start1, Size1, Size1),
                             right_ref.block(Start1, Start1, Size1, Size1),
                             assign);
    } else if (sds == 2) {

      int Start1 = SubDomains(0, 0);
      int Size1 = SubDomains(1, 0);
      int Start2 = SubDomains(0, 1);
      int Size2 = SubDomains(1, 1);

      ASSET::accumulate_impl(target_ref.block(Start1, Start1, Size1, Size1),
                             right_ref.block(Start1, Start1, Size1, Size1),
                             assign);

      ASSET::accumulate_impl(target_ref.block(Start2, Start1, Size2, Size1),
                             right_ref.block(Start2, Start1, Size2, Size1),
                             assign);

      ASSET::accumulate_impl(target_ref.block(Start1, Start2, Size1, Size2),
                             right_ref.block(Start1, Start2, Size1, Size2),
                             assign);

      ASSET::accumulate_impl(target_ref.block(Start2, Start2, Size2, Size2),
                             right_ref.block(Start2, Start2, Size2, Size2),
                             assign);


    } else {

      for (int i = 0; i < sds; i++) {
        int start = SubDomains(0, i);
        int size = SubDomains(1, i);

        ASSET::accumulate_impl(target_ref.middleCols(start, size), right_ref.middleCols(start, size), assign);
      }
    }
  }

  template<class Target, class JacType, class Assignment>
  void accumulate_vector_dynamic_domain_impl(const DomainMatrix& SubDomains,
                                             const Eigen::MatrixBase<Target>& target_,
                                             const Eigen::MatrixBase<JacType>& right,
                                             Assignment assign) {
    int sds = SubDomains.cols();
    if (sds == 0) {
      // ASSET::accumulate_impl(target_, right, assign);
    } else {
      const Eigen::MatrixBase<JacType>& right_ref(right.derived());
      Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

      for (int i = 0; i < sds; i++) {
        int start = SubDomains(0, i);
        int size = SubDomains(1, i);
        ASSET::accumulate_impl(target_ref.segment(start, size), right_ref.segment(start, size), assign);
      }
    }
  }

  template<class Target, class Scalar>
  void scale_vector_dynamic_domain_impl(const DomainMatrix& SubDomains,
                                        const Eigen::MatrixBase<Target>& target_,
                                        Scalar s) {
    int sds = SubDomains.cols();
    Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

    if (sds == 0) {
      target_ref *= s;
    } else {
      for (int i = 0; i < sds; i++) {
        int start = SubDomains(0, i);
        int size = SubDomains(1, i);
        target_ref.segment(start, size) *= s;
      }
    }
  }

  template<class Target, class Scalar>
  void scale_matrix_dynamic_domain_impl(const DomainMatrix& SubDomains,
                                        const Eigen::MatrixBase<Target>& target_,
                                        Scalar s) {
    int sds = SubDomains.cols();
    Eigen::MatrixBase<Target>& target_ref(target_.const_cast_derived());

    if (sds == 0) {
      target_ref *= s;
    } else {
      for (int i = 0; i < sds; i++) {
        int start = SubDomains(0, i);
        int size = SubDomains(1, i);
        target_ref.middleCols(start, size) *= s;
      }
    }
  }

}  // namespace ASSET

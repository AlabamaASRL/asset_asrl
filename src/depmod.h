// Dependency Modifications and Extensions
#pragma once

#include <Eigen/Core>
#include <boost/math/differentiation/autodiff.hpp>

// Use Boost Autodiff in Eigen
namespace Eigen {
template <typename RealType, size_t Order>
struct NumTraits<boost::math::differentiation::autodiff_v1::detail::
                     template fvar<RealType, Order>> : NumTraits<RealType> {
  using fvar =
      boost::math::differentiation::autodiff_v1::detail::template fvar<RealType,
                                                                       Order>;
  typedef fvar Real;

  enum {
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 16,
    MulCost = 16,
  };
};

#define BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(A)                                 \
  template <class RealType, size_t Order, typename BinaryOp>                  \
  struct ScalarBinaryOpTraits<boost::math::differentiation::autodiff_v1::     \
                                  detail::template fvar<RealType, Order>,     \
                              A, BinaryOp> {                                  \
    typedef boost::math::differentiation::autodiff_v1::detail::template fvar< \
        RealType, Order>                                                      \
        ReturnType;                                                           \
  };                                                                          \
  template <class RealType, size_t Order, typename BinaryOp>                  \
  struct ScalarBinaryOpTraits<A,                                              \
                              boost::math::differentiation::autodiff_v1::     \
                                  detail::template fvar<RealType, Order>,     \
                              BinaryOp> {                                     \
    typedef boost::math::differentiation::autodiff_v1::detail::template fvar< \
        RealType, Order>                                                      \
        ReturnType;                                                           \
  };

BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(float);
BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(double);
BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(long double);
BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(short);
BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(unsigned short);
BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(int);
BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(unsigned int);
BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(long);
BOOST_AUTODIFF_EIGEN_SCALAR_TRAITS(unsigned long);

}  // namespace Eigen
